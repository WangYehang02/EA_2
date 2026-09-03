#!/usr/bin/env python
"""Precision ablation smokes E (BF16 fwd) and F (FP32 fwd). Independent, sequential.

Both start from frozen epoch_2.copy.pt. FP32 loss, FP32 eval, DataParallel 4x8.
Does not overwrite the frozen FP16 SMOKE.FAILED scene. Does not start v4.
Does not write a resume checkpoint. Intermediate *.pt are forensic-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import resolve_instance_root, save_json
from earthquake.stage10.amp_forward import batch_loss_amp_forward_fp32_nll
from earthquake.stage10.dkpn_clean import build_dkpn_random
from earthquake.stage10.eval_fp32 import eval_dev_fp32
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage10.nonfinite import NonfiniteError
from earthquake.stage10.diag_infer import collapse_flag, psn_distribution as psn_dist
from earthquake.stage10.partial_label import partial_nll_numerator
from earthquake.stage10.state_finite import audit_model_opt
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr
from earthquake.stage10.train_monitor import ActivationMonitor, divergence_report, psn_means, snapshot_norms
from earthquake.stage10.v3_continue import OVERFLOW_FLAG
from earthquake.utils import ensure_dir

RUN = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
FORE = RUN / "forensic_nonfinite_e3"
SMOKE_EF = FORE / "smoke_ef"
PARENT_SHA = "6df929b9ee72665f8094bfe643750fa7134ebf4402fe126d14ec07d694a95222"
FAIL_BI = 2411
SAME_GPUS = (1, 3, 6, 7)


def _eval(model, loader, device):
    return eval_dev_fp32(
        model,
        loader,
        device,
        pick_metrics=pick_metrics_at_thr,
        psn_dist=psn_dist,
        collapse_flag=collapse_flag,
        height=OFFICIAL_HEIGHT,
    )


def eval_brief(vm: dict) -> dict:
    m = vm.get("metrics_height_0p2") or {}
    return {
        "val_loss": vm.get("val_loss"),
        "f1@0.5s": m.get("f1@0.5s", vm.get("f1")),
        "precision@0.5s": m.get("precision@0.5s"),
        "recall@0.5s": m.get("recall@0.5s"),
        "picks_per_trace": m.get("picks_per_trace"),
        "frac_no_s_peak": m.get("frac_no_s_peak"),
        "oracle@0.5s": vm.get("oracle_argmax_accuracy@0.5s"),
        "true_s_probability": vm.get("true_s_probability"),
        "local_s_peak_probability": vm.get("local_s_peak_probability"),
        "psn": vm.get("psn"),
        "all_n_collapse": vm.get("all_n_collapse"),
        "eval_dtype": vm.get("eval_dtype"),
    }


def _kw(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}


def dump_forensic(path: Path, *, model, opt, batch, extra: dict) -> None:
    raw = model.module if hasattr(model, "module") else model
    cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()} if batch else None
    grads = {n: (p.grad.detach().cpu() if p.grad is not None else None) for n, p in raw.named_parameters()}
    torch.save(
        {
            "not_for_resume": True,
            "forensic_only": True,
            "model": raw.state_dict(),
            "opt": opt.state_dict(),
            "grads": grads,
            "batch": cpu_batch,
            **extra,
        },
        path,
    )


def run_plan(
    *,
    plan: str,
    steps: int,
    gpus: list[int],
    v2h,
    cat,
    val_loader,
    wave,
    device,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)
    if plan == "E":
        amp, dtype, tag = True, torch.bfloat16, "bf16_forward_fp32_loss"
    elif plan == "F":
        amp, dtype, tag = False, None, "fp32_forward_fp32_loss"
    else:
        raise ValueError(plan)
    ck = torch.load(FORE / "epoch_2.copy.pt", map_location="cpu", weights_only=False)
    model = build_dkpn_random().to(device)
    model.load_state_dict(ck["model"])
    nvis = torch.cuda.device_count()
    if nvis != 4:
        raise SystemExit(f"plan {plan} requires 4 visible GPUs, have {nvis}")
    model = torch.nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.load_state_dict(ck["opt"])
    start_audit = audit_model_opt(model, opt)
    save_json({"when": "load_epoch2_copy", "not_for_resume": True, **start_audit}, out_dir / "state_at_load.json")
    mon = ActivationMonitor()
    mon.attach(model)
    model.eval()
    e0 = _eval(model, val_loader, device)
    model.train()
    evals = [{"when": "resume_before_train", **eval_brief(e0)}]
    ld, ds = v2h.make_loader(
        cat, wave, batch=32, workers=4, augment=True, seed=42 + 3,
        shuffle=False, balanced=True, num_samples=len(cat), drop_last=True,
    )
    ds.set_virtual_epoch(3)
    series: list[dict] = []
    last_loss = None
    last_batch = None
    bg_pseudo = 0
    n_skip = 0
    passed_fail_bi = False
    bi = -1
    jsonl = (out_dir / "monitor.jsonl").open("w", encoding="utf-8")
    try:
        for bi, batch in enumerate(ld):
            if bi >= steps:
                break
            last_batch = batch
            opt.zero_grad(set_to_none=True)
            want_mon = (bi % 100 == 0)
            if want_mon:
                mon.begin()
            try:
                loss, logits = batch_loss_amp_forward_fp32_nll(
                    model, batch, device, amp=amp, dtype=dtype, optimizer_step=int(ck["optimizer_steps"]) + bi
                )
            except NonfiniteError as e:
                if want_mon:
                    mon.end()
                dump_forensic(
                    out_dir / "FAILURE.forensic.pt",
                    model=model,
                    opt=opt,
                    batch=batch,
                    extra={"plan": plan, "bi": bi, "phase": e.phase, "meta": e.to_dict()},
                )
                fail = {
                    "ok": False,
                    "plan": plan,
                    "bi": bi,
                    "phase": e.phase,
                    "optimizer_step_called_this_batch": False,
                    "state_at_fail": audit_model_opt(model, opt),
                    **e.to_dict(),
                    "not_for_resume": True,
                    "confirm_read": False,
                }
                save_json(fail, out_dir / "SMOKE.FAILED.json")
                print(json.dumps({"plan": plan, "failed": True, "bi": bi, "phase": e.phase}), flush=True)
                return fail
            finally:
                act_sum = mon.summary() if want_mon else None
                if want_mon:
                    mon.end()
            loss.backward()
            ginfo = audit_model_opt(model, opt)["gradients"]
            if not ginfo["finite"]:
                dump_forensic(
                    out_dir / "FAILURE.forensic.pt",
                    model=model,
                    opt=opt,
                    batch=batch,
                    extra={"plan": plan, "bi": bi, "phase": "backward"},
                )
                fail = {
                    "ok": False,
                    "plan": plan,
                    "bi": bi,
                    "phase": "backward",
                    "state_at_fail": audit_model_opt(model, opt),
                    "not_for_resume": True,
                }
                save_json(fail, out_dir / "SMOKE.FAILED.json")
                return fail
            gnorm = ginfo["l2_norm"]
            opt.step()
            last_loss = float(loss.detach().cpu())
            for i, k in enumerate(batch["crop_kind"]):
                if k == "background" and not bool(batch["is_noise"][i]) and float(batch["n_pos"][i].sum()) > 1e-6:
                    bg_pseudo += 1
            if bi == FAIL_BI:
                passed_fail_bi = True
            if want_mon:
                kw = _kw(batch, device)
                _, _, terms = partial_nll_numerator(logits.float(), **kw)
                n_mod = int((act_sum or {}).get("n_modules") or 0)
                row = {
                    "bi": bi,
                    "loss": last_loss,
                    "finite": True,
                    "grad_norm": gnorm,
                    "monitor_empty": n_mod == 0,
                    "act_abs_max": (act_sum or {}).get("worst_abs_max"),
                    "act_worst_module": (act_sum or {}).get("worst_module"),
                    "n_modules_nonfinite": (act_sum or {}).get("n_modules_nonfinite"),
                    "loss_terms": {k: float(v.detach().cpu()) for k, v in terms.items()},
                    **psn_means(logits),
                    **snapshot_norms(model, opt),
                    "nan_inf": False,
                }
                series.append(row)
                jsonl.write(json.dumps(row) + "\n")
                jsonl.flush()
                print(json.dumps({"plan": plan, "bi": bi, "target": steps, "loss": last_loss, "act_abs_max": row["act_abs_max"]}), flush=True)
        jsonl.close()
        jsonl = None
        if bi < FAIL_BI:
            fail = {"ok": False, "plan": plan, "reason": "did_not_pass_bi_2411", "bi": bi, "not_for_resume": True}
            save_json(fail, out_dir / "SMOKE.FAILED.json")
            return fail
        passed_fail_bi = True
        model.eval()
        e1 = _eval(model, val_loader, device)
        model.train()
        evals.append({"when": "after_train", "steps": bi + 1, **eval_brief(e1)})
        # one more train step after second eval
        opt.zero_grad(set_to_none=True)
        post_batch = last_batch
        try:
            loss2, logits2 = batch_loss_amp_forward_fp32_nll(
                model, post_batch, device, amp=amp, dtype=dtype, optimizer_step=int(ck["optimizer_steps"]) + bi + 1
            )
            loss2.backward()
            opt.step()
            post_ok = True
            post_loss = float(loss2.detach().cpu())
        except NonfiniteError as e:
            dump_forensic(
                out_dir / "FAILURE.forensic.pt",
                model=model,
                opt=opt,
                batch=post_batch,
                extra={"plan": plan, "bi": "post_eval_train_step", "phase": e.phase, "meta": e.to_dict()},
            )
            fail = {"ok": False, "plan": plan, "phase": e.phase, "when": "post_eval_train_step", **e.to_dict(), "not_for_resume": True}
            save_json(fail, out_dir / "SMOKE.FAILED.json")
            return fail
        dump_forensic(
            out_dir / "last_state.forensic.pt",
            model=model,
            opt=opt,
            batch=post_batch,
            extra={"plan": plan, "bi": bi, "not_for_resume": True, "parent_sha256": PARENT_SHA},
        )
        div = divergence_report(series)
        f1s = [e.get("f1@0.5s") for e in evals if e.get("f1@0.5s") is not None]
        no_cat = (min(f1s) > 0.05) if f1s else False
        report = {
            "ok": True,
            "plan": plan,
            "tag": tag,
            "amp": amp,
            "dtype": str(dtype) if dtype is not None else "float32",
            "grad_scaler": False,
            "eval_dtype": "fp32",
            "parent_sha256": PARENT_SHA,
            "gpus": gpus,
            "nvis": nvis,
            "global_batch": 32,
            "per_rank_batch": 8,
            "dataparallel": 4,
            "steps_requested": steps,
            "steps_run": bi + 1,
            "passed_bi_2411": passed_fail_bi,
            "last_loss": last_loss,
            "post_eval_train_step_ok": post_ok,
            "post_eval_train_loss": post_loss,
            "bg_pseudo_n": bg_pseudo,
            "silent_skips": n_skip,
            "n_evals": len(evals),
            "evals": evals,
            "divergence": div,
            "no_catastrophic_f1_drop": no_cat,
            "stable": bool(div.get("stable") and no_cat and bg_pseudo == 0 and n_skip == 0),
            "not_for_resume": True,
            "not_30_epoch": True,
            "did_not_reset_bn": True,
            "did_not_switch_ddp": True,
            "confirm_read": False,
            "v4_not_started": True,
            "monitor_points": series,
        }
        save_json(report, out_dir / "SMOKE.REPORT.json")
        passed = report["ok"] and report["stable"] and report["passed_bi_2411"] and report["n_evals"] >= 2
        (out_dir / ("SMOKE.PASSED" if passed else "SMOKE.UNSTABLE")).write_text(json.dumps({"ok": passed, "stable": report["stable"]}, indent=2))
        if not passed:
            save_json({**report, "ok": False}, out_dir / "SMOKE.FAILED.json")
        print(json.dumps({"plan": plan, "passed": passed, "stable": report["stable"], "steps": bi + 1}, indent=2), flush=True)
        return report
    finally:
        if jsonl is not None:
            jsonl.close()
        mon.remove()
        del model
        del opt
        torch.cuda.empty_cache()


def compare_reports(e: dict, f: dict) -> dict:
    def f1(rep, i=0):
        ev = (rep.get("evals") or [{}])
        if i >= len(ev):
            return None
        return ev[i].get("f1@0.5s")

    e_st = bool(e.get("stable"))
    f_st = bool(f.get("stable"))
    if e_st and f_st:
        choice = "prefer_bf16_train_fp32_eval"
        reason = "both_stable"
    elif f_st and not e_st:
        choice = "fp32_train_only"
        reason = "only_fp32_stable"
    elif e_st and not f_st:
        choice = "unexpected_bf16_only"
        reason = "fp32_unstable_do_not_start_v4"
    else:
        choice = "neither"
        reason = "both_real_divergence_or_overflow_audit_lr_grad_opt_no_clip_resume"
    e0, e1 = f1(e, 0), f1(e, 1)
    f0, f1b = f1(f, 0), f1(f, 1)
    direction = None
    if None not in (e0, e1, f0, f1b):
        direction = "same" if (e1 - e0) * (f1b - f0) >= 0 else "opposite"
    return {
        "bf16_stable": e_st,
        "fp32_stable": f_st,
        "choice": choice,
        "reason": reason,
        "f1_direction": direction,
        "bf16_evals": e.get("evals"),
        "fp32_evals": f.get("evals"),
        "bf16_divergence": e.get("divergence"),
        "fp32_divergence": f.get("divergence"),
        "do_not_resume_epoch2": True,
        "v4_not_started": True,
        "v4_requires_user_approval": True,
        "mixed_fp16_then_bf16_not_paper_main": True,
        "confirm_read": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,3,6,7")
    ap.add_argument("--plans", default="E,F")
    ap.add_argument("--steps-e", type=int, default=5000)
    ap.add_argument("--steps-f", type=int, default=3000)
    args = ap.parse_args()
    ensure_dir(SMOKE_EF)
    frozen = FORE / "smoke" / "SMOKE.FAILED.FROZEN"
    if not frozen.is_file():
        save_json({"ok": False, "reason": "fp16_smoke_failed_not_frozen"}, SMOKE_EF / "BLOCKED.json")
        raise SystemExit(3)
    src = FORE / "smoke" / "SMOKE.FAILED.json"
    if src.is_file() and os.access(src, os.W_OK):
        save_json({"ok": False, "reason": "frozen_SMOKE.FAILED_is_writable"}, SMOKE_EF / "BLOCKED.json")
        raise SystemExit(3)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if tuple(gpus) != SAME_GPUS:
        save_json({"ok": False, "reason": "not_same_4gpu_set", "requested": gpus, "required": list(SAME_GPUS)}, SMOKE_EF / "BLOCKED.json")
        raise SystemExit(3)
    idle = idle_gpu_indices()
    not_idle = [g for g in gpus if g not in idle]
    if not_idle or len(gpus) != 4:
        save_json(
            {"ok": False, "reason": "requested_gpus_not_idle", "gpus": gpus, "not_idle": not_idle, "idle": idle, "confirm_read": False},
            SMOKE_EF / "BLOCKED.json",
        )
        print(json.dumps({"blocked": True, "not_idle": not_idle, "idle": idle}))
        raise SystemExit(3)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
    spec = importlib.util.spec_from_file_location("v2h", ROOT / "scripts/train_stage10_dkpn_v2.py")
    v2h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2h)
    import pandas as pd
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    cat = v2h.annotate_online_catalog(pd.read_parquet(RUN / "train_online_catalog.parquet"))
    val_cat = v2h.annotate_online_catalog(pd.read_parquet(RUN / "val_subset.parquet"))
    val_cat = val_cat.copy()
    val_cat["crop_kind"] = "s_centered"
    device = torch.device("cuda:0")
    val_loader, _ = v2h.make_loader(
        val_cat, wave, batch=32, workers=0, augment=False, seed=42, shuffle=False, balanced=False
    )
    plans = [p.strip().upper() for p in args.plans.split(",") if p.strip()]
    reports: dict[str, dict] = {}
    for plan in plans:
        out_dir = SMOKE_EF / f"plan_{plan}"
        steps = int(args.steps_e if plan == "E" else args.steps_f)
        reports[plan] = run_plan(
            plan=plan, steps=steps, gpus=gpus, v2h=v2h, cat=cat, val_loader=val_loader, wave=wave, device=device, out_dir=out_dir
        )
    if "E" in reports and "F" in reports:
        cmp = compare_reports(reports["E"], reports["F"])
        save_json(cmp, SMOKE_EF / "COMPARE.json")
        print(json.dumps(cmp, indent=2), flush=True)
    save_json(
        {
            "plans": list(reports),
            "ok_flags": {k: bool(v.get("ok") and v.get("stable")) for k, v in reports.items()},
            "v4_not_started": True,
            "do_not_resume_epoch2": True,
            "overflow_flag": OVERFLOW_FLAG,
            "confirm_read": False,
        },
        SMOKE_EF / "SUMMARY.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
