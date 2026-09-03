#!/usr/bin/env python
"""4-GPU full-lifecycle smoke: train AMP+FP32 loss, FP32 eval twice, no 30-epoch resume.

Does not overwrite the frozen v3 scene. Does not steal occupied GPUs.
"""

from __future__ import annotations

import argparse
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
from earthquake.stage10.eval_fp32 import eval_dev_fp32, eval_forward_amp_fp16_diagnostic, eval_forward_fp32
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage10.nonfinite import NonfiniteError
from earthquake.stage10.diag_infer import collapse_flag, psn_distribution as psn_dist
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr
from earthquake.utils import ensure_dir

RUN = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
FORE = RUN / "forensic_nonfinite_e3"
SMOKE = FORE / "smoke"
PAST = 1000
PARENT_SHA = "6df929b9ee72665f8094bfe643750fa7134ebf4402fe126d14ec07d694a95222"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--steps-past-fail", type=int, default=PAST)
    ap.add_argument("--min-gpus", type=int, default=4, help="4 = official smoke; 2 = diagnostic only, does not unlock resume")
    args = ap.parse_args()
    ensure_dir(SMOKE)
    frozen = SMOKE / "SMOKE.FAILED.FROZEN"
    if frozen.is_file() or (SMOKE / "SMOKE.FAILED.json").is_file():
        print(
            json.dumps(
                {
                    "refused": True,
                    "reason": "refusing_overwrite_frozen_SMOKE.FAILED",
                    "frozen_marker": str(frozen),
                    "use": "scripts/smoke_stage10_v3_precision_ef.py",
                }
            )
        )
        raise SystemExit(5)
    idle = idle_gpu_indices()
    meta_p = FORE / "offending_batch_meta.json"
    fail_bi = json.loads(meta_p.read_text())["batch_index"] if meta_p.is_file() else 0
    target = int(fail_bi) + int(args.steps_past_fail)
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = idle[: max(int(args.min_gpus), 4)]
    not_idle = [g for g in gpus if g not in idle]
    diagnostic = int(args.min_gpus) < 4 or len(gpus) < 4
    if not_idle:
        save_json(
            {
                "ok": False,
                "reason": "requested_gpus_not_idle",
                "gpus": gpus,
                "not_idle": not_idle,
                "idle": idle,
                "confirm_read": False,
            },
            SMOKE / "SMOKE.BLOCKED.json",
        )
        print(json.dumps({"blocked": True, "not_idle": not_idle, "idle": idle}))
        raise SystemExit(3)
    if len(gpus) < int(args.min_gpus):
        save_json(
            {
                "ok": False,
                "reason": f"need_{args.min_gpus}_idle_gpus_have_{len(gpus)}",
                "idle": idle,
                "note": "will not steal occupied GPUs",
                "confirm_read": False,
                "lifecycle": "resume→train AMP fwd→FP32 loss→backward/opt→eval FP32→train",
            },
            SMOKE / "SMOKE.BLOCKED.json",
        )
        print(json.dumps({"blocked": True, "idle": idle}))
        raise SystemExit(3)
    if 32 % len(gpus) != 0:
        raise SystemExit(f"global batch 32 not divisible by nvis={len(gpus)}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
    import importlib.util

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
    ck = torch.load(FORE / "epoch_2.copy.pt", map_location="cpu", weights_only=False)
    model = build_dkpn_random().to(device)
    model.load_state_dict(ck["model"])
    nvis = torch.cuda.device_count()
    if nvis >= 2:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.load_state_dict(ck["opt"])
    ld, ds = v2h.make_loader(
        cat, wave, batch=32, workers=4, augment=True, seed=42 + 3,
        shuffle=False, balanced=True, num_samples=len(cat), drop_last=True,
    )
    ds.set_virtual_epoch(3)
    val_loader, _ = v2h.make_loader(
        val_cat, wave, batch=32, workers=0, augment=False, seed=42, shuffle=False, balanced=False
    )

    # Old vs new eval on the first train batch (no param update).
    contrast = {"old_fp16_eval": None, "new_fp32_eval": None}
    first_batch = next(iter(ld))
    ds.set_virtual_epoch(3)
    try:
        model.eval()
        eval_forward_amp_fp16_diagnostic(model, first_batch["x"], device)
        contrast["old_fp16_eval"] = {"ok": True, "unexpected": "fp16_eval_did_not_overflow"}
    except NonfiniteError as e:
        contrast["old_fp16_eval"] = {"ok": False, "phase": e.phase, "first_module": e.context.get("first_nonfinite_module"), "expected_fail": True}
    try:
        logits = eval_forward_fp32(model, first_batch["x"], device)
        contrast["new_fp32_eval"] = {"ok": bool(torch.isfinite(logits).all()), "logits_max": float(logits.detach().float().max().cpu())}
    except NonfiniteError as e:
        save_json({"ok": False, "reason": "fp32_eval_failed_same_batch", "diagnostic_2gpu": diagnostic, **e.to_dict()}, SMOKE / "SMOKE.FAILED.json")
        raise SystemExit(4)
    finally:
        model.train()

    evals = []
    model.eval()
    e0 = _eval(model, val_loader, device)
    model.train()
    evals.append({"when": "resume_before_train", **{k: e0[k] for k in ("val_loss", "f1") if k in e0}})

    ld, ds = v2h.make_loader(
        cat, wave, batch=32, workers=4, augment=True, seed=42 + 3,
        shuffle=False, balanced=True, num_samples=len(cat), drop_last=True,
    )
    ds.set_virtual_epoch(3)
    bg_pseudo = 0
    n_skip = 0
    last_loss = None
    mid_eval = None
    for bi, batch in enumerate(ld):
        opt.zero_grad(set_to_none=True)
        try:
            loss, logits = batch_loss_amp_forward_fp32_nll(
                model, batch, device, amp=True, dtype=torch.float16, optimizer_step=71076 + bi
            )
        except NonfiniteError as e:
            cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.save({"batch": cpu_batch, "batch_index": bi, "meta": e.to_dict()}, FORE / "offending_batch.pt")
            save_json({"ok": False, "bi": bi, "phase": e.phase, **e.to_dict()}, SMOKE / "SMOKE.FAILED.json")
            raise SystemExit(4)
        loss.backward()
        g = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
        if not torch.isfinite(g):
            save_json({"ok": False, "bi": bi, "phase": "backward"}, SMOKE / "SMOKE.FAILED.json")
            raise SystemExit(4)
        opt.step()
        last_loss = float(loss.detach().cpu())
        for i, k in enumerate(batch["crop_kind"]):
            if k == "background" and not bool(batch["is_noise"][i]) and float(batch["n_pos"][i].sum()) > 1e-6:
                bg_pseudo += 1
        if bi % 50 == 0:
            print(json.dumps({"bi": bi, "target": target, "loss": last_loss, "nvis": nvis}), flush=True)
        if bi >= target:
            break
    model.eval()
    e1 = _eval(model, val_loader, device)
    model.train()
    evals.append({"when": "after_train_past_fail_plus_1000", "steps": bi + 1, **{k: e1[k] for k in ("val_loss", "f1") if k in e1}})

    # Continue a few train steps after eval to close the lifecycle.
    opt.zero_grad(set_to_none=True)
    loss, logits = batch_loss_amp_forward_fp32_nll(model, first_batch, device, amp=True, dtype=torch.float16)
    loss.backward()
    opt.step()

    f1s = [e["f1"] for e in evals if e.get("f1") is not None]
    report = {
        "ok": True,
        "parent_sha256": PARENT_SHA,
        "fail_batch_index_hint": fail_bi,
        "steps_run": bi + 1,
        "steps_past_fail": (bi + 1) - int(fail_bi),
        "last_loss": last_loss,
        "bg_pseudo_n": bg_pseudo,
        "silent_skips": n_skip,
        "nvis": nvis,
        "gpus": gpus,
        "optimizer_updated": True,
        "evals": evals,
        "n_evals": len(evals),
        "eval_contrast": contrast,
        "eval_fp16_reproduced": contrast["old_fp16_eval"] is not None and contrast["old_fp16_eval"].get("expected_fail") is True,
        "eval_fp32_same_batch_ok": bool(contrast["new_fp32_eval"] and contrast["new_fp32_eval"].get("ok")),
        "no_catastrophic_f1_drop": (min(f1s) > 0.05) if f1s else False,
        "confirm_read": False,
        "not_30_epoch": True,
        "did_not_reset_bn": True,
        "did_not_switch_ddp": True,
        "global_batch": 32,
        "dataparallel": nvis,
        "per_rank_batch": 32 // max(nvis, 1),
        "diagnostic_not_official_4gpu": diagnostic,
        "official_resume_still_blocked": True,
        "bn_semantics_note": "2-GPU is 16/replica; official resume remains 4x8. This run does not unlock RESUME.GATE.",
    }
    tag = "2gpu" if diagnostic else "4gpu"
    save_json(report, SMOKE / f"SMOKE.{tag}.REPORT.json")
    if not diagnostic:
        save_json(report, SMOKE / "SMOKE.REPORT.json")
    passed = (
        report["ok"]
        and bg_pseudo == 0
        and n_skip == 0
        and report["n_evals"] >= 2
        and report["eval_fp32_same_batch_ok"]
        and report["eval_fp16_reproduced"]
    )
    out_name = f"SMOKE.{tag}." + ("PASSED" if passed else "FAILED")
    (SMOKE / out_name).write_text(json.dumps(report, indent=2))
    if not diagnostic:
        (SMOKE / ("SMOKE.PASSED" if passed else "SMOKE.FAILED")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
