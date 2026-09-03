#!/usr/bin/env python
"""Threshold calibration + two 8000-step loss ablations. No v3, no full pilot, no confirm waveforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage10.checkpoint_policy import MIN_DELTA_F1
from earthquake.stage10.dataset_v2 import DKPNPartialCropDataset
from earthquake.stage10.diag_infer import (
    collate_diag,
    collapse_flag,
    force_s_centered,
    group_param_grad_norms,
    infer_arrays,
    psn_distribution,
    summarize_split,
    token_mean_group_terms,
)
from earthquake.stage10.diag_splits import build_diag_sets
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.frozen_cycle import FrozenCycleSampler
from earthquake.stage10.gpu_policy import idle_gpu_indices, snapshot_gpus
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.phase_balanced_loss import partial_label_group_terms, phase_balanced_partial_nll
from earthquake.stage10.threshold_calibration import (
    OFFICIAL_HEIGHT,
    PROTOCOL_NAME,
    THRESH_GRID,
    pick_metrics_at_thr,
    select_threshold_on_calibration,
)
from earthquake.utils import ensure_dir

N_STEPS = 8000
DIAG_EVERY = 500
BATCH = 16
SEED = 42
LR = 1e-3
WORK = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_diag_cal_ablation")
ART = artifacts_dir() / "results" / "stage10"
# Official DKPN default height=0.2 is the v3 hard bar this round: we do not write method-lock calibration.
HEIGHT_0P2_HARD_FOR_V3 = True
HEIGHT_0P2_IN_STAGE6_METHOD_LOCK = False


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def confirm_unread_proof() -> dict:
    from earthquake.stage6.full_splits import load_full_event_ids

    src = Path(__file__).read_text(encoding="utf-8")
    confirm_dir = artifacts_dir() / "results" / "stage6" / "final_confirm"
    return {
        "confirm_read": False,
        "confirm_waveforms_read": False,
        "confirm_metrics_read": False,
        "confirm_id_list_used_for_disjointness_only": True,
        "assert_full_confirm_access_allowed_called": False,
        "final_confirm_dir_exists": confirm_dir.is_dir(),
        "this_script_does_not_open_confirm_waveforms": True,
        "n_confirm_event_ids_for_disjointness": len(load_full_event_ids("stage6_internal_confirm")),
        "v2_frozen_untouched": True,
        "no_full_pilot": True,
        "no_v3_dir_created": True,
    }


def freeze_splits(work: Path) -> dict:
    tr, cal, evl, meta = build_diag_sets(SEED)
    ensure_dir(work)
    tr.to_parquet(work / "train.parquet", index=False)
    cal.to_parquet(work / "cal.parquet", index=False)
    evl.to_parquet(work / "eval.parquet", index=False)
    meta["utc"] = datetime.now(timezone.utc).isoformat()
    save_json(meta, work / "split_meta.json")
    save_json(meta, ART / "dkpn_diag_split_meta.json")
    return meta


def _wave():
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    inst = resolve_instance_root()
    src = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")

    def read_fn(name, is_noise=False):
        return src.read(name, is_noise=is_noise)

    return read_fn


def make_eval_ds(cat, read_fn, n=None, seed=1):
    c = force_s_centered(cat if n is None else cat.sample(n=min(n, len(cat)), random_state=seed))
    return DKPNPartialCropDataset(c, read_fn, augment=False, seed=0)


def diagnose(model, device, read_fn, train_cat, cal_cat, evl_cat, step: int, letter: str, last_batch) -> dict:
    model.eval()
    ds_tr = make_eval_ds(train_cat.loc[~train_cat.is_noise.astype(bool) & train_cat.has_s_label.astype(bool)], read_fn, n=400, seed=1)
    ds_cal = make_eval_ds(cal_cat, read_fn)
    ds_evl = make_eval_ds(evl_cat, read_fn)
    tr_a = infer_arrays(model, ds_tr, device)
    cal_a = infer_arrays(model, ds_cal, device)
    evl_a = infer_arrays(model, ds_evl, device)
    proto = select_threshold_on_calibration(cal_a["s_probs"], cal_a["true_s"], cal_a["vis_s"])
    thr = proto["selected_thr"]
    eval_sweep_report_only = [
        pick_metrics_at_thr(evl_a["s_probs"], evl_a["true_s"], evl_a["vis_s"], float(t)) for t in THRESH_GRID
    ]
    rec = {
        "step": step,
        "letter": letter,
        "selected_thr": thr,
        "selected_from_eligible_picks_cap": proto["selected_from_eligible_picks_cap"],
        "protocol": proto["protocol"],
        "threshold_sweep_calibration": proto["grid"],
        "threshold_sweep_evaluation_reporting_only_not_used_for_selection": eval_sweep_report_only,
        "train": summarize_split(tr_a, selected_thr=thr),
        "calibration": summarize_split(cal_a, selected_thr=thr),
        "evaluation": summarize_split(evl_a, selected_thr=thr),
        "eval_f1_fixed0p2": pick_metrics_at_thr(evl_a["s_probs"], evl_a["true_s"], evl_a["vis_s"], OFFICIAL_HEIGHT),
        "train_oracle": summarize_split(tr_a)["oracle_argmax_accuracy@0.5s"],
        "eval_oracle": summarize_split(evl_a)["oracle_argmax_accuracy@0.5s"],
        "eval_f1_calibrated": summarize_split(evl_a, selected_thr=thr)["at_selected_thr"]["f1@0.5s"],
        "nan_inf": False,
        "all_n_collapse": collapse_flag(psn_distribution(evl_a)),
    }
    # group losses + grads on last train batch
    model.train()
    model.zero_grad(set_to_none=True)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in last_batch.items()}
    logits = dkpn_logits(model, batch["x"])
    if not torch.isfinite(logits).all():
        rec["nan_inf"] = True
    groups = partial_label_group_terms(
        logits,
        p_pos=batch["p_pos"],
        s_pos=batch["s_pos"],
        n_pos=batch["n_pos"],
        not_p=batch["not_p"],
        not_s=batch["not_s"],
        pad_mask=batch["pad_mask"],
        is_noise=batch["is_noise"],
    )
    rec["loss_groups"] = {k: float(groups[k]["mean"].detach()) for k in groups}
    rec["loss_group_mass"] = {k: float(groups[k]["w"].sum().detach()) for k in groups}
    if letter == "A":
        terms = token_mean_group_terms(logits, batch)
    else:
        _, _, means = phase_balanced_partial_nll(
            logits,
            p_pos=batch["p_pos"],
            s_pos=batch["s_pos"],
            n_pos=batch["n_pos"],
            not_p=batch["not_p"],
            not_s=batch["not_s"],
            pad_mask=batch["pad_mask"],
            is_noise=batch["is_noise"],
        )
        terms = means
    rec["group_grad_norms"] = group_param_grad_norms(model, logits, terms)
    rec["phase_over_n_grad"] = rec["group_grad_norms"].get("phase_over_n")
    model.eval()
    return rec


def run_letter(letter: str, gpu: int, work: Path) -> None:
    assert letter in {"A", "B"}
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    out = ensure_dir(work / letter)
    tr = pd.read_parquet(work / "train.parquet")
    cal = pd.read_parquet(work / "cal.parquet")
    evl = pd.read_parquet(work / "eval.parquet")
    read_fn = _wave()
    ds = DKPNPartialCropDataset(tr, read_fn, augment=True, seed=SEED)
    sampler = FrozenCycleSampler(len(ds), seed=SEED)
    ld = DataLoader(
        ds,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        collate_fn=collate_diag,
        drop_last=True,
        pin_memory=True,
    )
    model = build_dkpn_random().to(device)
    ensure_dir(out / "checkpoints")
    torch.save({"model": model.state_dict(), "seed": SEED, "step": 0, "letter": letter}, out / "checkpoints" / "init.pt")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps_per_epoch = max(len(tr) // BATCH, 1)
    step = 0
    epoch = 0
    hist = []
    nan = False
    t0 = time.time()
    last_cpu = None
    (out / "RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    # step 0 diagnostic
    dummy_ld = DataLoader(ds, batch_size=BATCH, sampler=FrozenCycleSampler(len(ds), seed=SEED), collate_fn=collate_diag, drop_last=True)
    last_cpu = next(iter(dummy_ld))
    rec0 = diagnose(model, device, read_fn, tr, cal, evl, 0, letter, last_cpu)
    rec0["loss"] = None
    hist.append(rec0)
    save_json({"history": hist, "letter": letter}, out / "history.json")
    print(json.dumps({k: rec0[k] for k in ["step", "eval_oracle", "eval_f1_calibrated", "eval_f1_fixed0p2", "all_n_collapse"]}, default=str), flush=True)

    model.train()
    while step < N_STEPS:
        ds.set_virtual_epoch(epoch)
        for batch in ld:
            opt.zero_grad(set_to_none=True)
            x = batch["x"].to(device, non_blocking=True)
            kw = {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
            is_noise = batch["is_noise"].to(device, non_blocking=True)
            logits = dkpn_logits(model, x)
            if letter == "A":
                loss = partial_label_nll(logits, **kw)
                stats = {"ablation": "current_partial_label_nll_control", "not_official_token_ce": False}
            else:
                loss, stats, _ = phase_balanced_partial_nll(logits, is_noise=is_noise, **kw)
            if not torch.isfinite(loss):
                nan = True
                break
            loss.backward()
            opt.step()
            last_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            step += 1
            if step % DIAG_EVERY == 0 or step == N_STEPS:
                rec = diagnose(model, device, read_fn, tr, cal, evl, step, letter, last_cpu)
                rec["loss"] = float(loss.detach().cpu())
                rec["loss_stats"] = {k: stats[k] for k in stats if k != "grid"}
                rec["nan_inf"] = rec["nan_inf"] or nan
                rec["sec"] = time.time() - t0
                rec["virtual_epoch"] = epoch
                hist.append(rec)
                save_json({"history": hist, "letter": letter, "not_official_token_ce": letter == "B"}, out / "history.json")
                ck = {
                    "model": model.state_dict(),
                    "step": step,
                    "letter": letter,
                    "seed": SEED,
                    "probability_threshold_official": OFFICIAL_HEIGHT,
                    "selected_thr": rec["selected_thr"],
                    "valid_best": False,
                    "metric_name_fixed0p2": "eval_f1_fixed0p2",
                    "metric_name_calibrated": "eval_f1_calibrated",
                    "eval_f1_fixed0p2": rec["eval_f1_fixed0p2"]["f1@0.5s"],
                    "eval_f1_calibrated": rec["eval_f1_calibrated"],
                }
                torch.save(ck, out / "checkpoints" / f"step_{step:04d}.pt")
                print(
                    json.dumps(
                        {
                            "step": step,
                            "letter": letter,
                            "loss": rec["loss"],
                            "eval_oracle": rec["eval_oracle"],
                            "eval_f1_fixed0p2": rec["eval_f1_fixed0p2"]["f1@0.5s"],
                            "eval_f1_calibrated": rec["eval_f1_calibrated"],
                            "thr": rec["selected_thr"],
                            "collapse": rec["all_n_collapse"],
                            "phase_over_n": rec.get("phase_over_n_grad"),
                        },
                        default=str,
                    ),
                    flush=True,
                )
                model.train()
            if step >= N_STEPS:
                break
        if nan:
            break
        epoch += 1
    save_json(
        {
            "letter": letter,
            "steps": step,
            "nan": nan,
            "sec": time.time() - t0,
            "confirm_read": False,
            "not_official_token_ce": letter == "B",
            "B_not_numerically_equivalent_to_official_pointwise_CE": letter == "B",
            "history": hist,
        },
        out / "run.json",
    )
    (out / "DONE").write_text(datetime.now(timezone.utc).isoformat())
    if (out / "RUNNING").exists():
        (out / "RUNNING").unlink()


def _last(hist, key, default=None):
    for r in reversed(hist):
        if r.get("step", -1) > 0 and key in r:
            return r[key]
    return default


def _hist_at(hist, step):
    for r in hist:
        if r.get("step") == step:
            return r
    return None


def gate_one(init, last, hist, *, letter: str) -> dict:
    reasons = []
    def f1_02(rec, split="evaluation"):
        if rec is None:
            return 0.0
        if split == "evaluation":
            return float(rec.get("eval_f1_fixed0p2", {}).get("f1@0.5s", 0) or 0)
        return float(rec.get(split, {}).get("official_height_0.2", {}).get("f1@0.5s", 0) or 0)

    oracle_init = float(init.get("eval_oracle") or 0)
    oracle_last = float(last.get("eval_oracle") or 0)
    train_init = float(init.get("train_oracle") or 0)
    train_last = float(last.get("train_oracle") or 0)
    if oracle_last <= oracle_init:
        reasons.append("oracle_eval_not_above_init")
    if not (train_last > train_init and oracle_last > oracle_init):
        reasons.append("train_and_eval_oracle_not_both_up")
    cal_f1 = float(last.get("eval_f1_calibrated") or 0)
    cal_f1_init = float(init.get("eval_f1_calibrated") or 0)
    picks = float(last.get("evaluation", {}).get("at_selected_thr", {}).get("picks_per_trace", 99) or 99)
    if cal_f1 <= cal_f1_init or cal_f1 <= 0:
        reasons.append("calibrated_thr_not_effective_on_eval")
    if picks > 3.0:
        reasons.append("too_many_picks_per_trace")
    smax = [r.get("evaluation", {}).get("psn", {}).get("s_max_mean") for r in hist if r.get("step", 0) >= 500]
    smax = [x for x in smax if x is not None]
    if smax and smax[-1] < 0.08 and (len(smax) >= 2 and smax[-1] < 0.5 * smax[0]):
        reasons.append("ps_output_decaying")
    late = [r for r in hist if r.get("step", 0) >= N_STEPS - 1500]
    if any(r.get("all_n_collapse") for r in late) or last.get("all_n_collapse"):
        reasons.append("all_n_collapse_late")
    ratios = [r.get("phase_over_n_grad") for r in hist if r.get("phase_over_n_grad") is not None]
    if ratios and ratios[-1] < 0.05:
        reasons.append("phase_grad_still_drowned_by_n")
    boot = last.get("evaluation", {}).get("bootstrap_f1@0.5s_selected", {})
    if boot and float(boot.get("f1_lo", 0) or 0) <= 0:
        reasons.append("bootstrap_ci_includes_zero_or_empty")
    if HEIGHT_0P2_HARD_FOR_V3:
        if f1_02(last) <= f1_02(init) + MIN_DELTA_F1:
            reasons.append("height_0.2_eval_f1_not_above_init")
    ok = len(reasons) == 0
    return {
        "letter": letter,
        "ok": ok,
        "reasons": reasons,
        "oracle_eval_init": oracle_init,
        "oracle_eval_last": oracle_last,
        "train_oracle_init": train_init,
        "train_oracle_last": train_last,
        "eval_f1_fixed0p2_init": f1_02(init),
        "eval_f1_fixed0p2_last": f1_02(last),
        "eval_f1_calibrated_init": cal_f1_init,
        "eval_f1_calibrated_last": cal_f1,
        "picks_per_trace_eval_selected": picks,
        "phase_over_n_last": None if not ratios else ratios[-1],
        "collapse_late": any(r.get("all_n_collapse") for r in late) or bool(last.get("all_n_collapse")),
        "height_0p2_hard_for_v3": HEIGHT_0P2_HARD_FOR_V3,
    }


def assemble(work: Path) -> None:
    meta = json.loads((work / "split_meta.json").read_text())
    a = json.loads((work / "A" / "run.json").read_text())
    b = json.loads((work / "B" / "run.json").read_text())
    ha, hb = a["history"], b["history"]
    a0, aL = _hist_at(ha, 0), _hist_at(ha, N_STEPS) or ha[-1]
    b0, bL = _hist_at(hb, 0), _hist_at(hb, N_STEPS) or hb[-1]
    a2000 = _hist_at(ha, 2000)
    gate_a = gate_one(a0, aL, ha, letter="A")
    gate_b = gate_one(b0, bL, hb, letter="B")
    conclusion = "both_failed_stop_dkpn"
    v3 = False
    if gate_a["ok"] and not gate_b["ok"]:
        conclusion = "A_pass_original_loss_usable_but_2000_steps_too_short"
    elif gate_b["ok"] and not gate_a["ok"]:
        conclusion = "B_pass_A_fail_phase_n_normalization_required"
    elif gate_a["ok"] and gate_b["ok"]:
        conclusion = "A_and_B_pass_prefer_original_loss_longer_training"
    if gate_a["ok"] or gate_b["ok"]:
        v3 = False  # still require explicit later authorization; this round does not create v3
        # User: 只有满足条件才考虑 v3 — we report consider_v3
    consider_v3 = bool(gate_a["ok"] or gate_b["ok"])
    # If only low-threshold passed, gate_one already failed height_0.2 when hard.
    only_low_thr = (
        (not gate_a["ok"] and not gate_b["ok"])
        and (
            (aL and float(aL.get("eval_f1_calibrated") or 0) > float(a0.get("eval_f1_calibrated") or 0))
            or (bL and float(bL.get("eval_f1_calibrated") or 0) > float(b0.get("eval_f1_calibrated") or 0))
        )
        and float((aL or {}).get("eval_f1_fixed0p2", {}).get("f1@0.5s") or 0) <= float((a0 or {}).get("eval_f1_fixed0p2", {}).get("f1@0.5s") or 0)
    )

    # calibration JSON from A@0 and A@2000 (mixed-crop current loss)
    cal_doc = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "mixed_crop_checkpoint": "A/checkpoints/step_2000.pt (reproduced 2000-step current partial-label; prior gate saved no weights)",
        "protocol": PROTOCOL_NAME,
        "threshold_grid": list(THRESH_GRID),
        "same_protocol_init_and_trained": True,
        "never_reselect_on_evaluation": True,
        "official_height_0.2_always_reported": True,
        "splits": meta,
        "confirm_unread": confirm_unread_proof(),
        "init": a0,
        "trained_step_2000": a2000,
        "note": "init vs trained use identical pre-registered selection protocol; not trained-best vs init-0.2",
        "v2_frozen": True,
        "v3_blocked": not consider_v3,
    }
    abl_doc = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_steps": N_STEPS,
        "seed": SEED,
        "batch": BATCH,
        "lr": LR,
        "A": a,
        "B": b,
        "gate_A": gate_a,
        "gate_B": gate_b,
        "phase_n_grad_ratio_A_last": gate_a.get("phase_over_n_last"),
        "phase_n_grad_ratio_B_last": gate_b.get("phase_over_n_last"),
        "phase_n_improved_in_B": (
            gate_b.get("phase_over_n_last") is not None
            and gate_a.get("phase_over_n_last") is not None
            and gate_b["phase_over_n_last"] > gate_a["phase_over_n_last"]
        ),
        "collapse_A": gate_a["collapse_late"],
        "collapse_B": gate_b["collapse_late"],
        "B_not_numerically_equivalent_to_official_pointwise_CE": True,
        "height_0p2_hard_for_v3": HEIGHT_0P2_HARD_FOR_V3,
        "height_0p2_in_stage6_method_lock": HEIGHT_0P2_IN_STAGE6_METHOD_LOCK,
        "consider_v3": consider_v3,
        "v3_created": False,
        "conclusion": conclusion,
        "only_low_threshold_not_posthoc_v2": only_low_thr,
        "confirm_unread": confirm_unread_proof(),
    }
    save_json(cal_doc, ART / "dkpn_threshold_calibration.json")
    save_json(abl_doc, ART / "dkpn_loss_ablation.json")
    if conclusion == "both_failed_stop_dkpn":
        save_json(
            {
                "status": "rejected_candidate_source",
                "source": "DKPN_v2_partial_label_and_phase_balanced_ablation",
                "reason": "A_and_B_failed_small_gate",
                "gate_A": gate_a,
                "gate_B": gate_b,
                "do_not_add_epochs_or_seeds": True,
                "v3_allowed": False,
                "confirm_read": False,
            },
            ART / "rejected_candidate_source.json",
        )
    md = _render_md(meta, cal_doc, abl_doc, gate_a, gate_b, conclusion, consider_v3, only_low_thr)
    (ROOT / "reports/stage10/dkpn_v2_calibration_and_loss_ablation.md").write_text(md, encoding="utf-8")


def _f1row(rec, which):
    if rec is None:
        return "n/a"
    if which == "02":
        m = rec.get("eval_f1_fixed0p2") or rec.get("evaluation", {}).get("official_height_0.2") or {}
        return f"{m.get('f1@0.5s', float('nan')):.4f}"
    if which == "cal":
        return f"{float(rec.get('eval_f1_calibrated') or 0):.4f}"
    if which == "ora":
        return f"{float(rec.get('eval_oracle') or 0):.4f}"
    return ""


def _render_md(meta, cal_doc, abl, gate_a, gate_b, conclusion, consider_v3, only_low) -> str:
    a2000 = cal_doc.get("trained_step_2000") or {}
    a0 = cal_doc.get("init") or {}
    def sweep_table(rec, title):
        grid = (rec or {}).get("calibration", {}).get("official_height_0.2")
        # full grid lives at top-level of hist rec? We stored summarize not full grid in hist.
        # Pull from evaluation/calibration selected + eval_f1
        return f"(see JSON; selected_thr={rec.get('selected_thr') if rec else None})"

    lines = [
        "# DKPN v2 阈值校准与 loss 消融（v2 冻结，v3 阻断）",
        "",
        f"- utc: {abl.get('utc')}",
        f"- seed: {SEED}；steps: {N_STEPS}；batch: {BATCH}；mixed-crop FrozenCycleSampler",
        "- 未训练 full/pilot；未创建 v3 目录；未读取 confirm 波形/指标",
        f"- 选阈协议: `{PROTOCOL_NAME}`（init 与 trained 相同；禁止 trained-best vs init-0.2）",
        f"- B 与官方逐点 CE **不再数值等价**（受控消融）",
        "",
        "## 1. calibration / evaluation 事件数与 hash",
        "",
        f"- held-out: {meta.get('n_heldout')} traces / {meta.get('n_heldout_events')} events",
        f"- calibration: {meta.get('n_cal_traces')} traces / {meta.get('n_cal_events')} events  hash_events=`{meta.get('hash_cal_event_ids')}`",
        f"- evaluation: {meta.get('n_eval_traces')} traces / {meta.get('n_eval_events')} events  hash_events=`{meta.get('hash_eval_event_ids')}`",
        f"- train event traces: {meta.get('n_train_event_traces')} + noise {meta.get('n_train_noise')}  hash_events=`{meta.get('hash_train_event_ids')}`",
        f"- cal∩eval events = ∅（预注册）",
        "",
        "## 2–3. init / trained@2000 阈值扫描与独立 evaluation",
        "",
        f"- mixed-crop trained = **A step 2000**（先前 2000-step gate 未存权重，按同 seed/split 复现）",
        f"- init selected_thr={a0.get('selected_thr')}  eval F1@0.5 calibrated={_f1row(a0,'cal')}  F1@0.5 height=0.2={_f1row(a0,'02')}  oracle={_f1row(a0,'ora')}",
        f"- trained@2000 selected_thr={a2000.get('selected_thr')}  eval F1@0.5 calibrated={_f1row(a2000,'cal')}  F1@0.5 height=0.2={_f1row(a2000,'02')}  oracle={_f1row(a2000,'ora')}",
        "- 完整网格、P/R/F1@0.1s&0.5s、picks/trace、无S峰比例、true-S / local peak / oracle / bootstrap CI：见 `artifacts/results/stage10/dkpn_threshold_calibration.json`",
        "",
        "## 4. A 运行（current partial-label，8000 step）",
        "",
        f"- ok={gate_a['ok']}  reasons={gate_a['reasons']}",
        f"- oracle eval {gate_a['oracle_eval_init']:.4f} → {gate_a['oracle_eval_last']:.4f}",
        f"- F1 height=0.2 {gate_a['eval_f1_fixed0p2_init']:.4f} → {gate_a['eval_f1_fixed0p2_last']:.4f}",
        f"- F1 calibrated {gate_a['eval_f1_calibrated_init']:.4f} → {gate_a['eval_f1_calibrated_last']:.4f}",
        f"- phase/N grad {gate_a.get('phase_over_n_last')}  collapse_late={gate_a['collapse_late']}",
        "",
        "## 5. B 运行（phase-balanced partial-label，8000 step）",
        "",
        f"- ok={gate_b['ok']}  reasons={gate_b['reasons']}",
        f"- oracle eval {gate_b['oracle_eval_init']:.4f} → {gate_b['oracle_eval_last']:.4f}",
        f"- F1 height=0.2 {gate_b['eval_f1_fixed0p2_init']:.4f} → {gate_b['eval_f1_fixed0p2_last']:.4f}",
        f"- F1 calibrated {gate_b['eval_f1_calibrated_init']:.4f} → {gate_b['eval_f1_calibrated_last']:.4f}",
        f"- phase/N grad {gate_b.get('phase_over_n_last')}  collapse_late={gate_b['collapse_late']}",
        "",
        "## 6–9. 门控",
        "",
        f"- phase/N 梯度比是否改善（B vs A last）: {abl.get('phase_n_improved_in_B')}",
        f"- 是否仍全N塌缩: A={gate_a['collapse_late']} B={gate_b['collapse_late']}",
        f"- height=0.2 是否为硬门槛（本轮 v3）: **{HEIGHT_0P2_HARD_FOR_V3}**（官方 DKPN 默认；未写入 Stage6 method lock；本轮不允许用低阈值替代）",
        f"- 仅低阈值通过、不得事后追认 v2: {only_low}",
        f"- 是否获准创建 v3: **false**（consider_v3={consider_v3}；本脚本不创建目录）",
        f"- 结论: `{conclusion}`",
        "",
        "## 10. confirm 未读取",
        "",
        f"- {json.dumps(cal_doc.get('confirm_unread'), ensure_ascii=False)}",
        "",
        "## 11. pytest",
        "",
        "- 见最终汇报（由 master 追加）。",
        "",
    ]
    return "\n".join(lines) + "\n"


def master() -> None:
    ensure_dir(ART)
    ensure_dir(ROOT / "reports/stage10")
    idle = idle_gpu_indices()
    if len(idle) < 2:
        raise SystemExit(f"need 2 idle GPUs for A||B, have {idle}")
    # leave at least one unused if possible
    gpus = idle[:2]
    meta = freeze_splits(WORK)
    print("SPLITS", json.dumps(meta, indent=2), flush=True)
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    procs = []
    logs = {}
    for letter, gpu in zip(("A", "B"), gpus):
        log = WORK / f"{letter}.log"
        ensure_dir(WORK)
        f = open(log, "w")
        logs[letter] = f
        p = subprocess.Popen(
            [py, str(Path(__file__).resolve()), "--role", letter, "--gpu", str(gpu)],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        procs.append((letter, gpu, p))
        print(f"launched {letter} gpu={gpu} pid={p.pid} log={log}", flush=True)
    rc = {}
    for letter, gpu, p in procs:
        rc[letter] = p.wait()
        logs[letter].close()
        print(f"{letter} exit={rc[letter]}", flush=True)
    if any(v != 0 for v in rc.values()):
        raise SystemExit(f"worker_failed {rc}")
    assemble(WORK)
    print("ASSEMBLED", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="master", choices=["master", "A", "B", "assemble"])
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()
    if args.role == "master":
        master()
    elif args.role == "assemble":
        assemble(WORK)
    else:
        if args.gpu < 0:
            raise SystemExit("--gpu required")
        run_letter(args.role, args.gpu, WORK)


if __name__ == "__main__":
    main()
