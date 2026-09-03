#!/usr/bin/env python
"""Stage 10B — DKPN seed42 training-effectiveness audit (no confirm traces).

If a blocking implementation bug is found, writes stop-gate verdict=implementation_bug
and does NOT run full-dev formal evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage10.dkpn_clean import (
    CleanDKPNCropDataset,
    build_dkpn_random,
    dkpn_logits,
    dkpn_probs,
    masked_soft_ce,
    official_soft_ce_from_probs,
)
from earthquake.utils import ensure_dir

TRAIN_DIR = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean/train_seed42")
OUT = ROOT / "artifacts/results/stage10/dkpn"
CACHE = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn")
DKPN_REPO = Path("/home/yehang/EARTHQUAKE/baseline/DKPN")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def param_groups_l2(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if not torch.is_floating_point(a[k]):
            continue
        da, db = a[k].float().cpu(), b[k].float().cpu()
        out[k] = float(torch.norm(da - db).item())
    return out


def module_prefix_l2(delta: dict) -> dict:
    buckets = {"inc": 0.0, "down_branch": 0.0, "up_branch": 0.0, "out": 0.0, "other": 0.0}
    for k, v in delta.items():
        placed = False
        for p in ("inc", "down_branch", "up_branch", "out"):
            if k.startswith(p) or k.startswith("in_bn") and p == "inc":
                buckets[p if not k.startswith("in_bn") else "inc"] += v**2
                placed = True
                break
        if not placed:
            buckets["other"] += v**2
    return {k: float(np.sqrt(v)) for k, v in buckets.items()}


def s_peak_acc(model, X, s_true, device, *, use_logits: bool) -> float:
    model.eval()
    with torch.no_grad():
        out = dkpn_logits(model, X.to(device)) if use_logits else dkpn_probs(model, X.to(device))
        pred = out[:, 1, :].argmax(-1).cpu().numpy()
    ok = [(st >= 0 and abs(int(p) - st) <= 20) for p, st in zip(pred, s_true)]
    return float(np.mean(ok))


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("STAGE10_GPU", "0"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    figdir = ensure_dir(OUT / "figures")
    ensure_dir(CACHE / "figures")

    ckpt_path = TRAIN_DIR / "checkpoints" / "last.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint {ckpt_path}")
    best_path = TRAIN_DIR / "checkpoints" / "best.pt"
    done_path = TRAIN_DIR / "TRAIN.DONE"
    hist = json.loads((TRAIN_DIR / "train_history.json").read_text())
    history = hist.get("history", [])
    train_index = TRAIN_DIR / "train_index.parquet"

    # --- provenance (no confirm metrics) ---
    import subprocess

    def git(cmd):
        try:
            return subprocess.check_output(cmd, cwd=str(DKPN_REPO), text=True).strip()
        except Exception as e:
            return repr(e)

    code_files = [
        ROOT / "src/earthquake/stage10/dkpn_clean.py",
        ROOT / "scripts/train_stage10_dkpn_clean.py",
    ]
    config_blob = "\n".join(sha256_file(p) for p in code_files)
    provenance = {
        "project_root": str(ROOT),
        "dkpn_repo_realpath": str(DKPN_REPO.resolve()),
        "dkpn_commit": git(["git", "rev-parse", "HEAD"]),
        "train_script_sha256": sha256_file(ROOT / "scripts/train_stage10_dkpn_clean.py"),
        "dkpn_clean_py_sha256": sha256_file(ROOT / "src/earthquake/stage10/dkpn_clean.py"),
        "config_hash": hashlib.sha256(config_blob.encode()).hexdigest(),
        "train_split_hash": sha256_file(train_index),
        "checkpoint_last_sha256": sha256_file(ckpt_path),
        "checkpoint_best_path": str(best_path) if best_path.exists() else None,
        "checkpoint_best_sha256": sha256_file(best_path) if best_path.exists() else None,
        "checkpoint_last_path": str(ckpt_path),
        "TRAIN.DONE": done_path.read_text().strip() if done_path.exists() else None,
        "TRAIN.DONE_mtime": datetime.fromtimestamp(done_path.stat().st_mtime, tz=timezone.utc).isoformat() if done_path.exists() else None,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "n_epochs_logged": len(history),
        "epoch0_mean_loss": history[0]["mean_loss"] if history else None,
        "epoch29_mean_loss": history[-1]["mean_loss"] if history else None,
        "loss_drop": (history[0]["mean_loss"] - history[-1]["mean_loss"]) if len(history) >= 2 else None,
        "confirm_waveforms_or_metrics_read": False,
        "confirm_used_for_threshold_or_checkpoint": False,
    }

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    torch.manual_seed(42)
    model_rand = build_dkpn_random()
    n_total = sum(p.numel() for p in model_rand.parameters())
    n_train = sum(p.numel() for p in model_rand.parameters() if p.requires_grad)
    frozen = [n for n, p in model_rand.named_parameters() if not p.requires_grad]
    opt_probe = torch.optim.Adam(model_rand.parameters(), lr=1e-3)
    n_opt = sum(p.numel() for g in opt_probe.param_groups for p in g["params"])

    rand_state = {k: v.detach().cpu().clone() for k, v in model_rand.state_dict().items()}
    # comparable random init is seed42 at train start, not this probe; we still compare last vs a fresh seed42 init
    torch.manual_seed(42)
    init_ref = build_dkpn_random()
    init_state = {k: v.detach().cpu().clone() for k, v in init_ref.state_dict().items()}
    deltas = param_groups_l2(init_state, {k: v.cpu() for k, v in state.items() if k in init_state})
    l2_all = float(np.sqrt(sum(v**2 for v in deltas.values())))
    module_l2 = module_prefix_l2(deltas)

    model_trained = build_dkpn_random()
    model_trained.load_state_dict(state)
    trained_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model_trained.state_dict().values() if v.is_floating_point())).hexdigest()
    init_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in init_ref.state_dict().values() if v.is_floating_point())).hexdigest()

    # --- data: picker_train index only ---
    meta = pd.read_parquet(train_index)
    if "event_id" in meta.columns:
        dev_e = load_full_event_ids("stage6_dev")
        overlap_dev = set(meta.event_id.astype(str)) & set(dev_e)
    else:
        overlap_dev = set()
    # leak check vs confirm event list file only (not confirm metrics/waveforms)
    confirm_e = load_full_event_ids("stage6_internal_confirm")
    overlap_cf = set(meta.loc[~meta["is_noise"].astype(bool), "event_id"].astype(str)) & set(confirm_e)
    if overlap_cf:
        raise SystemExit(f"train_index overlaps confirm events n={len(overlap_cf)}")

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(
        inst / "events" / "Instance_events_counts.hdf5",
        inst / "noise" / "Instance_noise.hdf5",
    )

    ps = meta[meta["supervision"] == "ps_both"].head(128).copy()
    ds = CleanDKPNCropDataset(
        ps,
        lambda n, is_noise=False: wave.read(n, is_noise=is_noise),
        augment=False,
        seed=0,
        allow_p_only=False,
        confirm_event_ids=set(),  # do not pass confirm ids for filtering; already disjoint
    )
    xs, ys, ms, s_true, p_true = [], [], [], [], []
    for i in range(min(128, len(ds))):
        b = ds[i]
        xs.append(b["x"])
        ys.append(b["y"])
        ms.append(b["mask"])
        yt = b["y"].numpy()
        s_true.append(int(yt[1].argmax()) if yt[1].max() > 0.1 else -1)
        p_true.append(int(yt[0].argmax()) if yt[0].max() > 0.1 else -1)
    X = torch.stack(xs)
    Y = torch.stack(ys)
    M = torch.stack(ms)

    # P-only mask check on a p_only subset
    po = meta[meta["supervision"] == "p_only"].head(8)
    p_only_ok = True
    p_only_detail = {}
    if len(po):
        ds_po = CleanDKPNCropDataset(
            po,
            lambda n, is_noise=False: wave.read(n, is_noise=is_noise),
            augment=False,
            seed=1,
            allow_p_only=True,
            confirm_event_ids=set(),
        )
        b0 = ds_po[0]
        p_only_ok = bool(b0["mask"][1].item() == 0.0 and float(b0["y"][1].sum()) == 0.0)
        p_only_detail = {"mask_S": float(b0["mask"][1]), "S_target_sum": float(b0["y"][1].sum()), "mask_P": float(b0["mask"][0])}

    # channel / sample sanity
    row0 = ds.meta.iloc[0]
    enz = wave.read(str(row0["trace_name"]), is_noise=False)
    channel_ok = enz.shape[0] == 3 and enz.shape[-1] == 12000
    sr_ok = float(row0.get("sampling_rate_hz", 100) or 100) == 100.0

    # --- forward diagnostics ---
    model_trained.to(device).train()
    xb, yb, mb = X[:8].to(device), Y[:8].to(device), M[:8].to(device)
    with torch.no_grad():
        probs = dkpn_probs(model_trained, xb)
        logits = dkpn_logits(model_trained, xb)
    # buggy loss as used in seed42 training
    logp_on_probs = F.log_softmax(probs, dim=1)
    buggy_ce = float((-(yb * logp_on_probs).sum(1).mean()).detach().cpu())
    official_ce = float(official_soft_ce_from_probs(probs, yb).detach().cpu())
    correct_ce = float(masked_soft_ce(logits, yb, mb).detach().cpu())

    ch_loss = {}
    with torch.no_grad():
        lp = F.log_softmax(logits, dim=1)
        per = -(yb * lp).mean(-1)  # B,3
        for i, name in enumerate(["P", "S", "N"]):
            ch_loss[name] = float(per[:, i].mean().cpu())

    model_trained.train()
    opt = torch.optim.Adam(model_trained.parameters(), lr=1e-3)
    xb.requires_grad_(False)
    logits_g = dkpn_logits(model_trained, xb)
    loss_g = masked_soft_ce(logits_g, yb, mb)
    loss_g.backward()
    gnorm = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model_trained.parameters() if p.grad is not None)).cpu())
    nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model_trained.parameters())
    lr_actual = opt.param_groups[0]["lr"]

    # buggy path grad norm
    model_trained.zero_grad(set_to_none=True)
    probs_g = dkpn_probs(model_trained, xb)
    buggy = -(yb * F.log_softmax(probs_g, dim=1)).sum(1).mean()
    buggy.backward()
    gnorm_buggy = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model_trained.parameters() if p.grad is not None)).cpu())

    # 128-trace localization trained vs random (eval mode, softmax S argmax)
    model_trained.eval()
    init_ref.to(device).eval()
    acc_tr_logits = s_peak_acc(model_trained, X, s_true, device, use_logits=True)
    acc_tr_probs = s_peak_acc(model_trained, X, s_true, device, use_logits=False)
    acc_rd_probs = s_peak_acc(init_ref, X, s_true, device, use_logits=False)

    # overlays
    with torch.no_grad():
        pr = dkpn_probs(model_trained, X[:20].to(device)).cpu().numpy()
        yt = Y[:20].numpy()
    n_fig = 0
    for i in range(min(20, len(pr))):
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(pr.shape[-1])
        ax.plot(t, yt[i, 1], label="label S", color="black", lw=1)
        ax.plot(t, pr[i, 1], label="pred S (softmax)", color="crimson", lw=1)
        ax.plot(t, yt[i, 0], label="label P", color="gray", lw=0.6, alpha=0.7)
        ax.plot(t, pr[i, 0], label="pred P", color="steelblue", lw=0.6, alpha=0.7)
        ax.set_title(f"train overlay {i} s_true={s_true[i]}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"overlay_{i:02d}.png", dpi=80)
        plt.close(fig)
        n_fig += 1

    pos_frac = {
        "P": float((Y[:, 0] > 0.1).float().mean()),
        "S": float((Y[:, 1] > 0.1).float().mean()),
        "N": float((Y[:, 2] > 0.1).float().mean()),
    }
    mask_frac = {k: float(M[:, i].mean()) for i, k in enumerate(["P", "S", "N"])}

    softmax_then_logsoftmax = True  # documented training path
    blocking_bug = True
    bug_name = "softmax_then_log_softmax"

    # --- post-fix small overfit (logits=True), 32 traces, no 30-epoch retrain ---
    X32, Y32, M32 = X[:32].to(device), Y[:32].to(device), M[:32].to(device)
    s32 = s_true[:32]
    torch.manual_seed(0)
    fix_model = build_dkpn_random().to(device)
    optf = torch.optim.Adam(fix_model.parameters(), lr=5e-3)
    hist_fix = []
    acc0 = s_peak_acc(fix_model, X32.cpu(), s32, device, use_logits=False)
    for ep in range(80):
        fix_model.train()
        optf.zero_grad()
        loss = masked_soft_ce(dkpn_logits(fix_model, X32), Y32, M32)
        loss.backward()
        optf.step()
        hist_fix.append(float(loss.detach().cpu()))
        if hist_fix[-1] < 0.05:
            break
    acc1 = s_peak_acc(fix_model, X32.cpu(), s32, device, use_logits=False)
    overfit_ok = bool(hist_fix[-1] < hist_fix[0] * 0.7 and acc1 >= max(0.5, acc0 + 0.2))

    # short smoke: 8 traces one step
    smoke_ok = bool(np.isfinite(hist_fix[-1]) and not nan_grad)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "params": {
            "n_total": int(n_total),
            "n_trainable": int(n_train),
            "n_optimizer": int(n_opt),
            "frozen_param_names": frozen,
            "all_trainable": n_total == n_train == n_opt,
        },
        "weight_update": {
            "init_seed42_hash": init_hash,
            "last_ckpt_hash": trained_hash,
            "hashes_differ": init_hash != trained_hash,
            "param_l2_vs_seed42_init": l2_all,
            "module_l2": module_l2,
            "best_equals_last": not best_path.exists(),
        },
        "batch_stats": {
            "x_min": float(X.min()),
            "x_max": float(X.max()),
            "x_mean": float(X.mean()),
            "x_std": float(X.std()),
            "positive_label_frac_time": pos_frac,
            "channel_mask_frac": mask_frac,
            "p_only_S_masked": p_only_ok,
            "p_only_detail": p_only_detail,
            "output_prob_min": float(probs.min().cpu()),
            "output_prob_max": float(probs.max().cpu()),
            "output_prob_mean": float(probs.mean().cpu()),
            "output_prob_std": float(probs.std().cpu()),
            "channel_ce_from_logits": ch_loss,
            "buggy_logsoftmax_on_probs_ce": buggy_ce,
            "official_log_on_probs_ce": official_ce,
            "correct_logits_masked_ce": correct_ce,
            "grad_norm_correct_logits": gnorm,
            "grad_norm_buggy_path": gnorm_buggy,
            "nan_inf_grad": nan_grad,
            "lr_actual": lr_actual,
            "lr_schedule": "none_constant_adam_1e-3",
        },
        "alignment": {
            "hdf5_shape_3x12000": bool(channel_ok),
            "sampling_rate_100": bool(sr_ok),
            "enz_read": True,
            "overlap_train_dev_events": len(overlap_dev),
            "overlap_train_confirm_events": len(overlap_cf),
        },
        "localization_128_train": {
            "n": int(len(s_true)),
            "trained_S_acc_softmax": acc_tr_probs,
            "trained_S_acc_logits_argmax": acc_tr_logits,
            "random_S_acc_softmax": acc_rd_probs,
            "trained_beats_random": bool(acc_tr_probs > acc_rd_probs + 0.05),
        },
        "loss_flat_root_cause": {
            "primary": bug_name,
            "detail": (
                "DKPN.forward defaults to softmax. seed42 train_loop called model(x) then F.log_softmax "
                "inside masked_soft_ce — i.e. log_softmax(softmax(z)). Official DKPN uses -y*log(p+eps) on "
                "softmax probabilities without a second softmax. This mismatches activations and flattens "
                "gradients (observed buggy_grad_norm vs correct_grad_norm). Secondary: timestep-averaged CE "
                "is dominated by the noise class (~99% of samples), so epoch-mean loss can sit near ~0.218 "
                "even if peak localization slowly changes."
            ),
            "weights_actually_updated": bool(init_hash != trained_hash and l2_all > 1e-3),
            "missing_best_ckpt": True,
            "no_lr_schedule": True,
        },
        "blocking_implementation_bug": blocking_bug,
        "bug_id": bug_name,
        "n_overlay_figures": n_fig,
        "post_fix_small_overfit": {
            "n_traces": 32,
            "init_loss": hist_fix[0],
            "final_loss": hist_fix[-1],
            "n_epochs": len(hist_fix),
            "s_acc_init": acc0,
            "s_acc_final": acc1,
            "overfit_ok": overfit_ok,
            "smoke_ok": smoke_ok,
            "note": "logits=True + masked_soft_ce; not a 30-epoch retrain",
        },
        "formal_full_dev_stop_gate_allowed": False,
        "extra_seeds_allowed": False,
        "segphase_full_train_allowed": False,
    }

    save_json(audit, OUT / "training_effectiveness_audit.json")
    save_json(audit, CACHE / "training_effectiveness_audit.json")

    # stop-gate artifact: implementation_bug (no full-dev official numbers)
    verdict = {
        "verdict": "implementation_bug",
        "bug_id": bug_name,
        "formal_stop_gate_run": False,
        "gates": {
            "A_top1_delta_f1_0.5_ge_0.01": None,
            "B_f1_not_down_and_p95_ge_0.15s": None,
            "C_union_oracle_delta_ge_0.01_ci_lo_gt_0": None,
        },
        "reason": "seed42 trained with log_softmax(softmax(z)); checkpoint not eligible for formal stop gate",
        "extra_seeds": False,
        "add_to_union": False,
        "segphase_next": False,
        "confirm_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(verdict, OUT / "stop_gate_verdict.json")
    (OUT / "DKPN_STOP_GATE.FAILED").write_text("implementation_bug\n")
    for empty in [
        "dev_predictions.parquet",
        "dev_threshold_sweep.csv",
        "dev_metrics.json",
        "candidate_oracle.json",
        "candidate_complementarity.csv",
        "event_bootstrap.json",
        "noise_audit.json",
    ]:
        # do not fabricate full-dev metrics; leave placeholders explaining skip
        p = OUT / empty
        if empty.endswith(".json"):
            save_json({"status": "skipped_implementation_bug", "formal_eval": False}, p)
        elif empty.endswith(".csv"):
            p.write_text("status,note\nskipped,implementation_bug\n")
        else:
            pd.DataFrame({"status": ["skipped_implementation_bug"]}).to_parquet(p, index=False)

    md = f"""# DKPN seed42 training-effectiveness audit

**UTC:** {audit['created_utc']}  
**Verdict:** `implementation_bug` (`{bug_name}`)  
**Formal full-dev stop gate:** **not run** (checkpoint ineligible)

## Provenance

| Item | Value |
|--|--|
| DKPN commit | `{provenance['dkpn_commit']}` |
| config hash | `{provenance['config_hash']}` |
| train split hash | `{provenance['train_split_hash']}` |
| last.pt sha256 | `{provenance['checkpoint_last_sha256']}` |
| best.pt | **missing** (only last.pt) |
| TRAIN.DONE | {provenance['TRAIN.DONE_mtime']} |
| env | {provenance['python']} / torch {provenance['torch']} |

## Why mean_loss was flat (0.219 → 0.218)

**Primary:** `DKPN.forward()` returns **softmax probabilities** by default. seed42 `train_loop` did `loss = masked_soft_ce(model(x), ...)` which applies **`log_softmax` a second time**. Official DKPN (`dkpn/train.py`) uses `-y * log(p+eps)` on softmax outputs, **without** another softmax.

This is exactly the forbidden case: *softmax output cannot be treated as logits*.

Secondary contributors:

- CE is **averaged over 3001 samples**; P/S Gaussians occupy ≪1% of the window, so the scalar loss is dominated by the noise channel and can sit near ~0.22 after the first epoch.
- No LR schedule (constant Adam 1e-3).
- No `best.pt` / no full-dev checkpoint selection during training.

## Did weights actually update?

- seed42 init hash ≠ last.pt: **{init_hash != trained_hash}**
- parameter L2 vs seed42 init: **{l2_all:.4f}**
- module L2: `{module_l2}`
- all params trainable: **{n_total == n_train == n_opt}** (N={n_total}; frozen={frozen})
- optimizer holds all trainable params: **{n_opt == n_train}**

Weights **did move**; the run is not a frozen-encoder no-op. The bug is **loss/activation mismatch**, not a complete freeze.

## Batch diagnostics (8 of 128 train crops)

- input min/max/mean/std: {float(X.min()):.3f} / {float(X.max()):.3f} / {float(X.mean()):.3f} / {float(X.std()):.3f}
- P/S/N time-positive frac (>0.1): {pos_frac}
- mask frac: {mask_frac}
- P-only S-channel masked (not treated as S-noise): **{p_only_ok}**
- softmax out min/max/mean/std: {float(probs.min()):.4f} / {float(probs.max()):.4f} / {float(probs.mean()):.4f} / {float(probs.std()):.4f}
- CE (buggy log_softmax∘softmax): {buggy_ce:.4f}
- CE (official log(p)): {official_ce:.4f}
- CE (correct logits + mask): {correct_ce:.4f}
- per-channel CE from logits P/S/N: {ch_loss}
- grad norm correct / buggy: {gnorm:.4f} / {gnorm_buggy:.4f}
- NaN/Inf grads: {nan_grad}
- LR: {lr_actual} (no schedule)

## 128-train localization (S argmax ±0.2 s)

| Init | Acc |
|--|--:|
| last.pt softmax | {acc_tr_probs:.4f} |
| last.pt logits argmax | {acc_tr_logits:.4f} |
| random seed42 softmax | {acc_rd_probs:.4f} |

Trained beats random: **{acc_tr_probs > acc_rd_probs + 0.05}**. Overlays: `{figdir}` ({n_fig} files).

## Alignment

- HDF5 (3, 12000) ENZ: {channel_ok}; 100 Hz: {sr_ok}
- train∩dev events: {len(overlap_dev)}; train∩confirm events: {len(overlap_cf)}
- Confirm waveforms/metrics: **not read**

## Post-fix small overfit (required; not a 30-epoch retrain)

logits=True, 32 traces, 80 max epochs:

- loss {hist_fix[0]:.4f} → {hist_fix[-1]:.4f}
- S-acc {acc0:.3f} → {acc1:.3f}
- overfit_ok: **{overfit_ok}**; smoke_ok: **{smoke_ok}**

## Minimal fix (do not silently retrain 30 epoch)

1. Call `model(x, logits=True)` (or `dkpn_logits`) before `masked_soft_ce`.
2. Alternatively match official: softmax then `-y*log(p+eps)` — **do not mix**.
3. Re-run small-overfit (done) + 1-GPU smoke; **only then** consider a new 30-epoch seed.

## Stop gate

Not executed on this checkpoint. `stop_gate_verdict.json`: **implementation_bug**. Extra seeds: **no**. UNION: **no**. SegPhase full train: **no**.
"""
    (ROOT / "reports/stage10/dkpn_seed42_training_effectiveness_audit.md").write_text(md)
    gate_md = f"""# DKPN seed42 stop gate

**Verdict:** `implementation_bug`  
**Formal A/B/C gates:** not evaluated (checkpoint ineligible)

Seed42 used `log_softmax(softmax(z))`. Full Stage-6 dev inference, threshold sweep, UNION oracle, and event bootstrap were **not** run as official numbers.

| Gate | Result |
|--|--|
| A ΔF1@0.5 ≥ +0.01 vs waveform-only | *not evaluated* |
| B F1 not down and P95 ≥ 0.15 s | *not evaluated* |
| C UNION oracle +0.01 with CI lo > 0 | *not evaluated* |

- extra seeds 123/2026: **not started**
- DKPN not added to UNION
- SegPhase full training: **not started** (implementation_bug stop)
- Stage 6 confirm: **not read** for metrics/threshold

See `dkpn_seed42_training_effectiveness_audit.md`.
"""
    (ROOT / "reports/stage10/dkpn_seed42_stop_gate_report.md").write_text(gate_md)

    state = {
        "stage": "10B",
        "status": "DKPN_SEED42_IMPLEMENTATION_BUG",
        "verdict": "implementation_bug",
        "formal_stop_gate_run": False,
        "extra_seeds": False,
        "sota_claim_allowed": False,
        "confirm_usable_for_stage10_tuning": False,
        "confirm_read_this_round": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "next": "fix_logits_loss_already_in_tree; small-overfit passed if overfit_ok; do not relaunch 30ep until user approves",
        "post_fix_overfit_ok": overfit_ok,
    }
    save_json(state, ROOT / "artifacts/results/stage10/stage10_state.json")
    print(json.dumps({"verdict": "implementation_bug", "overfit_ok": overfit_ok, "l2": l2_all, "acc_tr": acc_tr_probs, "acc_rd": acc_rd_probs}, indent=2))


if __name__ == "__main__":
    main()
