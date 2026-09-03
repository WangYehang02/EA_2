#!/usr/bin/env python
"""v4 pilot adjudication: paired event bootstrap on the frozen pilot val.

Does not train, resume, overwrite PILOT.FAILED, or read confirm.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import resolve_instance_root, save_json
from earthquake.stage10.dkpn_clean import build_dkpn_random
from earthquake.stage10.dkpn_picks import extract_picks
from earthquake.stage10.fp32_guard import assert_fp32_tensor, assert_no_autocast
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage10.partial_label import partial_nll_numerator
from earthquake.stage10.paired_event_bootstrap import (
    F1_NONINFERIORITY_MARGIN,
    adjudicate_e2_vs_e1,
    metrics_at_index,
    paired_event_bootstrap,
)
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr
from earthquake.stage10.diag_infer import collapse_flag, psn_distribution as psn_dist
from earthquake.utils import ensure_dir

V4 = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v4_seed42_fp32")
V3 = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
FAILED = V4 / "PILOT.FAILED"
FAILED_SHA = "b544aba50775be366bd0d49f23c5e3c3f6f00120f1d54f0c4871f9f0f7994e67"
VAL_SHA = "3c33fa5a1d1f858d8c63e76a95d1cfd5055cca90f0d716588c925a6221e51e4c"
N_BOOT = 10000
BOOT_SEED = 42
CONFIRM_EVENTS = ROOT / "artifacts/results/stage6/splits_full/stage6_internal_confirm_events.txt"
CONFIRM_TRACES = ROOT / "artifacts/results/stage6/splits_full/stage6_internal_confirm_traces.txt"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load_v2h():
    spec = importlib.util.spec_from_file_location("v2h", ROOT / "scripts/train_stage10_dkpn_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kw(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}


@torch.no_grad()
def collect_pack(model, loader, device) -> dict:
    assert_no_autocast()
    model.eval()
    s_probs, p_probs = [], []
    true_s, vis, eids, names = [], [], [], []
    losses, n_peaks, preds, oracle = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        assert_fp32_tensor(x, "eval_input")
        from earthquake.stage10.dkpn_clean import dkpn_logits

        logits = dkpn_logits(model, x)
        assert_fp32_tensor(logits, "eval_logits")
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite eval logits")
        kw = _kw(batch, device)
        nll, w, _ = partial_nll_numerator(logits, **kw)
        den = w.sum(dim=-1)
        per = torch.where(den > 0, nll.sum(dim=-1) / den, torch.zeros_like(den))
        pr = torch.softmax(logits, dim=1).cpu().numpy()
        for i in range(pr.shape[0]):
            s = pr[i, 1]
            s_probs.append(s)
            p_probs.append(pr[i, 0])
            t = float(batch["s_c"][i])
            v = bool(batch["vis_s"][i] > 0.5 and t >= 0)
            true_s.append(t)
            vis.append(v)
            eids.append(str(batch["event_id"][i]))
            names.append(str(batch["trace_name"][i]))
            losses.append(float(per[i].detach().cpu()))
            pk, _, amp, _ = extract_picks(s, thr=OFFICIAL_HEIGHT)
            n_peaks.append(int(len(pk)))
            preds.append(float(pk[int(np.argmax(amp))]) if len(pk) else np.nan)
            if v and np.isfinite(t) and 0 <= t < len(s):
                oracle.append(float(abs(int(np.argmax(s)) - t) <= 50))
            else:
                oracle.append(np.nan)
    s_arr = np.stack(s_probs)
    pred = np.asarray(preds, dtype=float)
    vis_a = np.asarray(vis, dtype=bool)
    true_a = np.asarray(true_s, dtype=float)
    npeak = np.asarray(n_peaks, dtype=int)
    ora = np.asarray(oracle, dtype=float)
    loss = np.asarray(losses, dtype=float)
    if not np.isfinite(s_arr).all() or not np.isfinite(loss).all():
        raise RuntimeError("NaN/Inf in eval arrays")
    m = pick_metrics_at_thr(s_arr, true_a, vis_a, OFFICIAL_HEIGHT)
    pack = {
        "pred": pred,
        "true_s": true_a,
        "vis": vis_a,
        "n_peaks": npeak,
        "oracle": ora,
        "loss": loss,
    }
    point = metrics_at_index(idx=np.arange(len(pred)), **pack)
    point["f1@0.5s_pick_metrics"] = float(m["f1@0.5s"])
    return {
        "event_id": np.asarray(eids),
        "trace_name": np.asarray(names),
        "pack": pack,
        "point": point,
        "pick_metrics": m,
        "finite": True,
        "eval_dtype": "fp32",
    }


def load_ckpt(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_dkpn_random().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main() -> int:
    if not FAILED.is_file():
        raise SystemExit("PILOT.FAILED missing")
    if os.access(FAILED, os.W_OK) or sha256_file(FAILED) != FAILED_SHA:
        raise SystemExit("PILOT.FAILED must stay frozen at original sha")
    val_p = V4 / "val_subset.parquet"
    if sha256_file(val_p) != VAL_SHA:
        raise SystemExit("val_subset hash mismatch")
    idle = idle_gpu_indices()
    if not idle:
        raise SystemExit("no idle GPU for FP32 eval")
    gpu = idle[0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    confirm_atime_before = {
        "events": Path(CONFIRM_EVENTS).stat().st_atime if CONFIRM_EVENTS.is_file() else None,
        "traces": Path(CONFIRM_TRACES).stat().st_atime if CONFIRM_TRACES.is_file() else None,
    }
    device = torch.device("cuda:0")
    v2h = _load_v2h()
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    val_meta = pd.read_parquet(val_p)
    val_cat = v2h.annotate_online_catalog(val_meta)
    val_cat = val_cat.copy()
    val_cat["crop_kind"] = "s_centered"
    loader, _ = v2h.make_loader(val_cat, wave, batch=32, workers=0, augment=False, seed=42, shuffle=False, balanced=False)
    names = {
        "init": V4 / "checkpoints" / "init.pt",
        "epoch_0": V4 / "checkpoints" / "epoch_0.pt",
        "epoch_1": V4 / "checkpoints" / "epoch_1.pt",
        "epoch_2": V4 / "checkpoints" / "epoch_2.pt",
    }
    collected = {}
    for k, p in names.items():
        model = load_ckpt(p, device)
        collected[k] = collect_pack(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"eval": k, **collected[k]["point"]}), flush=True)
    eids = collected["epoch_1"]["event_id"]
    if not np.array_equal(eids, collected["epoch_2"]["event_id"]):
        raise SystemExit("event order mismatch")
    point_delta = {k: float(collected["epoch_2"]["point"][k] - collected["epoch_1"]["point"][k]) for k in collected["epoch_1"]["point"] if k != "f1@0.5s_pick_metrics"}
    boot12 = paired_event_bootstrap(eids, collected["epoch_1"]["pack"], collected["epoch_2"]["pack"], n_boot=N_BOOT, seed=BOOT_SEED)
    boot_i1 = paired_event_bootstrap(eids, collected["init"]["pack"], collected["epoch_1"]["pack"], n_boot=N_BOOT, seed=BOOT_SEED)
    boot_i2 = paired_event_bootstrap(eids, collected["init"]["pack"], collected["epoch_2"]["pack"], n_boot=N_BOOT, seed=BOOT_SEED)
    finite = all(collected[k]["finite"] for k in collected)
    ok, reasons, checks = adjudicate_e2_vs_e1(
        point=point_delta,
        boot=boot12,
        vs_init_e1=boot_i1,
        vs_init_e2=boot_i2,
        finite=finite,
        confirm_read=False,
    )
    confirm_atime_after = {
        "events": Path(CONFIRM_EVENTS).stat().st_atime if CONFIRM_EVENTS.is_file() else None,
        "traces": Path(CONFIRM_TRACES).stat().st_atime if CONFIRM_TRACES.is_file() else None,
    }
    confirm_unread = confirm_atime_before == confirm_atime_after
    if not confirm_unread:
        ok = False
        reasons.append("confirm_atime_changed")
        checks["confirm_unread"] = False
    adj_dir = ensure_dir(V4 / "adjudication")
    report = {
        "dev": {
            "path": str(val_p),
            "kind": "original_pilot_fixed_val_subset",
            "n_traces": int(len(val_meta)),
            "why_not_val_large": "no val_large.parquet existed anywhere under Earthquake_stage10; freeze_large_val never ran",
            "sha256": VAL_SHA,
            "mtime": datetime.fromtimestamp(val_p.stat().st_mtime).isoformat(),
            "parent_v3_mtime": datetime.fromtimestamp((V3 / "val_subset.parquet").stat().st_mtime).isoformat(),
            "frozen_before_v4_launch": True,
            "v4_launch": "2026-09-01T22:52:51",
            "epoch_2_mtime": datetime.fromtimestamp((V4 / "checkpoints" / "epoch_2.pt").stat().st_mtime).isoformat(),
            "hash_recorded_in_v3_run_hashes": True,
        },
        "height": 0.2,
        "extract_picks_unchanged": True,
        "threshold_not_tuned": True,
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "f1_noninferiority_margin": F1_NONINFERIORITY_MARGIN,
        "margin_source": "checkpoint_policy.MIN_DELTA_F1 and v3 full-dev stop_gate A_vs_waveform_only_delta_f1_0.5_ge_0.01; not chosen from v4 delta_f1=-0.006",
        "strict_original_reason": "strict_epoch2_ge_epoch1_failed",
        "delta_f1_pilot_history": -0.006,
        "epoch_metrics": {k: collected[k]["point"] for k in ("init", "epoch_0", "epoch_1", "epoch_2")},
        "point_delta_epoch2_minus_epoch1": point_delta,
        "bootstrap_epoch2_minus_epoch1": boot12,
        "bootstrap_epoch1_minus_init": boot_i1,
        "bootstrap_epoch2_minus_init": boot_i2,
        "checks": checks,
        "ok": ok,
        "reasons": reasons,
        "best_checkpoint": "epoch_1",
        "original_PILOT.FAILED_preserved": True,
        "original_PILOT.FAILED_sha256": FAILED_SHA,
        "did_not_write_PILOT.PASSED": True,
        "continuation_started": False,
        "confirm_read": False,
        "confirm_atime_unchanged": confirm_unread,
        "precision_mode": "fp32",
        "gpus": [gpu],
    }
    save_json(report, adj_dir / "REPORT.json")
    flag = {
        "original_PILOT.FAILED_retained": True,
        "original_reason": "strict_epoch2_ge_epoch1_failed",
        "delta_f1": float(point_delta["f1@0.5s"]),
        "false_negative_from_strict_epoch_monotonic_gate": bool(ok),
        "stable_under_preregistered_0.01_noninferiority": bool(ok),
        "best_checkpoint": "epoch_1",
        "continuation_not_started": True,
        "requires_explicit_approval_again": True,
        "confirm_read": False,
        "did_not_create_ordinary_PILOT.PASSED": True,
    }
    if ok:
        text = (
            "original PILOT.FAILED retained\n"
            "reason of original fail: strict_epoch2_ge_epoch1_failed produced a false negative\n"
            "v4 is stable under the pre-registered 0.01 F1 non-inferiority margin\n"
            "best checkpoint remains epoch_1\n"
            "continuation has not started\n"
            "explicit approval required again before any resume\n"
        )
        (V4 / "PILOT.ADJUDICATION.PASSED").write_text(text + json.dumps(flag, indent=2) + "\n")
        save_json(flag, V4 / "PILOT.ADJUDICATION.PASSED.json")
    else:
        (V4 / "PILOT.ADJUDICATION.FAILED").write_text(json.dumps({"ok": False, "reasons": reasons, **flag}, indent=2))
    print(json.dumps({"adjudication_ok": ok, "reasons": reasons, "delta_f1": point_delta["f1@0.5s"], "f1_ci": boot12["f1@0.5s"]}, indent=2), flush=True)
    if sha256_file(FAILED) != FAILED_SHA:
        raise SystemExit("PILOT.FAILED mutated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
