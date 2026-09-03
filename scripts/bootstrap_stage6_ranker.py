#!/usr/bin/env python
"""Event-level paired bootstrap for Phase C ranker vs baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.phaseB import event_level_bootstrap_delta, per_trace_hit
from earthquake.utils import ensure_dir

REPS = 5000
SEED = 20260816


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_bootstrap")
        raise SystemExit("method_lock")
    except RuntimeError:
        pass


def metric_vectors(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict[str, np.ndarray]:
    hit01 = per_trace_hit(pred, true, sr, 0.1)
    hit05 = per_trace_hit(pred, true, sr, 0.5)
    ae = np.abs((pred - true) / sr)
    wrong = ((np.isfinite(pred) & np.isfinite(true) & (ae > 0.5))).astype(float)
    miss = ((~np.isfinite(pred) & np.isfinite(true))).astype(float)
    # for p95 we bootstrap later specially
    return {"f1@0.1": hit01, "f1@0.5": hit05, "wrong_peak": wrong, "miss": miss, "ae": ae}


def boot_p95(eids, ae_a, ae_b, reps=REPS, seed=SEED):
    uniq = np.unique(eids)
    ev_idx = {}
    for i, e in enumerate(eids):
        ev_idx.setdefault(str(e), []).append(i)
    ev_idx = {e: np.asarray(ix, dtype=np.int64) for e, ix in ev_idx.items()}
    uniq_list = [str(e) for e in uniq]
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for r in range(reps):
        sampled = rng.choice(uniq_list, size=len(uniq_list), replace=True)
        sel = np.concatenate([ev_idx[e] for e in sampled])
        aa = ae_a[sel]
        bb = ae_b[sel]
        aa = aa[np.isfinite(aa)]
        bb = bb[np.isfinite(bb)]
        pa = float(np.percentile(aa, 95)) if aa.size else np.nan
        pb = float(np.percentile(bb, 95)) if bb.size else np.nan
        boots[r] = pb - pa
    aa = ae_a[np.isfinite(ae_a)]
    bb = ae_b[np.isfinite(ae_b)]
    d0 = float(np.percentile(bb, 95) - np.percentile(aa, 95))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_delta": d0, "ci95": [float(lo), float(hi)], "n_events": int(len(uniq)), "n_traces": int(len(eids))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-preds", default="artifacts/results/stage6/phaseC/ranker_dev_predictions.parquet")
    parser.add_argument("--baseline-preds-dir", default="artifacts/results/stage6/phaseC/baseline_preds")
    args = parser.parse_args()
    _guard()
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC")
    ranker = pd.read_parquet(ROOT / args.ranker_preds)
    eids = ranker["event_id"].astype(str).to_numpy()
    true = ranker["true_s_sample"].to_numpy(float)
    sr = ranker["sampling_rate_hz"].to_numpy(float)
    rp = ranker["pred_s_sample"].to_numpy(float)
    rm = metric_vectors(rp, true, sr)

    comparisons = {}
    bdir = Path(ROOT / args.baseline_preds_dir)
    for name in ["STEAD_top1", "fixed_rescore_STEAD", "fixed_rescore_UNION", "prob_heuristic_UNION", "R1"]:
        p = bdir / f"{name}.npy"
        if not p.exists():
            continue
        bp = np.load(p)
        bm = metric_vectors(bp, true, sr)
        comparisons[f"ranker_vs_{name}"] = {
            "f1@0.1": event_level_bootstrap_delta(eids, bm["f1@0.1"], rm["f1@0.1"], reps=REPS, seed=SEED),
            "f1@0.5": event_level_bootstrap_delta(eids, bm["f1@0.5"], rm["f1@0.5"], reps=REPS, seed=SEED),
            "wrong_peak": event_level_bootstrap_delta(eids, bm["wrong_peak"], rm["wrong_peak"], reps=REPS, seed=SEED),
            "miss": event_level_bootstrap_delta(eids, bm["miss"], rm["miss"], reps=REPS, seed=SEED),
            "detected_ae_p95": boot_p95(eids, bm["ae"], rm["ae"]),
        }
    save_json(comparisons, out / "phaseC_bootstrap.json")
    print(json.dumps({k: v.get("f1@0.5") for k, v in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
