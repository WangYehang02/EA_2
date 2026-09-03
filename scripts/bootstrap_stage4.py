#!/usr/bin/env python
"""Event-level paired bootstrap for Stage-4 confirmatory holdout (n=5000)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.metrics import match_picks, audit_pick_errors
from earthquake.utils import ensure_dir


def _metrics(pred_s, true_s, sr):
    m = match_picks(pred_s, true_s, sr, windows_s=(0.1, 0.5))
    a = audit_pick_errors(pred_s, true_s, sr, window_s=0.5)
    n = max(int(a["n_labeled"]), 1)
    return {
        "f1_0.1": m["f1@0.1s"],
        "f1_0.5": m["f1@0.5s"],
        "e2e_p95": m["p95_ae"],
        "wrong_peak_rate": float(a["n_wrong_peak_beyond_tol"] / n),
    }


def bootstrap_pair(df, col_a, col_b, n_boot, seed):
    events = df["event_id"].astype(str).to_numpy()
    uniq, inv = np.unique(events, return_inverse=True)
    event_rows = [np.where(inv == i)[0] for i in range(len(uniq))]
    a = df[col_a].to_numpy(dtype=np.float64)
    b = df[col_b].to_numpy(dtype=np.float64)
    true = df["true_s_sample"].to_numpy(dtype=np.float64)
    sr = df["sampling_rate_hz"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    keys = ["delta_f1_0.1", "delta_f1_0.5", "delta_e2e_p95", "delta_wrong_peak_rate"]
    samples = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([event_rows[j] for j in chosen])
        ma = _metrics(a[idx], true[idx], sr[idx])
        mb = _metrics(b[idx], true[idx], sr[idx])
        samples["delta_f1_0.1"][i] = mb["f1_0.1"] - ma["f1_0.1"]
        samples["delta_f1_0.5"][i] = mb["f1_0.5"] - ma["f1_0.5"]
        samples["delta_e2e_p95"][i] = mb["e2e_p95"] - ma["e2e_p95"]
        samples["delta_wrong_peak_rate"][i] = mb["wrong_peak_rate"] - ma["wrong_peak_rate"]
    out = {"n_events": int(len(uniq)), "n_traces": int(len(df)), "n_bootstrap": n_boot}
    for k, arr in samples.items():
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        if "f1" in k:
            p_imp = float(np.mean(arr > 0))
            sig = bool(lo > 0)
        else:
            p_imp = float(np.mean(arr < 0))
            sig = bool(hi < 0)
        out[k] = {"mean": float(np.nanmean(arr)), "ci95_low": float(lo), "ci95_high": float(hi), "probability_improved": p_imp, "significant": sig}
    out["_samples"] = samples
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/confirmatory.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    picks = out / "holdout_picks.parquet"
    if not picks.exists():
        raise SystemExit("Missing holdout_picks.parquet — run evaluate_stage4_holdout.py first")
    df = pd.read_parquet(picks)
    n_boot = int(cfg.get("n_bootstrap", 5000))
    seed = int(cfg.get("seed", 42))

    pairs = [
        ("phasenet", "pred_s_phasenet", "fixed_catalog_rescore", "pred_s_catalog_rescore"),
        ("learned_gate", "pred_s_learned_gate", "fixed_catalog_rescore", "pred_s_catalog_rescore"),
        ("distance_only", "pred_s_distance_only", "fixed_catalog_rescore", "pred_s_catalog_rescore"),
        ("shuffled_history", "pred_s_shuffled_history", "fixed_catalog_rescore", "pred_s_catalog_rescore"),
    ]
    report = {"comparisons": {}, "n_bootstrap": n_boot, "seed": seed, "sampling_unit": "event"}
    sample_frames = []
    for a_name, a_col, b_name, b_col in pairs:
        if a_col not in df.columns or b_col not in df.columns:
            report["comparisons"][f"{b_name}_vs_{a_name}"] = {"skipped": True, "reason": "missing column"}
            continue
        # note: bootstrap_pair computes b - a, so pass baseline as a and improved as b
        # For fixed vs gate we want fixed - gate when pair is (gate, fixed)? 
        # Spec: fixed vs PhaseNet => fixed - PhaseNet. So a=phasenet, b=fixed.
        key = f"{b_name}_vs_{a_name}"
        res = bootstrap_pair(df, a_col, b_col, n_boot, seed)
        samples = res.pop("_samples")
        report["comparisons"][key] = res
        print(key, res.get("delta_f1_0.5"), flush=True)
        for k, arr in samples.items():
            sample_frames.append(pd.DataFrame({"comparison": key, "metric": k, "draw": np.arange(n_boot), "value": arr}))

    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_parquet(out / "bootstrap_samples.parquet", index=False)
    save_json(report, out / "bootstrap_confirmatory.json")
    # table
    rows = []
    for k, v in report["comparisons"].items():
        if v.get("skipped"):
            continue
        for mk in ["delta_f1_0.1", "delta_f1_0.5", "delta_e2e_p95", "delta_wrong_peak_rate"]:
            d = v[mk]
            rows.append({"comparison": k, "metric": mk, **{x: d[x] for x in d}})
    pd.DataFrame(rows).to_csv(out / "tables" / "bootstrap_results.csv", index=False)
    print({"saved": str(out / "bootstrap_confirmatory.json"), "n_events": int(df.event_id.nunique())}, flush=True)


if __name__ == "__main__":
    main()
