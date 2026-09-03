#!/usr/bin/env python
"""Hard-subset analysis on Stage-4 holdout using Stage-2 predefined bins."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks
from earthquake.utils import ensure_dir

# Stage-2 predefined bins — DO NOT retune on holdout
BINS = {
    "s_n_cands": [(-0.1, 1.5, "single_peak"), (1.5, 100, "multi_peak")],
    "s_peak_probability": [(-0.01, 0.3, "<0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 1.01, ">0.7")],
    "residual_s_mad": [(-0.01, 0.2, "<0.2"), (0.2, 0.5, "0.2-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 100, ">1.0")],
    "history_count": [(-0.1, 5, "<5"), (5, 10, "5-9"), (10, 20, "10-19"), (20, 1e9, ">=20")],
    "distance_km": [(-0.1, 50, "<50"), (50, 100, "50-100"), (100, 200, "100-200"), (200, 1e9, ">=200")],
}


def _f1(pred, true, sr):
    m = match_picks(pred, true, sr)
    return m["f1@0.1s"], m["f1@0.5s"], m["p95_ae"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picks", default="artifacts/results/stage4/holdout_picks.parquet")
    args = parser.parse_args()
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    df = pd.read_parquet(ROOT / args.picks)
    rows = []
    for feat, bins in BINS.items():
        if feat not in df.columns:
            continue
        for lo, hi, lab in bins:
            sub = df[(df[feat] > lo) & (df[feat] <= hi)]
            if len(sub) == 0:
                continue
            f1_01_pn, f1_05_pn, p95_pn = _f1(sub.pred_s_phasenet, sub.true_s_sample, sub.sampling_rate_hz)
            f1_01_fx, f1_05_fx, p95_fx = _f1(sub.pred_s_catalog_rescore, sub.true_s_sample, sub.sampling_rate_hz)
            rows.append(
                {
                    "feature": feat,
                    "bin": lab,
                    "n_traces": int(len(sub)),
                    "n_events": int(sub.event_id.nunique()),
                    "small_n": bool(len(sub) < 30 or sub.event_id.nunique() < 5),
                    "phasenet_f1_0.1": f1_01_pn,
                    "phasenet_f1_0.5": f1_05_pn,
                    "phasenet_e2e_p95": p95_pn,
                    "fixed_f1_0.1": f1_01_fx,
                    "fixed_f1_0.5": f1_05_fx,
                    "fixed_e2e_p95": p95_fx,
                    "delta_f1_0.5": f1_05_fx - f1_05_pn,
                    "delta_e2e_p95": p95_fx - p95_pn,
                    "history_coverage": float(sub.history_available.mean()) if "history_available" in sub.columns else float("nan"),
                }
            )
    # history available / unavailable
    for lab, mask in [("history_available", df.history_available == True), ("history_unavailable", df.history_available == False)]:  # noqa: E712
        sub = df[mask]
        if len(sub) == 0:
            continue
        f1_01_pn, f1_05_pn, p95_pn = _f1(sub.pred_s_phasenet, sub.true_s_sample, sub.sampling_rate_hz)
        f1_01_fx, f1_05_fx, p95_fx = _f1(sub.pred_s_catalog_rescore, sub.true_s_sample, sub.sampling_rate_hz)
        rows.append(
            {
                "feature": "history_available_flag",
                "bin": lab,
                "n_traces": int(len(sub)),
                "n_events": int(sub.event_id.nunique()),
                "small_n": bool(len(sub) < 30 or sub.event_id.nunique() < 5),
                "phasenet_f1_0.1": f1_01_pn,
                "phasenet_f1_0.5": f1_05_pn,
                "phasenet_e2e_p95": p95_pn,
                "fixed_f1_0.1": f1_01_fx,
                "fixed_f1_0.5": f1_05_fx,
                "fixed_e2e_p95": p95_fx,
                "delta_f1_0.5": f1_05_fx - f1_05_pn,
                "delta_e2e_p95": p95_fx - p95_pn,
                "history_coverage": float(sub.history_available.mean()),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tables" / "hard_subset_results.csv", index=False)
    save_json({"n_rows": len(tab), "note": "Stage-2 predefined bins; small_n groups are not for strong claims"}, out / "hard_subset_meta.json")
    print(tab.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
