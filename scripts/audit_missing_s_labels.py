#!/usr/bin/env python
"""Audit why S labels are missing on INSTANCE event traces (do not treat missing-S as noise)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json


def main() -> None:
    path = artifacts_dir() / "index_full" / "events.parquet"
    if not path.exists():
        raise SystemExit(f"Need {path}")
    df = pd.read_parquet(path)
    n = len(df)
    p = pd.to_numeric(df.get("p_arrival_sample"), errors="coerce")
    s = pd.to_numeric(df.get("s_arrival_sample"), errors="coerce")
    npts = pd.to_numeric(df.get("n_samples"), errors="coerce")
    sr = pd.to_numeric(df.get("sampling_rate_hz"), errors="coerce")
    if sr.isna().all() and "trace_dt_s" in df.columns:
        dt = pd.to_numeric(df["trace_dt_s"], errors="coerce")
        sr = 1.0 / dt
    dist = pd.to_numeric(df.get("distance_km"), errors="coerce") if "distance_km" in df.columns else None
    tt_s = pd.to_numeric(df.get("path_travel_time_s_s"), errors="coerce") if "path_travel_time_s_s" in df.columns else None

    has_p = p.notna()
    has_s = s.notna()
    missing_s = ~has_s
    p_only = has_p & missing_s
    neither = (~has_p) & missing_s
    both = has_p & has_s

    # S beyond window: sample index >= npts or time >= duration
    beyond = pd.Series(False, index=df.index)
    if npts.notna().any():
        beyond = missing_s & False  # can't observe S sample if missing
        # among present S, how often near end
        s_near_end = has_s & npts.notna() & (s >= (npts - 1))
    else:
        s_near_end = pd.Series(False, index=df.index)

    # If travel time S exists and would land outside 120s window relative to trace start — approximate
    # Using path_travel_time_S and origin vs trace_start if available
    outside_est = pd.Series(False, index=df.index)
    if tt_s is not None and "trace_start_time" in df.columns and "origin_time" in df.columns:
        origin = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
        start = pd.to_datetime(df["trace_start_time"], utc=True, errors="coerce")
        dur = (npts / sr).where(npts.notna() & sr.notna(), 120.0)
        # expected S absolute time ≈ origin + tt_s
        exp_s = origin + pd.to_timedelta(tt_s, unit="s")
        rel = (exp_s - start).dt.total_seconds()
        outside_est = missing_s & tt_s.notna() & ((rel < 0) | (rel > dur))

    by_channel = None
    if "channel_prefix" in df.columns:
        tmp = pd.DataFrame({"ch": df["channel_prefix"].astype(str), "miss": missing_s})
        by_channel = tmp.groupby("ch")["miss"].mean().sort_values(ascending=False).head(20).to_dict()

    dist_bins = None
    if dist is not None:
        cats = pd.cut(dist, bins=[-0.1, 50, 100, 200, 400, 800, 1e6], include_lowest=True)
        dist_bins = (
            pd.DataFrame({"bin": cats.astype(str), "miss": missing_s})
            .groupby("bin")["miss"]
            .agg(["mean", "count"])
            .reset_index()
            .to_dict(orient="records")
        )

    out = {
        "n_traces": int(n),
        "n_with_P": int(has_p.sum()),
        "n_with_S": int(has_s.sum()),
        "n_with_P_and_S": int(both.sum()),
        "n_P_only": int(p_only.sum()),
        "n_neither_P_nor_S": int(neither.sum()),
        "S_missing_frac": float(missing_s.mean()),
        "interpretation": {
            "missing_S_is_not_noise": True,
            "primary_explanation": "No human S pick in metadata (P-only or unlabeled S), not automatic absence of S wave",
            "S_sample_at_last_index_among_present": int(s_near_end.sum()),
            "missing_S_with_travel_time_estimated_outside_window": int(outside_est.sum()),
            "outside_window_frac_of_missing": float(outside_est.sum() / max(int(missing_s.sum()), 1)),
        },
        "missing_rate_by_channel_top20": by_channel,
        "missing_rate_by_distance_bin": dist_bins,
        "phaseA_v1_policy": {
            "supervise_on": "traces_with_finite_P_and_S_only",
            "include_real_noise_hdf5": True,
            "include_P_only": False,
            "reason_exclude_P_only": "partial-label/masked loss not yet implemented+tested; must not treat unlabeled S regions as noise",
            "n_ps_both_for_training_pool_all_events": int(both.sum()),
        },
    }
    save_json(out, artifacts_dir() / "results" / "stage6" / "missing_s_label_audit.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
