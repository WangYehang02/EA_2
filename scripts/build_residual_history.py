#!/usr/bin/env python
"""Build residual path history features from frozen absolute history + distance baseline."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.history.residual_prior import attach_baseline_and_residuals, path_stats_to_residual_stats
from earthquake.history.shrinkage import search_shrinkage_k
from earthquake.history.temporal_store import PathStats
from earthquake.utils import ensure_dir


def row_to_stats(h: pd.Series) -> PathStats:
    return PathStats(
        history_count=int(h.get("history_count", 0) or 0),
        tau_p_median=float(h.get("tau_p_median", np.nan)),
        tau_p_mad=float(h.get("tau_p_mad", np.nan)),
        tau_s_median=float(h.get("tau_s_median", np.nan)),
        tau_s_mad=float(h.get("tau_s_mad", np.nan)),
        delta_sp_median=float(h.get("delta_sp_median", np.nan)),
        delta_sp_mad=float(h.get("delta_sp_mad", np.nan)),
        last_history_time=h.get("last_history_time"),
        nearest_historical_source_distance_km=float(h.get("nearest_historical_source_distance_km", np.nan)),
        history_available=bool(h.get("history_available", False)),
        matched_key=h.get("matched_key"),
        fallback_level=int(h.get("fallback_level", -1)) if pd.notna(h.get("fallback_level", np.nan)) else -1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_fixed.yaml")
    parser.add_argument("--protocol", default="frozen")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)

    out = ensure_dir(artifacts_dir() / "results" / "stage2")
    with open(out / "travel_time_baseline.pkl", "rb") as f:
        baseline = pickle.load(f)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    hist_path = artifacts_dir() / "history" / "history_features_frozen.parquet"
    hist = pd.read_parquet(hist_path)

    # Train residuals for global fallback + shrinkage search on val
    train = events[events["split"] == "train"].copy()
    val = events[events["split"] == "val"].copy()
    train_aug = attach_baseline_and_residuals(train, baseline)
    global_res = {
        "residual_p_median": float(train_aug["residual_tau_p"].median(skipna=True)),
        "residual_s_median": float(train_aug["residual_tau_s"].median(skipna=True)),
        "residual_sp_median": float(train_aug["residual_delta_sp"].median(skipna=True)),
    }

    # Attach residual stats for all traces with history features
    merged = events.merge(hist, on=["trace_name", "event_id"], how="left", suffixes=("", "_h"))
    rows = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="residual-history"):
        base = baseline.predict_row(row)
        stats = row_to_stats(row)
        rs = path_stats_to_residual_stats(stats, base)
        rows.append(
            {
                "trace_name": row["trace_name"],
                "event_id": row["event_id"],
                "split": row.get("split"),
                "protocol": args.protocol,
                "base_tau_p": base["base_tau_p"],
                "base_tau_s": base["base_tau_s"],
                "base_delta_sp": base["base_delta_sp"],
                "history_count": rs.history_count,
                "history_available": rs.history_available,
                "residual_p_median": rs.residual_p_median,
                "residual_p_mad": rs.residual_p_mad,
                "residual_s_median": rs.residual_s_median,
                "residual_s_mad": rs.residual_s_mad,
                "residual_sp_median": rs.residual_sp_median,
                "residual_sp_mad": rs.residual_sp_mad,
                "tau_p_median": rs.tau_p_median,
                "tau_s_median": rs.tau_s_median,
                "delta_sp_median": rs.delta_sp_median,
                "tau_p_mad": rs.tau_p_mad,
                "tau_s_mad": rs.tau_s_mad,
                "delta_sp_mad": rs.delta_sp_mad,
                "matched_key": rs.matched_key,
                "fallback_level": rs.fallback_level,
            }
        )
    feat = pd.DataFrame(rows)
    feat_path = artifacts_dir() / "history" / "residual_history_features_frozen.parquet"
    feat.to_parquet(feat_path, index=False)

    # shrinkage_k on val using path residual vs observed
    val_aug = attach_baseline_and_residuals(val, baseline)
    val_m = val_aug.merge(feat, on=["trace_name", "event_id"], how="left", suffixes=("", "_f"))
    ks = cfg.get("shrinkage_k_grid", [1, 2, 5, 10, 20, 50])
    search = {}
    for phase, obs_c, path_c, base_c in [
        ("p", "obs_tau_p", "residual_p_median", "base_tau_p"),
        ("s", "obs_tau_s", "residual_s_median", "base_tau_s"),
        ("sp", "obs_delta_sp", "residual_sp_median", "base_delta_sp"),
    ]:
        search[phase] = search_shrinkage_k(
            residuals_path=val_m[path_c].to_numpy(),
            residuals_fallback=np.full(len(val_m), global_res[f"residual_{'sp' if phase=='sp' else phase}_median"]),
            history_counts=val_m["history_count"].fillna(0).to_numpy(),
            observed=val_m[obs_c].to_numpy(),
            base=val_m[base_c].to_numpy(),
            ks=ks,
        )
    # choose single k by average of p/s best (prefer shared k minimizing mean MAE)
    best_k = None
    best_score = float("inf")
    for k in ks:
        maes = []
        for phase in ("p", "s"):
            row = next(r for r in search[phase]["grid"] if r["k"] == float(k))
            maes.append(row["mae"])
        sc = float(np.nanmean(maes))
        if sc < best_score:
            best_score = sc
            best_k = float(k)

    meta = {
        "global_residual": global_res,
        "shrinkage_search": search,
        "best_shrinkage_k": best_k,
        "best_shrinkage_score_mae": best_score,
        "baseline_kind": baseline.kind,
        "n_features": len(feat),
        "path": str(feat_path),
    }
    save_json(meta, out / "residual_history_meta.json")
    print(meta)


if __name__ == "__main__":
    main()
