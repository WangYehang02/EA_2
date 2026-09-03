from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from earthquake.history.region import depth_bin, path_key, source_region, station_id
from earthquake.history.temporal_store import TemporalHistoryStore


def analyze_coverage(
    events_df: pd.DataFrame,
    splits: dict[str, list[str]],
    grid_sizes: list[float],
    min_history: int = 5,
) -> dict[str, Any]:
    """Analyze frozen-history coverage on test events for multiple grid sizes."""
    train_ids = set(map(str, splits["train"]))
    test_ids = set(map(str, splits["test"]))

    train_df = events_df[events_df["event_id"].astype(str).isin(train_ids)].copy()
    test_df = events_df[events_df["event_id"].astype(str).isin(test_ids)].copy()

    rows = []
    hist_count_samples: dict[float, list[int]] = {}
    mad_samples: dict[float, list[float]] = {}

    for gs in grid_sizes:
        store = TemporalHistoryStore(grid_size=gs, min_history=min_history)
        # Build frozen history from train only, in time order
        train_events = (
            train_df.groupby("event_id")["origin_time"].min().sort_values().index.astype(str).tolist()
        )
        for eid in train_events:
            store.update_event(train_df[train_df["event_id"].astype(str) == eid])

        counts = []
        available = 0
        thresholds = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
        mads = []
        n_test = len(test_df)
        for _, row in test_df.iterrows():
            stats = store.query_row(row)
            counts.append(stats.history_count)
            if stats.history_available:
                available += 1
            for thr in thresholds:
                if stats.history_count >= thr:
                    thresholds[thr] += 1
            if np.isfinite(stats.tau_p_mad):
                mads.append(stats.tau_p_mad)
            if np.isfinite(stats.tau_s_mad):
                mads.append(stats.tau_s_mad)

        hist_count_samples[gs] = counts
        mad_samples[gs] = mads
        rows.append(
            {
                "grid_size": gs,
                "n_paths": store.n_paths(),
                "n_test_traces": n_test,
                "frac_history_available": available / max(n_test, 1),
                "frac_count_ge_1": thresholds[1] / max(n_test, 1),
                "frac_count_ge_3": thresholds[3] / max(n_test, 1),
                "frac_count_ge_5": thresholds[5] / max(n_test, 1),
                "frac_count_ge_10": thresholds[10] / max(n_test, 1),
                "frac_count_ge_20": thresholds[20] / max(n_test, 1),
                "median_history_count": float(np.median(counts)) if counts else 0.0,
                "median_travel_time_mad": float(np.median(mads)) if mads else float("nan"),
            }
        )

    coverage_df = pd.DataFrame(rows)
    recommendation = recommend_grid(coverage_df, min_history=min_history)
    return {
        "coverage_df": coverage_df,
        "hist_count_samples": hist_count_samples,
        "mad_samples": mad_samples,
        "recommendation": recommendation,
    }


def recommend_grid(coverage_df: pd.DataFrame, min_history: int = 5) -> dict[str, Any]:
    """Prefer grid with high >=min_history coverage and not-too-large MAD."""
    if coverage_df.empty:
        return {"grid_size": 0.2, "reason": "empty coverage; default 0.2"}
    col = f"frac_count_ge_{min_history}" if f"frac_count_ge_{min_history}" in coverage_df.columns else "frac_history_available"
    df = coverage_df.copy()
    # score: coverage primary, penalize very large MAD
    mad = df["median_travel_time_mad"].fillna(df["median_travel_time_mad"].median())
    score = df[col] - 0.05 * np.clip(mad / 0.5, 0, 5)
    idx = int(score.idxmax())
    row = df.loc[idx]
    return {
        "grid_size": float(row["grid_size"]),
        "score": float(score.loc[idx]),
        "frac_history_available": float(row["frac_history_available"]),
        "median_history_count": float(row["median_history_count"]),
        "median_travel_time_mad": float(row["median_travel_time_mad"]),
        "reason": f"maximize {col} with MAD penalty",
    }
