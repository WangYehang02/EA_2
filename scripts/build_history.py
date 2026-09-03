#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.data.splits import load_event_splits
from earthquake.history.temporal_store import TemporalHistoryStore
from earthquake.utils import ensure_dir


def resolve_grid_size(cli_value: float | None) -> float:
    if cli_value is not None:
        return float(cli_value)
    rec = artifacts_dir() / "coverage" / "recommendation.json"
    if rec.exists():
        return float(load_json(rec)["grid_size"])
    return 0.2


def build_frozen(events: pd.DataFrame, splits: dict[str, list[str]], grid_size: float, min_history: int) -> pd.DataFrame:
    store = TemporalHistoryStore(grid_size=grid_size, min_history=min_history)
    train_ids = set(map(str, splits["train"]))
    train_df = events[events["event_id"].astype(str).isin(train_ids)]
    train_order = train_df.groupby("event_id")["origin_time"].min().sort_values().index.astype(str).tolist()

    # Populate store with all train events first (frozen DB)
    for eid in train_order:
        store.update_event(train_df[train_df["event_id"].astype(str) == eid])

    # For train features themselves, re-scan forward to avoid self-history
    store_train = TemporalHistoryStore(grid_size=grid_size, min_history=min_history)
    rows = []
    for eid in train_order:
        cur = train_df[train_df["event_id"].astype(str) == eid]
        for _, row in cur.iterrows():
            stats = store_train.query_row(row)
            rows.append(_stats_row(row, stats, split="train", protocol="frozen"))
        store_train.update_event(cur)

    # Val/test query frozen train store only; never update with val/test labels
    for split in ("val", "test"):
        ids = set(map(str, splits[split]))
        sub = events[events["event_id"].astype(str).isin(ids)]
        order = sub.groupby("event_id")["origin_time"].min().sort_values().index.astype(str).tolist()
        for eid in order:
            cur = sub[sub["event_id"].astype(str) == eid]
            for _, row in cur.iterrows():
                stats = store.query_row(row)
                rows.append(_stats_row(row, stats, split=split, protocol="frozen"))
    return pd.DataFrame(rows)


def build_online(events: pd.DataFrame, splits: dict[str, list[str]], grid_size: float, min_history: int, split: str) -> pd.DataFrame:
    """Online protocol for a single split stream after seeding with earlier splits."""
    store = TemporalHistoryStore(grid_size=grid_size, min_history=min_history)
    seed_splits = ["train"] if split == "val" else ["train", "val"]
    for ss in seed_splits:
        ids = set(map(str, splits[ss]))
        sub = events[events["event_id"].astype(str).isin(ids)]
        order = sub.groupby("event_id")["origin_time"].min().sort_values().index.astype(str).tolist()
        for eid in order:
            store.update_event(sub[sub["event_id"].astype(str) == eid])

    ids = set(map(str, splits[split]))
    sub = events[events["event_id"].astype(str).isin(ids)]
    order = sub.groupby("event_id")["origin_time"].min().sort_values().index.astype(str).tolist()
    rows = []
    for eid in order:
        cur = sub[sub["event_id"].astype(str) == eid]
        for _, row in cur.iterrows():
            stats = store.query_row(row)
            rows.append(_stats_row(row, stats, split=split, protocol="online"))
        store.update_event(cur)
    return pd.DataFrame(rows)


def _stats_row(row: pd.Series, stats, split: str, protocol: str) -> dict:
    return {
        "trace_name": str(row["trace_name"]),
        "event_id": str(row["event_id"]),
        "origin_time": row["origin_time"],
        "split": split,
        "protocol": protocol,
        "history_available": bool(stats.history_available),
        "history_count": int(stats.history_count),
        "tau_p_median": stats.tau_p_median,
        "tau_p_mad": stats.tau_p_mad,
        "tau_s_median": stats.tau_s_median,
        "tau_s_mad": stats.tau_s_mad,
        "delta_sp_median": stats.delta_sp_median,
        "delta_sp_mad": stats.delta_sp_mad,
        "last_history_time": stats.last_history_time,
        "time_since_last_history": (
            (pd.Timestamp(row["origin_time"]) - pd.Timestamp(stats.last_history_time)).total_seconds()
            if stats.last_history_time is not None
            else float("nan")
        ),
        "nearest_historical_source_distance_km": stats.nearest_historical_source_distance_km,
        "matched_key": stats.matched_key,
        "fallback_level": stats.fallback_level,
        "source_latitude": row["source_latitude"],
        "source_longitude": row["source_longitude"],
        "source_depth_km": row["source_depth_km"],
        "station_id": row.get("station_id"),
        "sampling_rate_hz": row["sampling_rate_hz"],
        "trace_start_time": row["trace_start_time"],
        "p_arrival_sample": row.get("p_arrival_sample"),
        "s_arrival_sample": row.get("s_arrival_sample"),
        "snr_db": row.get("snr_db"),
        "distance_km": row.get("distance_km"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--protocol", choices=["frozen", "online", "both"], default="frozen")
    parser.add_argument("--grid-size", type=float, default=None)
    parser.add_argument("--min-history", type=int, default=5)
    args = parser.parse_args()

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    event_times = events.groupby("event_id")["origin_time"].min().sort_values()
    keep = set(event_times.index.astype(str)[: args.max_events].tolist())
    events = events[events["event_id"].astype(str).isin(keep)].copy()
    splits = load_event_splits(artifacts_dir() / "splits")
    splits = {k: [e for e in v if e in keep] for k, v in splits.items()}
    grid_size = resolve_grid_size(args.grid_size)

    out = ensure_dir(artifacts_dir() / "history")
    meta = {"grid_size": grid_size, "min_history": args.min_history, "max_events": args.max_events, "protocol": args.protocol}
    save_json(meta, out / "build_meta.json")

    if args.protocol in ("frozen", "both"):
        df = build_frozen(events, splits, grid_size, args.min_history)
        df.to_parquet(out / "history_features_frozen.parquet", index=False)
        print({"frozen_rows": len(df), "frac_available": float(df["history_available"].mean())})

    if args.protocol in ("online", "both"):
        parts = [build_online(events, splits, grid_size, args.min_history, sp) for sp in ("val", "test")]
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(out / "history_features_online.parquet", index=False)
        print({"online_rows": len(df), "frac_available": float(df["history_available"].mean())})


if __name__ == "__main__":
    main()
