#!/usr/bin/env python
"""Build INSTANCE events/noise parquet indexes.

IMPORTANT:
- ``--max-events`` defaults to None (keep ALL events). Silent 10k truncation is forbidden.
- Default output is ``artifacts/index_full/`` so the legacy capped
  ``artifacts/index/events.parquet`` is never overwritten by accident.
- If the written event count is less than raw unique events and no explicit
  ``--max-events`` was provided, this script raises.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, resolve_instance_root, save_json
from earthquake.data.metadata import (
    find_events_csv,
    find_noise_csv,
    iter_csv_chunks,
    normalize_events_frame,
    normalize_noise_frame,
)
from earthquake.data.schema import SchemaMapping
from earthquake.data.splits import assign_split_column, event_time_split, noise_time_split, save_event_splits
from earthquake.history.region import station_id
from earthquake.utils import ensure_dir


def load_schema() -> SchemaMapping:
    raw = load_json(artifacts_dir() / "schema" / "schema_mapping.json")
    return SchemaMapping(**raw)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="If set, keep the chronologically earliest N events. Default: keep ALL (no silent truncation).",
    )
    parser.add_argument(
        "--max-noise",
        type=int,
        default=None,
        help="If set, keep at most N noise traces (CSV stream order). Default: keep ALL.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="artifacts/index_full",
        help="Output index directory (default artifacts/index_full; does not overwrite legacy artifacts/index).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Optional event split dir. Default: <index-dir>/../splits_full relative to artifacts, or artifacts/splits_full.",
    )
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument(
        "--allow-overwrite-legacy-index",
        action="store_true",
        help="Required if --index-dir points at legacy artifacts/index.",
    )
    args = parser.parse_args()

    instance_root = resolve_instance_root()
    schema = load_schema()
    events_csv = find_events_csv(instance_root)
    noise_csv = find_noise_csv(instance_root)

    index_dir = ensure_dir(ROOT / args.index_dir)
    legacy = (artifacts_dir() / "index").resolve()
    if index_dir.resolve() == legacy and not args.allow_overwrite_legacy_index:
        raise SystemExit(
            "Refusing to write into legacy artifacts/index without --allow-overwrite-legacy-index. "
            "Use default artifacts/index_full."
        )

    if args.splits_dir:
        out_splits = ensure_dir(ROOT / args.splits_dir)
    else:
        out_splits = ensure_dir(artifacts_dir() / "splits_full")

    header = list(pd.read_csv(events_csv, nrows=0).columns)
    id_col = schema.require("event_id")
    time_col = schema.require("origin_time")

    # Pass 1: collect unique event origin times (+ raw unique count)
    event_time: dict[str, pd.Timestamp] = {}
    n_raw_rows = 0
    for chunk in iter_csv_chunks(events_csv, chunksize=args.chunksize, usecols=[id_col, time_col]):
        n_raw_rows += len(chunk)
        chunk[time_col] = pd.to_datetime(chunk[time_col], utc=True, errors="coerce")
        chunk[id_col] = chunk[id_col].astype(str)
        g = chunk.groupby(id_col)[time_col].min()
        for eid, t in g.items():
            if eid not in event_time or (pd.notna(t) and t < event_time[eid]):
                event_time[eid] = t

    times = pd.Series(event_time).dropna().sort_values()
    n_raw_events = int(len(times))
    if args.max_events is not None:
        if args.max_events <= 0:
            raise SystemExit("--max-events must be positive when provided")
        keep_ids = set(times.index.astype(str)[: int(args.max_events)].tolist())
        truncated = True
    else:
        keep_ids = set(times.index.astype(str).tolist())
        truncated = False
    if not keep_ids:
        raise RuntimeError("No events found")

    # Pass 2: load full rows for selected events
    usecols = sorted({c for c in schema.mapping.values() if c is not None} | {"trace_dt_s"})
    usecols = [c for c in usecols if c in header]
    parts = []
    for chunk in iter_csv_chunks(events_csv, chunksize=args.chunksize, usecols=usecols):
        chunk = normalize_events_frame(chunk, schema)
        chunk = chunk[chunk["event_id"].astype(str).isin(keep_ids)]
        if len(chunk):
            parts.append(chunk)
    events = pd.concat(parts, ignore_index=True)

    n_index_events = int(events["event_id"].nunique())
    if (not truncated) and n_index_events < n_raw_events:
        raise RuntimeError(
            f"Index has {n_index_events} events but raw metadata has {n_raw_events} unique events "
            "and --max-events was not set. Refusing silent truncation."
        )

    splits = event_time_split(events)
    events = assign_split_column(events, splits)
    events["station_id"] = [
        station_id(str(n), str(s), loc, str(ch))
        for n, s, loc, ch in zip(
            events["network"],
            events["station"],
            events["location"].fillna("") if "location" in events.columns else [""] * len(events),
            events["channel_prefix"],
        )
    ]

    # Noise
    noise_header = list(pd.read_csv(noise_csv, nrows=0).columns)
    noise_usecols = [
        c
        for c in sorted({c for c in schema.mapping.values() if c is not None} | {"trace_dt_s"})
        if c in noise_header
    ]
    noise_parts: list[pd.DataFrame] = []
    n_noise = 0
    n_noise_raw_rows = 0
    for chunk in iter_csv_chunks(noise_csv, chunksize=args.chunksize, usecols=noise_usecols):
        chunk = normalize_noise_frame(chunk, schema)
        n_noise_raw_rows += len(chunk)
        if args.max_noise is not None and n_noise >= int(args.max_noise):
            break
        if args.max_noise is not None:
            remain = int(args.max_noise) - n_noise
            if remain <= 0:
                break
            chunk = chunk.head(remain)
        noise_parts.append(chunk)
        n_noise += len(chunk)
        if args.max_noise is not None and n_noise >= int(args.max_noise):
            break
    noise = pd.concat(noise_parts, ignore_index=True) if noise_parts else pd.DataFrame()
    if args.max_noise is None and len(noise) < n_noise_raw_rows:
        raise RuntimeError(
            f"Noise index has {len(noise)} rows but CSV streamed {n_noise_raw_rows} and --max-noise unset."
        )
    if len(noise):
        noise_splits = noise_time_split(noise)
        inv = {nm: sp for sp, names in noise_splits.items() for nm in names}
        noise["split"] = noise["trace_name"].astype(str).map(inv)

    station_cols = [
        c
        for c in [
            "network",
            "station",
            "location",
            "channel_prefix",
            "station_latitude",
            "station_longitude",
            "station_elevation_m",
            "station_id",
        ]
        if c in events.columns
    ]
    stations = events[station_cols].drop_duplicates(subset=["station_id"])

    save_event_splits(splits, out_splits)
    events.to_parquet(index_dir / "events.parquet", index=False)
    noise.to_parquet(index_dir / "noise.parquet", index=False)
    stations.to_parquet(index_dir / "stations.parquet", index=False)

    manifest = {
        "events_csv_path": str(events_csv),
        "noise_csv_path": str(noise_csv),
        "events_csv_sha256": _file_sha256(Path(events_csv)),
        "noise_csv_sha256": _file_sha256(Path(noise_csv)),
        "events_csv_data_rows_streamed": n_raw_rows,
        "raw_unique_events": n_raw_events,
        "index_unique_events": n_index_events,
        "index_unique_traces": int(events["trace_name"].nunique()),
        "index_n_traces": int(len(events)),
        "origin_time_min": str(events["origin_time"].min()),
        "origin_time_max": str(events["origin_time"].max()),
        "truncated": bool(truncated),
        "max_events": args.max_events,
        "max_noise": args.max_noise,
        "n_noise_index": int(len(noise)),
        "n_noise_csv_rows_streamed": int(n_noise_raw_rows),
        "n_stations": int(len(stations)),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "index_dir": str(index_dir),
        "splits_dir": str(out_splits),
        "legacy_index_untouched": index_dir.resolve() != legacy,
    }
    save_json(manifest, index_dir / "manifest.json")

    print(manifest)


if __name__ == "__main__":
    main()
