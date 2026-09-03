#!/usr/bin/env python
"""Audit full INSTANCE noise metadata + HDF5; write full_noise_counts.json. Does not touch legacy index."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.data.metadata import find_noise_csv, iter_csv_chunks, normalize_noise_frame
from earthquake.data.schema import SchemaMapping
from earthquake.config import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-h5", type=int, default=500)
    args = parser.parse_args()

    root = resolve_instance_root()
    noise_csv = find_noise_csv(root)
    noise_h5 = root / "noise" / "Instance_noise.hdf5"
    schema = SchemaMapping(**load_json(artifacts_dir() / "schema" / "schema_mapping.json"))

    n_rows = 0
    names: set[str] = set()
    stations: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    srates: Counter[str] = Counter()
    header = list(pd.read_csv(noise_csv, nrows=0).columns)
    usecols = [c for c in ["trace_name", "station_network_code", "station_code", "station_channels", "trace_dt_s", "trace_start_time"] if c in header]
    for chunk in iter_csv_chunks(noise_csv, chunksize=200_000, usecols=usecols):
        n_rows += len(chunk)
        tr = chunk["trace_name"].astype(str)
        names.update(tr.unique().tolist())
        if "station_network_code" in chunk.columns:
            key = chunk["station_network_code"].astype(str) + "." + chunk["station_code"].astype(str)
            stations.update(key.value_counts().to_dict())
        if "station_channels" in chunk.columns:
            channels.update(chunk["station_channels"].astype(str).value_counts().to_dict())
        if "trace_dt_s" in chunk.columns:
            dt = pd.to_numeric(chunk["trace_dt_s"], errors="coerce")
            for v in dt.dropna().unique():
                if float(v) > 0:
                    srates[str(round(1.0 / float(v), 6))] += int((dt == v).sum())

    # HDF5 readable count
    h5_n = None
    h5_ok = 0
    h5_fail = 0
    sample_names = list(names)[: args.sample_h5]
    with InstanceHDF5Reader(noise_h5).open() as reader:
        summary = reader.structure_summary()
        if summary.get("data_n_keys") is not None:
            h5_n = int(summary["data_n_keys"])
        for nm in sample_names:
            try:
                w = reader.read_waveform(nm)
                assert w.ndim == 2 and w.shape[0] == 3
                h5_ok += 1
            except Exception:
                h5_fail += 1

    # Legacy 20k source
    legacy = artifacts_dir() / "index" / "noise.parquet"
    legacy_info = None
    if legacy.exists():
        lg = pd.read_parquet(legacy)
        legacy_info = {
            "path": str(legacy),
            "n_rows": int(len(lg)),
            "source": "scripts/build_index.py historical default max_noise=min(2*max_events,20000) with max_events=10000 → 20000",
            "had_max_noise_default_limit": True,
        }

    out = {
        "noise_csv_path": str(noise_csv),
        "noise_hdf5_path": str(noise_h5),
        "metadata_rows": n_rows,
        "unique_trace_name": len(names),
        "hdf5_data_n_keys": h5_n,
        "hdf5_sample_read_ok": h5_ok,
        "hdf5_sample_read_fail": h5_fail,
        "hdf5_sample_n": len(sample_names),
        "n_stations_net_sta": len(stations),
        "channel_top20": dict(channels.most_common(20)),
        "sampling_rate_hz_counts": dict(srates),
        "legacy_20k_noise_index": legacy_info,
        "full_index_target": "artifacts/index_full/noise.parquet",
        "legacy_index_not_overwritten": True,
        "build_index_max_noise_default_now": None,
        "note": "Current build_index.py --max-noise defaults to None (all noise). Legacy 20k came from old default cap.",
    }
    save_json(out, artifacts_dir() / "results" / "stage6" / "full_noise_counts.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
