#!/usr/bin/env python
"""Build held-out noise manifest for Phase B (excludes PhaseNet train noise; no confirm)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.full_splits import load_full_event_ids
from earthquake.stage6.phaseB import sha256_file, sha256_text
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    train_meta = pd.read_parquet(artifacts_dir() / "models" / "stage6" / "phasenet_ida_full_seed42" / "train_meta.parquet")
    train_noise = set(train_meta.loc[train_meta["is_noise"].astype(bool), "trace_name"].astype(str))
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    confirm_ev = set(load_full_event_ids("stage6_internal_confirm"))
    # noise event_id may be synthetic; still exclude any confirm id if present
    held = noise[~noise["trace_name"].astype(str).isin(train_noise)].copy()
    if "event_id" in held.columns:
        held = held[~held["event_id"].astype(str).isin(confirm_ev)]
    held = held.sample(n=min(args.n, len(held)), random_state=args.seed).reset_index(drop=True)
    held = held.sort_values("trace_name").reset_index(drop=True)
    held["s_arrival_sample"] = pd.NA
    held["p_arrival_sample"] = pd.NA
    held["event_id"] = held.get("event_id", "NOISE")

    out = ensure_dir(artifacts_dir() / "results" / "stage6")
    csv_path = out / "phaseB_noise_manifest.csv"
    cols = [c for c in ["trace_name", "event_id", "network", "station", "channel_prefix", "trace_start_time", "sampling_rate_hz", "s_arrival_sample", "p_arrival_sample"] if c in held.columns]
    held[cols].to_csv(csv_path, index=False)
    payload = {
        "n": int(len(held)),
        "seed": args.seed,
        "excluded_train_noise": len(train_noise),
        "held_out_pool": int((~noise["trace_name"].astype(str).isin(train_noise)).sum()),
        "csv_sha256": sha256_file(csv_path),
        "trace_list_sha256": sha256_text("\n".join(held["trace_name"].astype(str)) + "\n"),
        "confirm_excluded": True,
        "overlap_with_train_noise": 0,
    }
    assert set(held["trace_name"].astype(str)).isdisjoint(train_noise)
    save_json(payload, out / "phaseB_noise_manifest.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
