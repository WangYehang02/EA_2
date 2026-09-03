#!/usr/bin/env python
"""Build fixed Phase B eval manifest from splits_full stage6_dev S-labelled traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, load_full_event_ids, load_full_trace_names
from earthquake.stage6.phaseB import build_s_labelled_eval_manifest, sha256_file, sha256_text
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-traces", type=int, default=-1, help="-1 = all S-labelled")
    parser.add_argument("--max-traces-per-event", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--force-subset", action="store_true", help="Allow capped subset even if full fits time")
    args = parser.parse_args()

    # Never touch confirm
    seal = artifacts_dir() / "results" / "stage6" / "splits_full" / "CONFIRM_SEALED"
    assert seal.exists(), "CONFIRM_SEALED missing"
    try:
        assert_full_confirm_access_allowed(purpose="phaseB_manifest_guard")
        raise SystemExit("method_lock unexpectedly present — abort")
    except RuntimeError:
        pass

    traces = load_full_trace_names("stage6_dev")
    events_ids = load_full_event_ids("stage6_dev")
    # refuse confirm names
    conf = set(load_full_trace_names("stage6_internal_confirm"))
    assert not (set(traces) & conf)

    ev = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    max_tr = None if args.max_traces < 0 else args.max_traces
    max_pe = None if args.max_traces_per_event < 0 else args.max_traces_per_event
    df, summary = build_s_labelled_eval_manifest(
        ev,
        split_trace_names=traces,
        split_event_ids=events_ids,
        seed=args.seed,
        max_traces=max_tr,
        max_traces_per_event=max_pe,
    )

    out_dir = ensure_dir(artifacts_dir() / "results" / "stage6")
    csv_path = out_dir / "phaseB_eval_manifest.csv"
    json_path = out_dir / "phaseB_eval_manifest.json"
    cols = [
        c
        for c in [
            "trace_name",
            "event_id",
            "network",
            "station",
            "channel_prefix",
            "origin_time",
            "trace_start_time",
            "sampling_rate_hz",
            "s_arrival_sample",
            "p_arrival_sample",
            "distance_km",
            "hyp_distance_km",
            "source_depth_km",
            "snr_db",
        ]
        if c in df.columns
    ]
    df[cols].to_csv(csv_path, index=False)
    manifest_sha = sha256_file(csv_path)
    payload = {
        **summary,
        "csv": str(csv_path),
        "csv_sha256": manifest_sha,
        "splits": "artifacts/results/stage6/splits_full",
        "index": "artifacts/index_full",
        "pilot_10k_excluded": True,
        "confirm_excluded": True,
        "selection_used_predictions": False,
        "trace_list_sha256": sha256_text("\n".join(df["trace_name"].tolist()) + "\n"),
        "event_list_sha256": sha256_text("\n".join(sorted(df["event_id"].unique())) + "\n"),
    }
    save_json(payload, json_path)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
