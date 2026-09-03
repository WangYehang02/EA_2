#!/usr/bin/env python
"""Validate Stage6 ID-A best.pt for Phase B (never loads last.pt)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.stage6.full_splits import load_full_trace_names
from earthquake.stage6.phaseB import validate_ida_best_checkpoint
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-n", type=int, default=32)
    args = parser.parse_args()

    ckpt = ROOT / args.ckpt
    if ckpt.name != "best.pt":
        raise SystemExit("Refusing any checkpoint other than best.pt")

    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    # fixed smoke: first N of sorted stage6_dev traces present in index (not head of raw file alone)
    names = load_full_trace_names("stage6_dev")
    # deterministic smoke list saved for reuse
    smoke_path = ensure_dir(artifacts_dir() / "results" / "stage6") / "phaseB_ida_smoke_traces.txt"
    if not smoke_path.exists():
        smoke_path.write_text("\n".join(names[: max(args.smoke_n, 32)]) + "\n")
    smoke_names = [ln.strip() for ln in smoke_path.read_text().splitlines() if ln.strip()][: args.smoke_n]
    rows = events[events["trace_name"].astype(str).isin(smoke_names)].copy()
    rows = rows.set_index("trace_name").loc[smoke_names].reset_index()

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    try:
        result = validate_ida_best_checkpoint(
            ckpt,
            smoke_rows=rows,
            waveform_reader=reader,
            device=args.device,
            expected_epoch=14,
        )
    finally:
        reader.close()

    out = artifacts_dir() / "results" / "stage6" / "phaseB_ida_checkpoint_validation.json"
    save_json(result, out)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
