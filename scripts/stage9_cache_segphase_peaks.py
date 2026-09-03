#!/usr/bin/env python
"""Cache SegPhase S peaks on a manifest (sharded). Stores peak sample+prob for offline thresholding.

For each trace, also stores top-5 candidates at min_height=0.05 for complementarity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import resolve_instance_root
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage9.segphase_adapter import SegPhaseConfig, SegPhasePicker
from earthquake.utils import ensure_dir


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_shard(args) -> None:
    manifest = pd.read_csv(ROOT / args.manifest)
    n = len(manifest)
    idxs = list(range(args.rank, n, args.world_size))
    out = ensure_dir(ROOT / args.out_dir)
    shard = out / f"shard_{args.rank:02d}.parquet"
    rows = []
    done: set[str] = set()
    if shard.exists():
        prev = pd.read_parquet(shard)
        done = set(prev.trace_name.astype(str))
        rows = prev.to_dict("records")

    cfg = SegPhaseConfig(device=args.device, window_scheme=args.window_scheme, peak_height=0.05)
    picker = SegPhasePicker(cfg)
    reader = InstanceHDF5Reader(resolve_instance_root() / "events" / "Instance_events_counts.hdf5").open()
    try:
        for j, i in enumerate(idxs):
            row = manifest.iloc[i]
            tn = str(row.trace_name)
            if tn in done:
                continue
            try:
                wave = reader.read_waveform(tn)
                # store peaks at low height; thresholds applied offline
                pred = picker.predict_s(wave, threshold=0.05, top_k=5)
                top = pred["candidates"][0] if pred["candidates"] else None
                rec = {
                    "trace_name": tn,
                    "event_id": str(row.event_id),
                    "sampling_rate_hz": float(row.sampling_rate_hz),
                    "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                    "pred_s_sample": float(top["sample"]) if top else np.nan,
                    "s_peak_probability": float(top["prob"]) if top else np.nan,
                    "n_peaks_h005": int(pred.get("n_peaks") or 0),
                    "window_scheme": args.window_scheme,
                    "cand_json": json.dumps(pred["candidates"]),
                }
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "trace_name": tn,
                    "event_id": str(row.event_id),
                    "sampling_rate_hz": float(row.sampling_rate_hz),
                    "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                    "pred_s_sample": np.nan,
                    "s_peak_probability": np.nan,
                    "n_peaks_h005": 0,
                    "window_scheme": args.window_scheme,
                    "cand_json": "[]",
                    "error": repr(exc),
                }
                if j < 5:
                    print({"skip": tn, "err": repr(exc)}, flush=True)
            rows.append(rec)
            done.add(tn)
            if len(rows) % 100 == 0:
                _atomic_parquet(pd.DataFrame(rows), shard)
                print({"rank": args.rank, "n": len(rows), "of": len(idxs)}, flush=True)
    finally:
        reader.close()
    _atomic_parquet(pd.DataFrame(rows), shard)
    (out / f"shard_{args.rank:02d}.DONE").write_text(f"{len(rows)}\n")
    print({"rank": args.rank, "done": len(rows)}, flush=True)


def merge(out_dir: Path, world: int, scheme: str) -> Path:
    parts = []
    for r in range(world):
        p = out_dir / f"shard_{r:02d}.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("trace_name", keep="last")
    final = out_dir / f"segphase_{scheme}_peaks.parquet"
    tmp = final.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    (out_dir / "MERGE.DONE").write_text(json.dumps({"n": len(df), "sha256": sha256_file(final), "scheme": scheme}) + "\n")
    print({"merged": str(final), "n": len(df)})
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-scheme", choices=["A", "B"], default="A")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    out = ensure_dir(ROOT / args.out_dir)
    if args.merge_only:
        merge(out, args.world_size, args.window_scheme)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
