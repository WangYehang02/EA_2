#!/usr/bin/env python
"""Cache PhaseNet peak picks (sample + peak_prob) for Stage 7A — resume-safe shards."""

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
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.phaseB import sha256_file
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
    done = out / f"shard_{args.rank:02d}.DONE"
    rows = []
    done_names: set[str] = set()
    if shard.exists():
        prev = pd.read_parquet(shard)
        done_names = set(prev.trace_name.astype(str))
        rows = prev.to_dict("records")

    ref = SeisBenchPhaseNetReference(weight=args.weight, device=args.device)
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    try:
        for j, i in enumerate(idxs):
            row = manifest.iloc[i]
            tn = str(row.trace_name)
            if tn in done_names:
                continue
            try:
                wave = reader.read_waveform(tn)
                # Peak sample/prob do not require full proba remap (large speed win).
                pred = ref.predict_row(wave, row, remap_to_waveform=False)
                s_samp = float(pred["s_pred_sample_on_waveform"])
                s_prob = float(pred["s_peak_probability"])
                p_samp = float(pred["p_pred_sample_on_waveform"])
                p_prob = float(pred["p_peak_probability"])
            except Exception as exc:  # noqa: BLE001
                s_samp = p_samp = float("nan")
                s_prob = p_prob = float("nan")
                if j < 5:
                    print({"skip": tn, "err": repr(exc)}, flush=True)
            rows.append(
                {
                    "trace_name": tn,
                    "event_id": str(row.event_id),
                    "sampling_rate_hz": float(row.sampling_rate_hz),
                    "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                    "pred_s_sample": s_samp,
                    "s_peak_probability": s_prob,
                    "pred_p_sample": p_samp,
                    "p_peak_probability": p_prob,
                    "weight": args.weight,
                }
            )
            done_names.add(tn)
            if len(rows) % 200 == 0:
                _atomic_parquet(pd.DataFrame(rows), shard)
                print({"rank": args.rank, "n": len(rows), "of": len(idxs)}, flush=True)
    finally:
        reader.close()
    _atomic_parquet(pd.DataFrame(rows), shard)
    done.write_text(f"{len(rows)}\n")
    print({"rank": args.rank, "done": len(rows)}, flush=True)


def merge(out_dir: Path, world: int, weight: str) -> Path:
    parts = []
    for r in range(world):
        p = out_dir / f"shard_{r:02d}.parquet"
        if not p.exists():
            raise SystemExit(f"missing shard {p}")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("trace_name", keep="last")
    final = out_dir / f"phasenet_{weight}_peaks.parquet"
    tmp = final.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    (out_dir / "MERGE.DONE").write_text(json.dumps({"n": len(df), "sha256": sha256_file(final)}) + "\n")
    print({"merged": str(final), "n": len(df)})
    return final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True, choices=["ethz", "scedc", "stead"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()
    out = ensure_dir(ROOT / args.out_dir)
    if args.merge_only:
        merge(out, args.world_size, args.weight)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
