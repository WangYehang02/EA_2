#!/usr/bin/env python
"""Measure Stage-4 compute cost (CPU/GPU, timings, sizes)."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    tables = ensure_dir(out / "tables")
    hist = artifacts_dir() / "history" / "residual_history_features_frozen.parquet"
    baseline = artifacts_dir() / "history" / "travel_time_baseline.pkl"
    timing = {}
    for p in (out / "cache").glob("phasenet_*_timing.json"):
        import json

        timing[p.stem] = json.loads(p.read_text())

    # parameter counts
    n_mlp = 0
    try:
        import pickle

        with open(baseline, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "model"):
            n_mlp = sum(int(getattr(c, "size", 0) or 0) for c in getattr(obj.model, "coefs_", []))
            n_mlp += sum(int(getattr(c, "size", 0) or 0) for c in getattr(obj.model, "intercepts_", []))
    except Exception:
        n_mlp = -1

    rows = [
        {"item": "history_index_disk_bytes", "value": _size(hist)},
        {"item": "travel_time_baseline_bytes", "value": _size(baseline)},
        {"item": "distance_mlp_approx_params", "value": n_mlp},
        {"item": "fixed_rescore_trainable_params", "value": 0},
        {"item": "candidate_k", "value": 5},
        {"item": "shrinkage_k", "value": 50},
        {"item": "gpu", "value": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        {"item": "cuda_available", "value": int(torch.cuda.is_available())},
        {"item": "platform", "value": platform.platform()},
        {"item": "cpu", "value": platform.processor() or platform.machine()},
    ]
    if timing:
        for k, v in timing.items():
            rows.append({"item": f"{k}_sec_per_trace", "value": v.get("sec_per_trace")})
            rows.append({"item": f"{k}_n", "value": v.get("n")})
    # rescore microbench from picks if present
    picks = out / "holdout_picks.parquet"
    if picks.exists():
        df = pd.read_parquet(picks)
        rows.append({"item": "holdout_n_traces", "value": len(df)})
        rows.append({"item": "holdout_n_events", "value": df.event_id.nunique()})

    tab = pd.DataFrame(rows)
    tab.to_csv(tables / "compute_cost.csv", index=False)
    save_json({"rows": rows}, out / "compute_cost.json")
    print(tab.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
