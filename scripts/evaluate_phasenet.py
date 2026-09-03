#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.metrics import groupby_history_metrics, metrics_from_pick_frame
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    thr = float(cfg.get("pick_threshold", 0.3))

    picks_path = artifacts_dir() / "phasenet" / "phasenet_picks.parquet"
    picks = pd.read_parquet(picks_path)

    # Apply threshold (peak already chosen; mark undetected if below thr)
    for phase in ("p", "s"):
        prob_col = f"{phase}_peak_probability"
        pred_col = f"pred_{phase}_sample"
        mask = picks[prob_col] < thr
        picks.loc[mask, pred_col] = float("nan")

    out = ensure_dir(artifacts_dir() / "results")
    metrics = {}
    for split in sorted(picks["split"].unique()):
        sub = picks[picks["split"] == split]
        metrics[split] = {
            "P": metrics_from_pick_frame(sub, phase="p"),
            "S": metrics_from_pick_frame(sub, phase="s"),
            "n": len(sub),
        }
    save_json(metrics, out / "phasenet_metrics.json")
    picks.to_parquet(out / "picks.parquet", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
