#!/usr/bin/env python
"""Fit train-only travel-time baselines; select on val; freeze for test."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.history.travel_time_baseline import evaluate_baseline, fit_all_and_select
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_fixed.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    seed = int(cfg.get("seed", 42))

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    train = events[events["split"] == "train"].copy()
    val = events[events["split"] == "val"].copy()
    test = events[events["split"] == "test"].copy()

    best, val_table = fit_all_and_select(train, val, seed=seed)
    test_met = evaluate_baseline(best, test)
    out = ensure_dir(artifacts_dir() / "results" / "stage2")
    val_table.to_csv(out / "travel_time_baseline_val.csv", index=False)
    payload = {
        "selected_kind": best.kind,
        "val_comparison": val_table.to_dict(orient="records"),
        "test_metrics_frozen": test_met,
        "meta": best.meta,
        "note": "Fitted on chronological train only; selected on val; test metrics are frozen reporting only.",
    }
    save_json(payload, out / "travel_time_baseline_selection.json")
    with open(out / "travel_time_baseline.pkl", "wb") as f:
        pickle.dump(best, f)
    print(payload)


if __name__ == "__main__":
    main()
