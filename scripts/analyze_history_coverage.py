#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.data.splits import load_event_splits
from earthquake.history.coverage import analyze_coverage
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--grid-sizes", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.50])
    args = parser.parse_args()

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    # Respect max-events in case index is larger
    event_times = events.groupby("event_id")["origin_time"].min().sort_values()
    keep = set(event_times.index.astype(str)[: args.max_events].tolist())
    events = events[events["event_id"].astype(str).isin(keep)].copy()
    splits = load_event_splits(artifacts_dir() / "splits")
    # filter splits
    splits = {k: [e for e in v if e in keep] for k, v in splits.items()}

    result = analyze_coverage(events, splits, grid_sizes=args.grid_sizes, min_history=args.min_history)
    out = ensure_dir(artifacts_dir() / "coverage")
    cov = result["coverage_df"]
    cov.to_csv(out / "coverage_by_grid.csv", index=False)
    save_json(result["recommendation"], out / "recommendation.json")

    # histograms
    fig, ax = plt.subplots(figsize=(8, 4))
    for gs, counts in result["hist_count_samples"].items():
        ax.hist(counts, bins=30, alpha=0.4, label=f"grid={gs}")
    ax.set_xlabel("history_count on test traces")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("History count histogram")
    fig.tight_layout()
    fig.savefig(out / "history_count_histogram.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for gs, mads in result["mad_samples"].items():
        if mads:
            ax.hist(mads, bins=30, alpha=0.4, label=f"grid={gs}")
    ax.set_xlabel("travel-time MAD (s)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Travel-time MAD distribution")
    fig.tight_layout()
    fig.savefig(out / "travel_time_mad.png", dpi=140)
    plt.close(fig)

    print(json.dumps({"recommendation": result["recommendation"], "coverage": cov.to_dict(orient="records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
