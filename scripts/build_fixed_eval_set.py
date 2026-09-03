#!/usr/bin/env python
from __future__ import annotations

"""Build fixed eval set: >=10000 event traces + >=2000 noise traces."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir
from earthquake.utils import ensure_dir, write_lines


def sample_event_traces(events: pd.DataFrame, n: int, seed: int = 0, max_per_event: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Prefer val+test for evaluation (unseen chronologically), fallback include train if needed
    pool = events[events["split"].isin(["val", "test"])].copy()
    if len(pool) < n:
        pool = events.copy()
    parts = []
    eids = pool["event_id"].astype(str).unique().tolist()
    rng.shuffle(eids)
    for eid in eids:
        g = pool[pool["event_id"].astype(str) == eid]
        take = min(len(g), max_per_event)
        parts.append(g.sample(n=take, random_state=int(rng.integers(0, 1e9))))
        if sum(map(len, parts)) >= n:
            break
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events-traces", type=int, default=10000)
    parser.add_argument("--n-noise", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index" / "noise.parquet")
    ev = sample_event_traces(events, args.n_events_traces, seed=args.seed)
    if len(noise):
        # prefer val/test noise
        nz_pool = noise[noise["split"].isin(["val", "test"])] if "split" in noise.columns else noise
        if len(nz_pool) < args.n_noise:
            nz_pool = noise
        nz = nz_pool.sample(n=min(args.n_noise, len(nz_pool)), random_state=args.seed)
    else:
        nz = pd.DataFrame(columns=["trace_name"])

    out = ensure_dir(artifacts_dir() / "diagnostics")
    names = ev["trace_name"].astype(str).tolist() + nz["trace_name"].astype(str).tolist()
    write_lines(out / "fixed_eval_traces.txt", names)
    ev.to_parquet(out / "fixed_eval_events.parquet", index=False)
    nz.to_parquet(out / "fixed_eval_noise.parquet", index=False)
    meta = {
        "n_event_traces": len(ev),
        "n_noise_traces": len(nz),
        "n_events": int(ev["event_id"].nunique()),
        "splits": ev["split"].value_counts().to_dict() if "split" in ev.columns else {},
    }
    (out / "fixed_eval_meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
