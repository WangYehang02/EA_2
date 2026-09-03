#!/usr/bin/env python
"""Build Stage 6 nested chronological splits from original TRAIN event pool only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir, write_lines


def sha_lines(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picker", type=int, default=3850)
    parser.add_argument("--ranker", type=int, default=1050)
    parser.add_argument("--dev", type=int, default=600)
    parser.add_argument("--confirm", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "splits")
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    train = events[events["split"] == "train"].copy()

    # Event-level chronological order (min origin_time); NO labels/SNR/preds used for ordering
    ev = (
        train.groupby("event_id", sort=False)
        .agg(origin_time=("origin_time", "min"), n_traces=("trace_name", "count"))
        .reset_index()
        .sort_values("origin_time")
    )
    eids = ev["event_id"].astype(str).tolist()
    n = len(eids)
    need = args.picker + args.ranker + args.dev + args.confirm
    if need != n:
        # scale proportionally if train size differs
        raise SystemExit(f"Expected {need} train events for configured sizes, found {n}. Adjust CLI sizes.")

    picker_ids = eids[: args.picker]
    ranker_ids = eids[args.picker : args.picker + args.ranker]
    dev_ids = eids[args.picker + args.ranker : args.picker + args.ranker + args.dev]
    confirm_ids = eids[args.picker + args.ranker + args.dev :]

    sets = {
        "stage6_picker_train": picker_ids,
        "stage6_ranker_train": ranker_ids,
        "stage6_dev": dev_ids,
        "stage6_internal_confirm": confirm_ids,
    }

    # Disjointness
    all_ids = []
    for name, ids in sets.items():
        write_lines(out / f"{name}_events.txt", ids)
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids)) == n

    # Trace manifests (selection uses only event membership — no label filtering)
    manifests = {}
    station_sets = {}
    for name, ids in sets.items():
        idset = set(ids)
        sub = train[train["event_id"].astype(str).isin(idset)].copy()
        traces = sorted(sub["trace_name"].astype(str).unique().tolist())
        write_lines(out / f"{name}_traces.txt", traces)
        manifests[name] = {
            "n_events": len(ids),
            "n_traces": len(traces),
            "event_hash": sha_lines(ids),
            "trace_hash": sha_lines(traces),
            "origin_time_min": str(pd.Timestamp(ev.set_index("event_id").loc[ids[0], "origin_time"])),
            "origin_time_max": str(pd.Timestamp(ev.set_index("event_id").loc[ids[-1], "origin_time"])),
        }
        sid = (
            sub["network"].astype(str)
            + "."
            + sub["station"].astype(str)
            + "."
            + sub.get("location", pd.Series([""] * len(sub))).fillna("").astype(str)
        )
        station_sets[name] = set(sid.astype(str))

    # Station overlap audit (allowed, but reported)
    overlap = {}
    names = list(sets.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap[f"{a}∩{b}"] = len(station_sets[a] & station_sets[b])

    audit = {
        "source_pool": "original chronological train only",
        "n_train_events_pool": n,
        "n_train_traces_pool": int(len(train)),
        "rationale": (
            "Adjusted from 70/15/10/5 to 3850/1050/600/1500 so internal_confirm≈1500 events "
            "(5% of 7000 would be only 350)."
        ),
        "sizes": {k: {"events": v["n_events"], "traces": v["n_traces"]} for k, v in manifests.items()},
        "manifests": manifests,
        "event_disjoint": True,
        "trace_disjoint": True,
        "station_overlap_counts": overlap,
        "selection_features_used": ["event_id", "origin_time"],
        "forbidden_selection_features": ["p/s labels", "snr", "model predictions"],
        "internal_confirm_sealed": True,
        "internal_confirm_inference_allowed_before_method_lock": False,
        "seed_note": args.seed,
    }
    save_json(audit, artifacts_dir() / "results" / "stage6" / "split_audit.json")
    # also copy into splits dir
    save_json(audit, out / "split_audit.json")
    print(json.dumps({"ok": True, "sizes": audit["sizes"]}, indent=2))


if __name__ == "__main__":
    main()
