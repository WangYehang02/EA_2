#!/usr/bin/env python
"""Build Stage 6 FULL chronological nested splits into splits_full/ (does not replace pilot splits/)."""
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
from earthquake.stage6.full_splits import (
    FULL_SUBSETS,
    assert_confirm_never_used,
    full_stage6_paths,
    write_full_confirm_seal,
)
from earthquake.utils import ensure_dir, write_lines


def _sha_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode())
        h.update(b"\n")
    return h.hexdigest()


def _load_prior_used_event_ids() -> set[str]:
    """Events that appeared in Stage1–5 working universe (legacy 10k index + registry)."""
    used: set[str] = set()
    legacy = artifacts_dir() / "index" / "events.parquet"
    if legacy.exists():
        ev = pd.read_parquet(legacy, columns=["event_id"])
        used |= set(ev["event_id"].astype(str))
    for name in ("train_events.txt", "val_events.txt", "test_events.txt"):
        p = artifacts_dir() / "splits" / name
        if p.exists():
            used |= {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    reg = artifacts_dir() / "results" / "stage6" / "prior_event_usage_registry.csv"
    if reg.exists():
        df = pd.read_csv(reg, usecols=["event_id"])
        used |= set(df["event_id"].astype(str))
    return used


def _subset_stats(df: pd.DataFrame) -> dict:
    s = pd.to_numeric(df.get("s_arrival_sample"), errors="coerce")
    p = pd.to_numeric(df.get("p_arrival_sample"), errors="coerce")
    per = df.groupby(df["event_id"].astype(str)).size()
    return {
        "n_events": int(df["event_id"].nunique()),
        "n_traces": int(len(df)),
        "n_s_labelled_traces": int(s.notna().sum()) if s is not None else 0,
        "n_p_labelled_traces": int(p.notna().sum()) if p is not None else 0,
        "n_ps_both_labelled_traces": int((p.notna() & s.notna()).sum()) if p is not None else 0,
        "n_stations": int(df["station_id"].nunique()) if "station_id" in df.columns else None,
        "origin_time_min": str(df["origin_time"].min()),
        "origin_time_max": str(df["origin_time"].max()),
        "traces_per_event": {
            "min": int(per.min()) if len(per) else 0,
            "median": float(per.median()) if len(per) else 0,
            "mean": float(per.mean()) if len(per) else 0,
            "max": int(per.max()) if len(per) else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-parquet", default="artifacts/index_full/events.parquet")
    parser.add_argument("--picker-ratio", type=float, default=0.70)
    parser.add_argument("--ranker-ratio", type=float, default=0.15)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    # remainder -> confirm (~0.05)
    parser.add_argument("--confirm-target-frac", type=float, default=0.05)
    args = parser.parse_args()

    events_path = ROOT / args.events_parquet
    if not events_path.exists():
        raise SystemExit(f"Missing full events index: {events_path}. Run scripts/build_index.py first.")

    # Selection itself uses only event_id + origin_time. Extra columns are for post-hoc audit stats only.
    available = list(pd.read_parquet(events_path).columns)
    select_cols = [c for c in ("event_id", "origin_time", "trace_name") if c in available]
    audit_cols = [c for c in ("station_id", "p_arrival_sample", "s_arrival_sample") if c in available]
    ev = pd.read_parquet(events_path, columns=list(dict.fromkeys(select_cols + audit_cols)))
    ev["event_id"] = ev["event_id"].astype(str)
    ev["origin_time"] = pd.to_datetime(ev["origin_time"], utc=True)

    event_times = ev.groupby("event_id")["origin_time"].min().sort_values()
    all_ids = event_times.index.astype(str).tolist()
    n = len(all_ids)
    n_picker = int(round(n * args.picker_ratio))
    n_ranker = int(round(n * args.ranker_ratio))
    n_dev = int(round(n * args.dev_ratio))
    n_confirm = n - n_picker - n_ranker - n_dev
    # adjust rounding to exact cover
    while n_picker + n_ranker + n_dev + n_confirm > n:
        n_confirm -= 1
    while n_picker + n_ranker + n_dev + n_confirm < n:
        n_confirm += 1

    prior_used = _load_prior_used_event_ids()

    # Initial chronological cut
    picker_ids = all_ids[:n_picker]
    ranker_ids = all_ids[n_picker : n_picker + n_ranker]
    dev_ids = all_ids[n_picker + n_ranker : n_picker + n_ranker + n_dev]
    confirm_ids = all_ids[n_picker + n_ranker + n_dev :]

    # Confirm must be entirely never-used in Stage1–5. If contaminated, drop and backfill from earlier never-used.
    confirm_set = list(confirm_ids)
    contaminated = [e for e in confirm_set if e in prior_used]
    if contaminated:
        confirm_clean = [e for e in confirm_set if e not in prior_used]
        need = len(confirm_set) - len(confirm_clean)
        # candidates: earlier than confirm window, never-used, not already in picker/ranker/dev
        assigned = set(picker_ids) | set(ranker_ids) | set(dev_ids) | set(confirm_clean)
        # walk backward from just before confirm window
        pool = [e for e in reversed(all_ids[: n_picker + n_ranker + n_dev]) if e not in prior_used and e not in assigned]
        take = pool[:need]
        # remove taken from earlier splits (prefer steal from end of dev, then ranker, then picker)
        steal = set(take)
        def _filter(ids):
            return [e for e in ids if e not in steal]

        picker_ids = _filter(picker_ids)
        ranker_ids = _filter(ranker_ids)
        dev_ids = _filter(dev_ids)
        # re-append stolen to confirm (keep chronological order within confirm)
        confirm_ids = sorted(confirm_clean + take, key=lambda e: event_times.loc[e])
        # refill earlier splits to target sizes from remaining earliest never-assigned
        assigned = set(picker_ids) | set(ranker_ids) | set(dev_ids) | set(confirm_ids)
        remaining = [e for e in all_ids if e not in assigned]
        # fill picker, ranker, dev in chronological order from remaining
        while len(picker_ids) < n_picker and remaining:
            picker_ids.append(remaining.pop(0))
        while len(ranker_ids) < n_ranker and remaining:
            ranker_ids.append(remaining.pop(0))
        while len(dev_ids) < n_dev and remaining:
            dev_ids.append(remaining.pop(0))
        # any leftover go to picker_train (development) — never confirm
        picker_ids.extend(remaining)
        picker_ids = sorted(set(picker_ids), key=lambda e: event_times.loc[e])
        ranker_ids = sorted(set(ranker_ids), key=lambda e: event_times.loc[e])
        dev_ids = sorted(set(dev_ids), key=lambda e: event_times.loc[e])
    else:
        contaminated = []

    subsets = {
        "stage6_picker_train": picker_ids,
        "stage6_ranker_train": ranker_ids,
        "stage6_dev": dev_ids,
        "stage6_internal_confirm": confirm_ids,
    }

    # Disjoint checks
    for a in FULL_SUBSETS:
        for b in FULL_SUBSETS:
            if a >= b:
                continue
            if set(subsets[a]) & set(subsets[b]):
                raise RuntimeError(f"event overlap {a} vs {b}")

    assert_confirm_never_used(subsets["stage6_internal_confirm"], prior_used)

    paths = full_stage6_paths()
    out = ensure_dir(paths["splits"])
    manifests = {}
    for name, ids in subsets.items():
        df = ev[ev["event_id"].isin(ids)].copy()
        # traces follow events
        tr = df["trace_name"].astype(str).tolist()
        write_lines(out / f"{name}_events.txt", ids)
        write_lines(out / f"{name}_traces.txt", tr)
        manifests[name] = {
            **_subset_stats(df),
            "events_sha256": _sha_lines(ids),
            "traces_sha256": _sha_lines(tr),
        }

    # trace disjoint
    trace_sets = {
        name: set(ev[ev["event_id"].isin(subsets[name])]["trace_name"].astype(str)) for name in FULL_SUBSETS
    }
    for a in FULL_SUBSETS:
        for b in FULL_SUBSETS:
            if a >= b:
                continue
            if trace_sets[a] & trace_sets[b]:
                raise RuntimeError(f"trace overlap {a} vs {b}")

    # chronological order check
    def tmax(ids):
        return event_times.loc[ids].max() if ids else None

    def tmin(ids):
        return event_times.loc[ids].min() if ids else None

    chrono_ok = (
        tmax(picker_ids) <= tmin(ranker_ids)
        and tmax(ranker_ids) <= tmin(dev_ids)
        and tmax(dev_ids) <= tmin(confirm_ids)
    )

    audit = {
        "protocol": "stage6_full_chronological_event_nested",
        "n_total_events": n,
        "ratios": {
            "picker": args.picker_ratio,
            "ranker": args.ranker_ratio,
            "dev": args.dev_ratio,
            "confirm_remainder": round(1.0 - args.picker_ratio - args.ranker_ratio - args.dev_ratio, 4),
        },
        "sizes": {k: {"events": v["n_events"], "traces": v["n_traces"]} for k, v in manifests.items()},
        "manifests": manifests,
        "event_disjoint": True,
        "trace_disjoint": True,
        "chronological_non_overlapping_windows": bool(chrono_ok),
        "confirm_all_never_used_in_stage1_5": True,
        "confirm_contaminated_excluded_n": len(contaminated),
        "prior_used_universe_n": len(prior_used),
        "selection_used_labels_or_snr_or_predictions": False,
        "pilot_splits_untouched": True,
        "pilot_splits_dir": str(artifacts_dir() / "results" / "stage6" / "splits"),
        "full_splits_dir": str(out),
        "history_protocol": {
            "primary_confirm": "frozen_history_from_picker_train_only",
            "distance_mlp_fit_on": "stage6_picker_train",
            "temporal_history_store_on": "stage6_picker_train",
            "phasenet_train_on": "stage6_picker_train",
            "ranker_train_on": "stage6_ranker_train",
            "dev_for": "checkpoint_and_hyperparams_only",
            "confirm_in_fitting": False,
            "shared_history_snapshot_for": ["ranker_train", "dev", "confirm"],
            "rolling_online_history": "secondary_future_only_not_primary_confirm",
        },
    }
    save_json(audit, artifacts_dir() / "results" / "stage6" / "full_split_audit.json")
    save_json(audit, out / "full_split_audit.json")

    # Seal confirm immediately
    write_full_confirm_seal(
        events_sha256=manifests["stage6_internal_confirm"]["events_sha256"],
        traces_sha256=manifests["stage6_internal_confirm"]["traces_sha256"],
        n_events=manifests["stage6_internal_confirm"]["n_events"],
        n_traces=manifests["stage6_internal_confirm"]["n_traces"],
    )
    print(json.dumps({k: audit["sizes"][k] for k in audit["sizes"]}, indent=2))
    print({"confirm_sealed": True, "chrono_ok": chrono_ok, "contaminated_excluded": len(contaminated)})


if __name__ == "__main__":
    main()
