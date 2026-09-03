#!/usr/bin/env python
"""Build Stage-4 confirmatory holdout from unused INSTANCE test events only.

Selection uses ONLY event_id, origin_time, waveform availability, counts.
Never reads P/S labels, SNR, or model predictions during selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.utils import ensure_dir


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_used_event_ids(events: pd.DataFrame) -> tuple[set[str], dict]:
    """Union of event_ids that entered Stage1–3 debug/val/test/fixed-eval lists or result caches."""
    ad = artifacts_dir()
    sources: dict[str, int] = {}
    used: set[str] = set()

    def add_from_parquet(path: Path, label: str) -> None:
        if not path.exists():
            return
        df = pd.read_parquet(path)
        if "event_id" not in df.columns and "trace_name" in df.columns:
            df = df.merge(events[["trace_name", "event_id"]], on="trace_name", how="left")
        if "event_id" not in df.columns:
            return
        s = set(df["event_id"].astype(str))
        sources[label] = len(s)
        used.update(s)

    def add_from_trace_list(path: Path, label: str) -> None:
        if not path.exists():
            return
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        m = events[events.trace_name.astype(str).isin(names)]
        s = set(m.event_id.astype(str))
        sources[label] = len(s)
        used.update(s)

    # Explicit eval / fixed lists
    add_from_parquet(ad / "diagnostics" / "fixed_eval_events.parquet", "fixed_eval_events")
    add_from_parquet(ad / "diagnostics" / "fixed_eval_test_only_events.parquet", "fixed_eval_test_only")
    add_from_trace_list(ad / "diagnostics" / "fixed_eval_traces.txt", "fixed_eval_traces.txt")
    add_from_trace_list(ad / "diagnostics" / "fixed_eval_test_only_traces.txt", "fixed_eval_test_only_traces.txt")

    # Stage2/3 evaluation outputs
    add_from_parquet(ad / "results" / "stage2" / "candidate_rescoring_picks.parquet", "stage2_picks")
    add_from_parquet(ad / "results" / "stage2" / "phasenet_fixed_cache.parquet", "stage2_phasenet_cache")
    add_from_parquet(ad / "results" / "stage3" / "learned_gate_picks_test.parquet", "stage3_learned_picks")
    add_from_parquet(ad / "candidates" / "stage3_test.zarr" / "meta.parquet", "stage3_test_zarr")
    # debug baselines may include small test subset
    add_from_parquet(ad / "results" / "stage3" / "gate_baselines_debug.parquet", "stage3_baselines_debug")

    # Note: stage3 train/val zarr events are chronological train/val — excluded automatically
    # when intersecting with test split; still recorded for audit completeness.
    add_from_parquet(ad / "candidates" / "stage3_train.zarr" / "meta.parquet", "stage3_train_zarr")
    add_from_parquet(ad / "candidates" / "stage3_val.zarr" / "meta.parquet", "stage3_val_zarr")

    return used, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/confirmatory.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    hold = ensure_dir(out / "holdout")

    # Selection columns ONLY — do not touch arrival/SNR/pred columns for decisions
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    select_cols = ["event_id", "trace_name", "origin_time", "split", "station_id", "n_samples"]
    for c in select_cols:
        if c not in events.columns:
            raise SystemExit(f"Missing required index column for holdout selection: {c}")

    used, sources = collect_used_event_ids(events)
    test = events[events["split"] == "test"][["event_id", "trace_name", "origin_time", "split", "station_id", "n_samples"]].copy()
    test["event_id"] = test["event_id"].astype(str)
    test["origin_time"] = pd.to_datetime(test["origin_time"], utc=True)

    unused = test[~test["event_id"].isin(used)].copy()
    # Prefer newest events (temporal holdout)
    ev = (
        unused.groupby("event_id", as_index=False)
        .agg(origin_time=("origin_time", "min"), n_traces=("trace_name", "size"), n_stations=("station_id", "nunique"))
        .sort_values("origin_time", ascending=False)
    )

    target_tr = int(cfg.get("holdout_target_traces", 20000))
    target_ev = int(cfg.get("holdout_target_events", 1000))
    insufficient = True
    selected_e: list[str] = []
    cum = 0
    for _, r in ev.iterrows():
        selected_e.append(str(r.event_id))
        cum += int(r.n_traces)
        if cum >= target_tr and len(selected_e) >= target_ev:
            insufficient = False
            break
    # If still insufficient after all unused, keep all
    if insufficient:
        selected_e = ev["event_id"].astype(str).tolist()
        cum = int(ev["n_traces"].sum()) if len(ev) else 0

    holdout = unused[unused.event_id.isin(selected_e)].copy()
    # Attach full metadata AFTER selection (for evaluation), including labels — selection already finalized
    full = events[events.trace_name.isin(holdout.trace_name)].copy()
    full = full.sort_values(["origin_time", "trace_name"]).reset_index(drop=True)

    # Writes
    list_path = hold / "holdout_traces.txt"
    list_path.write_text("\n".join(full.trace_name.astype(str).tolist()) + ("\n" if len(full) else ""))
    full.to_parquet(hold / "holdout_events.parquet", index=False)
    pd.DataFrame({"event_id": sorted(used)}).to_parquet(hold / "excluded_event_ids.parquet", index=False)
    pd.DataFrame({"event_id": selected_e}).to_parquet(hold / "holdout_event_ids.parquet", index=False)

    # Leakage checks
    train_e = set(events.loc[events.split == "train", "event_id"].astype(str))
    val_e = set(events.loc[events.split == "val", "event_id"].astype(str))
    ho_e = set(full.event_id.astype(str))
    audit = {
        "n_traces": int(len(full)),
        "n_events": int(full.event_id.nunique()),
        "n_stations": int(full.station_id.nunique()) if "station_id" in full.columns else None,
        "time_start": str(full.origin_time.min()) if len(full) else None,
        "time_end": str(full.origin_time.max()) if len(full) else None,
        "all_split_test": bool((full.split == "test").all()) if len(full) else True,
        "overlap_used_events": int(len(ho_e & used)),
        "overlap_train_events": int(len(ho_e & train_e)),
        "overlap_val_events": int(len(ho_e & val_e)),
        "duplicate_trace_names": int(full.trace_name.duplicated().sum()),
        "unused_test_events_available": int(ev.event_id.nunique()) if len(ev) else 0,
        "unused_test_traces_available": int(len(unused)),
        "target_traces": target_tr,
        "target_events": target_ev,
        "insufficient_unused_events": bool(insufficient),
        "selection_policy": "all unused chronological-test events, newest-first; labels/SNR/preds not used for selection",
        "used_event_sources": sources,
        "used_events_total": int(len(used)),
        "list_sha256": _sha256_text(list_path.read_text()),
        "parquet_sha256": _sha256_file(hold / "holdout_events.parquet"),
        "warning": (
            "Holdout is far below target 20k traces / 1k events because Stage-3 test-only already consumed "
            "1482/1500 chronological test events. Confirmatory power is limited; report this explicitly."
            if insufficient
            else None
        ),
    }
    assert audit["overlap_used_events"] == 0
    assert audit["overlap_train_events"] == 0
    assert audit["overlap_val_events"] == 0
    assert audit["duplicate_trace_names"] == 0

    save_json(audit, hold / "split_audit.json")
    save_json(audit, out / "holdout_split_audit.json")
    pd.DataFrame(
        [
            {
                "split": "stage4_holdout",
                "n_traces": audit["n_traces"],
                "n_events": audit["n_events"],
                "n_stations": audit["n_stations"],
                "time_start": audit["time_start"],
                "time_end": audit["time_end"],
                "insufficient": audit["insufficient_unused_events"],
            }
        ]
    ).to_csv(out / "tables" / "data_split_audit.csv", index=False)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
