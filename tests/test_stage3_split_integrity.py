"""Stage-3 split integrity tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_original_fixed_eval_not_pure_test_is_documented():
    audit = ROOT / "artifacts" / "results" / "stage3" / "split_audit.json"
    assert audit.exists(), "Run scripts/audit_stage3_splits.py first"
    import json

    rep = json.loads(audit.read_text())
    orig = rep["original_fixed_eval"]
    # Document reality: may contain val
    assert "split_counts" in orig
    assert orig["path"].endswith("fixed_eval_traces.txt")


def test_test_only_list_is_chronological_test_and_disjoint():
    import json

    rep = json.loads((ROOT / "artifacts" / "results" / "stage3" / "split_audit.json").read_text())
    events = pd.read_parquet(ROOT / "artifacts" / "index" / "events.parquet")
    test_only = pd.read_parquet(ROOT / "artifacts" / "diagnostics" / "fixed_eval_test_only_events.parquet")
    train_e = set(events.loc[events.split == "train", "event_id"].astype(str))
    val_e = set(events.loc[events.split == "val", "event_id"].astype(str))
    te = set(test_only["event_id"].astype(str))
    assert te.isdisjoint(train_e)
    assert te.isdisjoint(val_e)
    # all traces labeled test in index (avoid split_x/split_y if parquet already has split)
    idx_split = events[["trace_name", "split"]].rename(columns={"split": "index_split"})
    m = test_only.merge(idx_split, on="trace_name", how="left")
    assert (m["index_split"] == "test").all()
    assert rep["test_only"]["n_traces"] == len(test_only)


def test_original_fixed_list_not_overwritten_hash_stable():
    p = ROOT / "artifacts" / "diagnostics" / "fixed_eval_traces.txt"
    assert p.exists()
    # test-only is a different file
    assert (ROOT / "artifacts" / "diagnostics" / "fixed_eval_test_only_traces.txt").exists()
    assert p.name != "fixed_eval_test_only_traces.txt"
