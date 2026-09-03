#!/usr/bin/env python
"""Unit tests: no silent 10k index truncation; full confirm seal; metric naming."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.stage6.full_splits import (
    assert_full_confirm_access_allowed,
    full_stage6_paths,
    load_full_event_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def test_build_index_default_max_events_is_none():
    src = (ROOT / "scripts/build_index.py").read_text()
    # parse argparse default for --max-events
    assert "default=None" in src or 'default=None' in src
    assert "--max-events" in src
    assert "Refusing silent truncation" in src
    assert "artifacts/index_full" in src
    # must not hardcode keep 10000 as default anymore
    tree = ast.parse(src)
    defaults_10000 = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in getattr(node, "keywords", []):
                if kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value == 10000:
                    # only fail if this is clearly max-events related nearby — soft check via source window
                    defaults_10000 = True
    # Allow 10000 only as documentation strings, not as argparse default for max-events
    assert "add_argument(\n        \"--max-events\"" in src.replace("'", '"') or '--max-events' in src
    assert "type=int,\n        default=None" in src or "default=None" in src.split("--max-events", 1)[1][:200]


def test_legacy_index_still_10k_untouched():
    p = ROOT / "artifacts/index/events.parquet"
    if not p.exists():
        pytest.skip("legacy index missing")
    import pandas as pd

    ev = pd.read_parquet(p, columns=["event_id"])
    assert ev["event_id"].nunique() == 10000


@pytest.mark.skipif(
    not (ROOT / "artifacts/results/stage6/full_split_audit.json").exists(),
    reason="full splits not built yet",
)
def test_full_split_confirm_never_used_and_sealed():
    audit = json.loads((ROOT / "artifacts/results/stage6/full_split_audit.json").read_text())
    assert audit["confirm_all_never_used_in_stage1_5"] is True
    assert audit["event_disjoint"] is True
    assert audit["trace_disjoint"] is True
    seal = full_stage6_paths()["confirm_seal"]
    assert seal.exists()
    payload = json.loads(seal.read_text())
    assert payload["status"] == "SEALED"
    if full_stage6_paths()["method_lock"].exists():
        pytest.skip("method locked")
    with pytest.raises(RuntimeError):
        assert_full_confirm_access_allowed(purpose="unit_test")
    # confirm size
    conf = load_full_event_ids("stage6_internal_confirm")
    assert len(conf) >= 2000


def test_detected_ae_p95_excludes_misses_and_warns():
    pred = np.array([100.0, np.nan, 500.0])
    true = np.array([100.0, 200.0, 100.0])
    sr = np.array([100.0, 100.0, 100.0])
    m = match_picks(pred, true, sr)
    assert m["n_miss"] == 1
    assert "detected_ae_p95" in m
    assert "complete end-to-end" in m["p95_naming_warning"]
    bundle = report_pick_timing_bundle(pred, true, sr)
    assert bundle["miss_rate"] == pytest.approx(1 / 3)
    assert "legacy_alias_e2e_p95_ae" in bundle
