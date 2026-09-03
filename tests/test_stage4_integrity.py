"""Stage-4 integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stage3_artifacts_not_deleted():
    assert (ROOT / "artifacts/results/stage3/learned_gate_eval_test.json").exists()
    assert (ROOT / "artifacts/results/stage2/best_lambdas.json").exists()


def test_method_lock_exists_and_stable_keys():
    p = ROOT / "artifacts/results/stage4/method_lock.json"
    if not p.exists():
        pytest.skip("run lock_stage4_method.py first")
    lock = json.loads(p.read_text())
    assert lock["method_name"] == "fixed_catalog_rescore"
    assert lock["fixed_rescore_lambdas"]["selected_using_stage3_test"] is False
    assert lock["residual_history"]["shrinkage_k"] == 50.0
    assert lock["peak_detection"]["candidate_k"] == 5
    assert (ROOT / "artifacts/results/stage4/method_lock_meta.json").exists()


def test_holdout_event_disjoint_from_stage3_test():
    hold = ROOT / "artifacts/results/stage4/holdout/holdout_events.parquet"
    audit = ROOT / "artifacts/results/stage4/holdout_split_audit.json"
    if not hold.exists():
        pytest.skip("holdout not built")
    a = json.loads(audit.read_text())
    assert a["overlap_used_events"] == 0
    assert a["overlap_train_events"] == 0
    assert a["overlap_val_events"] == 0
    df = pd.read_parquet(hold)
    assert (df.split == "test").all()
    s3 = pd.read_parquet(ROOT / "artifacts/diagnostics/fixed_eval_test_only_events.parquet")
    assert set(df.event_id.astype(str)).isdisjoint(set(s3.event_id.astype(str)))


def test_holdout_selection_uses_metadata_only_columns():
    src = (ROOT / "scripts/build_stage4_holdout.py").read_text()
    assert 'select_cols = ["event_id", "trace_name", "origin_time", "split", "station_id", "n_samples"]' in src


def test_bootstrap_is_event_level_when_present():
    p = ROOT / "artifacts/results/stage4/bootstrap_confirmatory.json"
    if not p.exists():
        pytest.skip("bootstrap not run")
    rep = json.loads(p.read_text())
    assert rep["sampling_unit"] == "event"
    assert rep["n_bootstrap"] >= 1000


def test_fixed_rescore_deterministic_unit():
    from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
    from earthquake.fusion.peak_candidates import PeakCandidate

    cands = [
        PeakCandidate(1000, None, 0.9, 0.5, 10, 1.0, 0, False, "s"),
        PeakCandidate(1500, None, 0.4, 0.2, 8, 1.0, 1, False, "s"),
    ]
    a, _ = rescore_phase_candidates(
        cands, expected_sample=1490, sigma_samples=40, lambda_wave=0.5, lambda_history=2.0, lambda_prominence=0.0, history_available=True
    )
    b, _ = rescore_phase_candidates(
        cands, expected_sample=1490, sigma_samples=40, lambda_wave=0.5, lambda_history=2.0, lambda_prominence=0.0, history_available=True
    )
    assert a.sample_index == b.sample_index == 1500


def test_metrics_tolerances_include_0_2():
    from earthquake.metrics import match_picks

    m = match_picks(np.array([100.0]), np.array([115.0]), np.array([100.0]), windows_s=(0.1, 0.2, 0.5))
    assert m["f1@0.1s"] == 0.0
    assert m["f1@0.2s"] == 1.0
    assert m["f1@0.5s"] == 1.0


def test_history_unavailable_identity_near_phasenet_ranking():
    from earthquake.fusion.candidate_rescorer import pick_argmax_probability, rescore_phase_candidates
    from earthquake.fusion.peak_candidates import PeakCandidate

    cands = [
        PeakCandidate(1000, None, 0.8, 0.4, 10, 1.0, 0, False, "s"),
        PeakCandidate(2000, None, 0.5, 0.2, 8, 1.0, 1, False, "s"),
    ]
    pn = pick_argmax_probability(cands)
    best, _ = rescore_phase_candidates(
        cands, expected_sample=float("nan"), sigma_samples=1.0, lambda_wave=0.5, lambda_history=2.0, lambda_prominence=0.0, history_available=False
    )
    assert best.sample_index == pn.sample_index
