"""Tests for pick matching / metric audit definitions."""

from __future__ import annotations

import numpy as np

from earthquake.metrics import audit_pick_errors, match_picks


def test_wrong_peak_is_fp_and_fn_once():
    true = np.array([100.0])
    pred = np.array([500.0])  # 4s error @100Hz
    m = match_picks(pred, true, sampling_rate=100.0, windows_s=(0.5,))
    assert m["tp@0.5s"] == 0
    assert m["fp@0.5s"] == 1
    assert m["fn@0.5s"] == 1
    assert m["f1@0.5s"] == 0.0


def test_miss_and_false_not_double_counted():
    true = np.array([100.0, np.nan])
    pred = np.array([np.nan, 200.0])
    m = match_picks(pred, true, 100.0, windows_s=(0.5,))
    assert m["tp@0.5s"] == 0
    assert m["fn@0.5s"] == 1
    assert m["fp@0.5s"] == 1


def test_audit_separates_e2e_and_matched_timing():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([100.0, 250.0, np.nan])  # perfect, wrong(+0.5s), miss
    a = audit_pick_errors(pred, true, 100.0, window_s=0.1)
    assert a["n_matched_within_tol"] == 1
    assert a["n_wrong_peak_beyond_tol"] == 1
    assert a["n_missed_pick"] == 1
    assert a["matched_timing_mae"] == 0.0
    assert a["e2e_mae"] > 0.0
    assert a["p95_e2e_scope"] == "all_labeled_with_prediction"


def test_multi_peak_protocol_can_recover_secondary():
    true = np.array([100.0])
    pred = np.array([500.0])  # top peak wrong
    multi = [np.array([500.0, 100.0])]
    a = audit_pick_errors(pred, true, 100.0, window_s=0.5, multi_pred_samples=multi)
    assert a["multi_peak_protocol"]["n_any_cand_within_tol"] == 1
    assert a["multi_peak_protocol"]["recall_any_cand"] == 1.0
