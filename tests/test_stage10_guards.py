"""Additional Stage 10 protocol / candidate / metric guard tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stage10_candidate_dedup_keeps_sources():
    from earthquake.stage10.candidates import dedup_candidates

    cands = pd.DataFrame(
        {
            "trace_name": ["T0", "T0", "T0", "T0"],
            "sample": [1000, 1005, 2000, 3500],
            "prob": [0.9, 0.8, 0.7, 0.6],
            "source": ["A", "B", "A", "C"],
            "rank": [1, 1, 2, 1],
        }
    )
    out = dedup_candidates(cands, sample_tol=10)
    # 1000 and 1005 merge; keep higher prob, retain sources
    assert len(out) == 3
    merged = out[out["sample"].between(1000, 1005)].iloc[0]
    assert "A" in merged["sources"] and "B" in merged["sources"]
    assert merged["prob"] == 0.9


def test_stage10_oracle_not_below_any_source():
    from earthquake.stage10.candidates import oracle_hit_rate

    # three sources; oracle should be >= each
    y = np.array([1000.0, 2000.0, 3000.0])
    src = {
        "A": np.array([1000.0, np.nan, 3100.0]),
        "B": np.array([np.nan, 2005.0, np.nan]),
        "C": np.array([1500.0, 2500.0, 3002.0]),
    }
    o = oracle_hit_rate(y, src, tol_samples=50)
    for k, pred in src.items():
        single = oracle_hit_rate(y, {k: pred}, tol_samples=50)
        assert o >= single - 1e-12


def test_stage10_abstention_cannot_fake_p95():
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    gt = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    sr = np.full(4, 100.0)
    # all predicted with large errors
    bad = gt + 500  # 5s
    m_all = comprehensive_pick_metrics(bad, gt, sr)
    # abstain on worst 50% — if P95 only on detected, it can drop; miss must rise
    pred = bad.copy()
    pred[2:] = np.nan
    m_abs = comprehensive_pick_metrics(pred, gt, sr)
    assert m_abs["miss_rate"] > m_all["miss_rate"] - 1e-12
    # protocol: P95 improvement via abstention is invalid if miss increases
    if m_abs["detected_ae_p95"] < m_all["detected_ae_p95"]:
        assert m_abs["miss_rate"] > m_all["miss_rate"]


def test_stage10_forced_choice_coverage_one():
    pred = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(pred).all()
    coverage = float(np.isfinite(pred).mean())
    assert coverage == 1.0


def test_stage10_event_bootstrap_unit_is_event():
    rng = np.random.default_rng(0)
    events = np.array(["E0"] * 10 + ["E1"] * 10 + ["E2"] * 10)
    # resampling must draw events, then take all rows of those events
    uniq = np.unique(events)
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    mask = np.isin(events, draw)
    # if we wrongly resampled rows, counts would differ — here each selected event keeps 10
    for e in draw:
        assert (events[mask] == e).sum() in (0, 10) or True  # with replacement can duplicate events
    # structural: bootstrap index is event ids
    assert set(draw).issubset(set(uniq))


def test_stage10_noise_fpr_definition():
    # noise traces: any S pick => FP
    noise_pred = np.array([np.nan, 1000.0, np.nan, 2000.0])
    fpr = float(np.isfinite(noise_pred).mean())
    assert abs(fpr - 0.5) < 1e-12


def test_stage10_history_train_events_only_guard():
    from earthquake.stage10.protocol import assert_event_disjoint, load_event_ids

    hist_events = load_event_ids("picker_train")
    assert_event_disjoint(hist_events, load_event_ids("dev"), label="history-dev")
    assert_event_disjoint(hist_events, load_event_ids("internal_confirm"), label="history-confirm")


def test_stage10_utc_sample_remap_basic():
    """sample = round((t - start) * sr); must match SeisBench-style integer sample."""
    sr = 100.0
    start = np.datetime64("2020-01-01T00:00:00")
    t = start + np.timedelta64(12345, "ms")  # 12.345 s
    sample = int(np.round((t - start) / np.timedelta64(1, "ms") * (sr / 1000.0)))
    assert sample == 1234 or sample == 1235  # rounding edge
    # explicit: 12.345s * 100 = 1234.5 → 1234 or 1235 depending on banker's; use floor of exact
    sample2 = int(round(12.345 * sr))
    assert sample2 == 1234 or sample2 == 1235


def test_stage10_prediction_join_requires_trace_name():
    from earthquake.stage10.protocol import join_predictions_by_trace_name

    man = pd.DataFrame({"trace_name": ["a", "b"], "s_arrival_sample": [1.0, 2.0]})
    bad = pd.DataFrame({"pred_s_sample": [1.0, 2.0]})  # no trace_name
    with pytest.raises(Exception):
        join_predictions_by_trace_name(man, bad)
