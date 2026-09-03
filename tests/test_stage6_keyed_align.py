"""Unit tests for Stage-6 keyed alignment and C.2 invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earthquake.stage6.keyed_align import (
    AlignmentError,
    keyed_align_predictions,
    metrics_from_aligned,
    shuffle_invariant_check,
)
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

ROOT = Path(__file__).resolve().parents[1]


def _toy(n: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    traces = [f"t{i}" for i in range(n)]
    events = [f"e{i//5}" for i in range(n)]
    true = rng.uniform(100, 5000, size=n)
    sr = np.full(n, 100.0)
    pred = true + rng.normal(0, 20, size=n)
    man = pd.DataFrame(
        {
            "trace_name": traces,
            "event_id": events,
            "s_arrival_sample": true,
            "sampling_rate_hz": sr,
        }
    )
    preds = pd.DataFrame(
        {
            "trace_name": traces,
            "event_id": events,
            "pred_s_sample": pred,
            "true_s_sample": true,
            "sampling_rate_hz": sr,
            "none_of_k": False,
        }
    )
    return man, preds


def test_shuffle_invariance_manifest_and_preds():
    man, preds = _toy()
    out = shuffle_invariant_check(man, preds, seed=7)
    assert out["ok"] is True


def test_groupby_order_change_invariant():
    man, preds = _toy()
    preds2 = preds.iloc[::-1].reset_index(drop=True)
    m0 = metrics_from_aligned(keyed_align_predictions(man, preds))
    m1 = metrics_from_aligned(keyed_align_predictions(man, preds2))
    assert m0["f1@0.5"] == pytest.approx(m1["f1@0.5"])


def test_duplicate_trace_fails():
    man, preds = _toy()
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(AlignmentError, match="duplicate"):
        keyed_align_predictions(man, preds)


def test_missing_trace_fails():
    man, preds = _toy()
    with pytest.raises(AlignmentError, match="missing"):
        keyed_align_predictions(man, preds.iloc[:-1])


def test_extra_trace_fails():
    man, preds = _toy()
    extra = preds.iloc[[0]].copy()
    extra["trace_name"] = "extra_trace"
    with pytest.raises(AlignmentError, match="extra"):
        keyed_align_predictions(man, pd.concat([preds, extra], ignore_index=True))


def test_sampling_rate_mismatch_fails():
    man, preds = _toy()
    preds = preds.copy()
    preds.loc[0, "sampling_rate_hz"] = 50.0
    with pytest.raises(AlignmentError, match="sampling_rate"):
        keyed_align_predictions(man, preds)


def test_event_id_mismatch_fails():
    man, preds = _toy()
    preds = preds.copy()
    preds.loc[0, "event_id"] = "WRONG"
    with pytest.raises(AlignmentError, match="event_id"):
        keyed_align_predictions(man, preds)


def test_sample_second_and_p95_denom():
    pred = np.array([100.0, np.nan, 500.0])
    true = np.array([100.0, 200.0, 100.0])
    sr = np.array([100.0, 100.0, 100.0])
    m = comprehensive_pick_metrics(pred, true, sr, none_mask=np.array([False, True, False]))
    assert m["detected_ae_p95_denominator"] == 2
    assert m["miss_count"] == 1
    assert m["detected_ae_mae"] == pytest.approx(2.0)


@pytest.mark.skipif(
    not (ROOT / "artifacts/results/stage6/phaseC/ranker_dev_predictions.parquet").exists(),
    reason="frozen preds missing",
)
def test_frozen_preds_keyed_matches_phaseC1():
    man = pd.read_csv(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv")
    preds = pd.read_parquet(ROOT / "artifacts/results/stage6/phaseC/ranker_dev_predictions.parquet")
    aligned = keyed_align_predictions(man, preds)
    m = metrics_from_aligned(aligned)
    assert m["f1@0.5"] == pytest.approx(0.8322776841575145, rel=0, abs=1e-9)
    bad = comprehensive_pick_metrics(
        preds["pred_s_sample"].to_numpy(float),
        man["s_arrival_sample"].to_numpy(float),
        man["sampling_rate_hz"].to_numpy(float),
        none_mask=preds["none_of_k"].to_numpy(bool),
    )
    assert abs(bad["f1@0.5"] - m["f1@0.5"]) > 0.01
