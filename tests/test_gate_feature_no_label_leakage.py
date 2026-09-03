"""Ensure gate features never include label-derived leakage columns."""

from __future__ import annotations

import pandas as pd
import pytest

from earthquake.gating.features import (
    FEATURE_COLUMNS_CATALOG,
    assert_no_forbidden_columns,
    build_trace_features,
    fit_scaler,
)


def test_feature_schema_has_no_forbidden_names():
    assert_no_forbidden_columns(FEATURE_COLUMNS_CATALOG)
    with pytest.raises(ValueError):
        assert_no_forbidden_columns(["trace_Z_snr_db"])
    with pytest.raises(ValueError):
        assert_no_forbidden_columns(["true_s_sample"])


def test_build_trace_features_ignores_labels_on_row():
    row = pd.Series(
        {
            "pred_p_sample": 1000.0,
            "history_count": 12,
            "history_available": True,
            "residual_s_mad": 0.15,
            "residual_s_median": 0.1,
            "fallback_level": 0,
            "distance_km": 40.0,
            "source_depth_km": 10.0,
            "station_elevation_m": 100.0,
            "true_s_sample": 2000.0,
            "s_arrival_sample": 2000.0,
            "snr_db": 12.0,
        }
    )
    cands = [
        {
            "sample_index": 2100,
            "peak_probability": 0.8,
            "prominence": 0.4,
            "peak_width": 10,
            "local_entropy": 1.0,
            "fallback_peak": False,
        },
        {
            "sample_index": 2500,
            "peak_probability": 0.3,
            "prominence": 0.1,
            "peak_width": 8,
            "local_entropy": 1.2,
            "fallback_peak": False,
        },
    ]
    feats = build_trace_features(
        row,
        cands,
        expected_s_sample=2050.0,
        history_sigma_samples=20.0,
        sampling_rate=100.0,
        mode="catalog",
    )
    assert "true_s_sample" not in feats
    assert "snr_db" not in feats
    assert set(feats) == set(FEATURE_COLUMNS_CATALOG)
    scaler = fit_scaler(pd.DataFrame([feats]), FEATURE_COLUMNS_CATALOG)
    X, M = scaler.transform(pd.DataFrame([feats]))
    assert X.shape == (1, len(FEATURE_COLUMNS_CATALOG))
    assert M.shape == X.shape
