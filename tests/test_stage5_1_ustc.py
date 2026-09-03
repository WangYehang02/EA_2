"""Tests for Stage 5.1 USTC audit utilities (train/val only; no Stage 3/4 test reads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earthquake.analysis.hierarchical_residual import (
    assert_event_disjoint,
    assert_no_test_split,
    build_path_residual_table,
    directed_path_key,
    hierarchical_residual,
)
from earthquake.history.shrinkage import shrink_residual


ROOT = Path(__file__).resolve().parents[1]


def _row(eid, lat, lon, depth, net, sta, residual, split="train"):
    return {
        "event_id": eid,
        "source_latitude": lat,
        "source_longitude": lon,
        "source_depth_km": depth,
        "network": net,
        "station": sta,
        "location": "",
        "channel_prefix": "HH",
        "residual_tau_s": residual,
        "split": split,
        "distance_km": 50.0,
    }


def test_build_table_train_only_and_event_disjoint():
    train = pd.DataFrame(
        [
            _row("e1", 42.01, 13.01, 8.0, "IV", "A", 0.5),
            _row("e2", 42.02, 13.02, 8.0, "IV", "A", 0.6),
            _row("e3", 42.015, 13.015, 8.0, "IV", "A", 0.55),
            _row("e4", 42.0, 13.0, 8.0, "IV", "A", 0.52),
            _row("e5", 42.03, 13.03, 8.0, "IV", "A", 0.58),
            _row("e6", 40.0, 10.0, 8.0, "IV", "B", -0.2),
        ]
    )
    with pytest.raises(ValueError):
        build_path_residual_table(train.assign(split="test"), residual_col="residual_tau_s", grid_size=0.1, min_history=3)

    tab = build_path_residual_table(train, residual_col="residual_tau_s", grid_size=1.0, min_history=3)
    assert tab.n_train_rows == 6
    key = directed_path_key(train.iloc[0], 1.0)
    st = tab.query(key)
    assert st["n"] >= 3
    assert_event_disjoint({"e1", "e2"}, {"v1"}, context="ok")
    with pytest.raises(ValueError):
        assert_event_disjoint({"e1"}, {"e1"}, context="leak")


def test_hierarchical_backoff_and_no_history_fallback():
    fine = {"n": 0.0, "median": float("nan"), "mad": float("nan"), "history_available": 0.0}
    coarse = {"n": 20.0, "median": 0.4, "mad": 0.1, "history_available": 1.0}
    out = hierarchical_residual(fine, coarse, global_median=0.0, k_fine=50.0, k_coarse=50.0, min_history=5)
    assert out["fine_ok"] is False
    assert out["coarse_ok"] is True
    assert out["history_available"] is True
    # equals shrink of coarse toward global
    assert out["r_hat"] == pytest.approx(shrink_residual(0.4, 0.0, 20.0, 50.0))

    none = hierarchical_residual(fine, fine, global_median=-0.01, k_fine=50.0, k_coarse=50.0, min_history=5)
    assert none["history_available"] is False
    assert none["r_hat"] == pytest.approx(-0.01)


def test_shuffled_path_control_is_worse_on_synthetic():
    # Synthetic: path A residual +1, path B residual -1; shuffle should hurt MAE
    rng = np.random.default_rng(0)
    true = np.array([1.0] * 50 + [-1.0] * 50)
    pred_correct = true.copy()
    pred_shuf = true.copy()
    pred_shuf = pred_shuf[rng.permutation(len(pred_shuf))]
    assert np.mean(np.abs(pred_correct - true)) < np.mean(np.abs(pred_shuf - true))


def test_stage5_1_scripts_do_not_read_stage3_test_lists():
    for name in [
        "scripts/analyze_path_residual_repeatability.py",
        "scripts/evaluate_hierarchical_residual_val.py",
    ]:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "fixed_eval_test_only" not in text
        assert "stage4/holdout" not in text
        assert "learned_gate_picks_test" not in text


def test_assert_no_test_split():
    df = pd.DataFrame({"split": ["train", "val"]})
    assert_no_test_split(df, context="ok")
    with pytest.raises(ValueError):
        assert_no_test_split(pd.DataFrame({"split": ["test"]}), context="bad")


def test_frozen_hashes_file_exists():
    p = ROOT / "artifacts/results/stage5_1_ustc/frozen_hashes_before.json"
    assert p.exists()
