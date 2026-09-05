"""Tests for scalar pairwise full-dev protocol (no confirm IO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from earthquake.pairwise.fulldev import (
    TAU,
    apply_switch,
    build_phaseb_pairs,
    resid_control_pred,
    scalar_vec_from_values,
)
from earthquake.pairwise.guards import (
    assert_no_confirm_or_phaseb_train_path,
    assert_no_confirm_path,
)


def test_confirm_guard_blocks_confirm():
    with pytest.raises(RuntimeError):
        assert_no_confirm_path("/data/results/stage6_internal_confirm_events.txt")
    with pytest.raises(RuntimeError):
        assert_no_confirm_path("artifacts/results/confirmation/metrics.json")
    with pytest.raises(RuntimeError):
        assert_no_confirm_path("/tmp/final_confirm/preds.npy")


def test_confirm_guard_allows_phaseb_for_fulldev():
    # phaseB paths are OK for full-dev evaluator (confirm-only guard)
    assert_no_confirm_path(
        str(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv")
    )


def test_train_guard_still_blocks_phaseb():
    with pytest.raises(RuntimeError):
        assert_no_confirm_or_phaseb_train_path(
            str(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv")
        )


def test_tau_frozen():
    assert TAU == 0.50


def test_c1_c2_rank_independent_of_true_s():
    # synthetic: true S near cand1 but fixed_score prefers cand0
    enr = pd.DataFrame(
        {
            "trace_name": ["t0", "t0"],
            "event_id": ["e0", "e0"],
            "candidate_index": [0, 1],
            "candidate_sample": [100.0, 200.0],
            "fixed_score": [2.0, 5.0],  # c1 should be index 1
            "cand_prob": [0.9, 0.5],
            "source": ["stead", "ida"],
            "resid_s": [0.1, -0.2],
            "resid_sp": [0.0, 0.0],
            "delta_sp": [1.0, 2.0],
            "stead_probability": [0.9, 0.0],
            "ida_probability": [0.0, 0.5],
            "sampling_rate_hz": [100.0, 100.0],
            "s_arrival_sample": [100.0, 100.0],
        }
    )
    man = pd.DataFrame(
        {
            "trace_name": ["t0"],
            "s_arrival_sample": [100.0],
            "sampling_rate_hz": [100.0],
        }
    )
    pairs = build_phaseb_pairs(enr, man)
    assert float(pairs.iloc[0].c1_sample) == 200.0
    assert float(pairs.iloc[0].c2_sample) == 100.0
    assert int(pairs.iloc[0].c1_index) == 1


def test_lt2_candidates_stay_c1():
    enr = pd.DataFrame(
        {
            "trace_name": ["t1"],
            "event_id": ["e1"],
            "candidate_index": [0],
            "candidate_sample": [50.0],
            "fixed_score": [1.0],
            "cand_prob": [0.8],
            "source": ["stead"],
            "resid_s": [0.0],
            "resid_sp": [0.0],
            "delta_sp": [1.0],
            "stead_probability": [0.8],
            "ida_probability": [np.nan],
            "sampling_rate_hz": [100.0],
            "s_arrival_sample": [50.0],
        }
    )
    man = pd.DataFrame({"trace_name": ["t1"], "s_arrival_sample": [50.0], "sampling_rate_hz": [100.0]})
    pairs = build_phaseb_pairs(enr, man)
    p = np.array([0.99])
    pred, switch = apply_switch(pairs, p, tau=0.5)
    assert switch[0] == False
    assert pred[0] == 50.0


def test_resid_control_frozen_kernel():
    pairs = pd.DataFrame(
        {
            "n_candidates": [2, 2],
            "c1_sample": [10.0, 10.0],
            "c2_sample": [20.0, 20.0],
            "c1_fixed_score": [1.0, 1.0],
            "c2_fixed_score": [1.0, 0.0],
            "c1_resid_s": [0.0, 0.0],
            "c2_resid_s": [0.0, 5.0],
        }
    )
    pred = resid_control_pred(pairs, lam=1.0, sigma=0.5)
    # first: equal resid, equal score -> stay c1 (s2 not > s1)
    assert pred[0] == 10.0
    # second: c2 resid huge -> stay c1
    assert pred[1] == 10.0


def test_scalar_feature_order():
    v = scalar_vec_from_values(
        prob=0.1,
        stead_prob=0.2,
        ida_prob=0.3,
        fixed_score=0.4,
        resid_s=0.5,
        resid_sp=0.6,
        delta_sp=0.7,
        source="ida",
    )
    assert v.shape == (10,)
    assert v[8] == 1.0 and v[7] == 0.0 and v[9] == 0.0


def test_method_lock_if_present():
    p = ROOT / "artifacts/results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    if not p.exists():
        pytest.skip("lock not written yet")
    lock = json.loads(p.read_text())
    assert lock["locked_before_phaseB_full_dev_evaluation"] is True
    assert lock["calibration_tau"] == 0.5
    assert lock["architecture"]["use_waveform"] is False
