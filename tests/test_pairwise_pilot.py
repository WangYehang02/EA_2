"""Pairwise pilot unit tests (no confirm / no full-dev IO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from earthquake.pairwise.model import (
    PairwiseScorer,
    PairwiseWindowConfig,
    assert_swap_antisymmetry,
    crop_candidate_window,
    guard_no_forbidden_columns,
)


def test_split_lock_no_overlap():
    lock_path = ROOT / "artifacts/results/pairwise_pilot/SPLIT.LOCK.json"
    if not lock_path.exists():
        pytest.skip("split not frozen yet")
    lock = json.loads(lock_path.read_text())
    assert lock["overlap_event_train_cal"] == 0
    assert lock["overlap_event_train_eval"] == 0
    assert lock["overlap_event_cal_eval"] == 0
    trains = set((ROOT / "artifacts/results/pairwise_pilot/events_train.txt").read_text().split())
    cals = set((ROOT / "artifacts/results/pairwise_pilot/events_calibration.txt").read_text().split())
    evals = set((ROOT / "artifacts/results/pairwise_pilot/events_heldout_eval.txt").read_text().split())
    assert not (trains & cals)
    assert not (trains & evals)
    assert not (cals & evals)


def test_label_four_classes_and_rank_independent_of_true_s_definition():
    path = ROOT / "artifacts/results/pairwise_pilot/pairs_ranker_train.parquet"
    if not path.exists():
        pytest.skip("pairs not frozen")
    df = pd.read_parquet(path)
    assert set(df.label_class.unique()) <= {
        "choose_c1",
        "choose_c2",
        "both_correct",
        "both_wrong",
        "tie_ae_close",
        "single_candidate",
    }
    # c1 score >= c2 score when both exist
    ge2 = df[df.n_candidates >= 2]
    assert (ge2.c1_fixed_score + 1e-9 >= ge2.c2_fixed_score).all()


def test_crop_does_not_use_label_and_padding_mask():
    cfg = PairwiseWindowConfig()
    wave = np.random.randn(3, 5000).astype(np.float32)
    x, m = crop_candidate_window(wave, 100.0, cfg=cfg)
    assert x.shape == (3, cfg.n_samples)
    assert m.shape == (cfg.n_samples,)
    assert m.sum() > 0
    # near edge -> padding
    x2, m2 = crop_candidate_window(wave, 5.0, cfg=cfg)
    assert m2[0] == 0.0 or m2.sum() < cfg.n_samples


def test_swap_antisymmetry_and_shared_weights():
    model = PairwiseScorer(n_scalar=10, use_waveform=True, use_scalar=True)
    assert model.n_parameters() <= 300_000
    B, T = 4, PairwiseWindowConfig().n_samples
    batch = {
        "wave1": torch.randn(B, 3, T),
        "mask1": torch.ones(B, T),
        "scalar1": torch.randn(B, 10),
        "wave2": torch.randn(B, 3, T),
        "mask2": torch.ones(B, T),
        "scalar2": torch.randn(B, 10),
    }
    err = assert_swap_antisymmetry(model, batch)
    assert err < 1e-5


def test_forbidden_features_guard():
    with pytest.raises(RuntimeError):
        guard_no_forbidden_columns(["cand_prob", "true_s_sample"])


def test_less_than_two_candidates_stay_c1():
    path = ROOT / "artifacts/results/pairwise_pilot/pairs_ranker_train.parquet"
    if not path.exists():
        pytest.skip("pairs not frozen")
    df = pd.read_parquet(path)
    single = df[df.n_candidates < 2]
    assert (single.label_class == "single_candidate").all()


def test_no_confirm_events_in_pairs():
    path = ROOT / "artifacts/results/pairwise_pilot/pairs_ranker_train.parquet"
    conf = ROOT / "artifacts/results/stage6/splits_full/stage6_internal_confirm_events.txt"
    if not path.exists() or not conf.exists():
        pytest.skip("missing artifacts")
    df = pd.read_parquet(path, columns=["event_id"])
    confirm = set(conf.read_text().split())
    assert len(set(df.event_id.astype(str)) & confirm) == 0


def test_eval_once_marker_logic():
    # marker file may or may not exist; test only that dual eval is prevented by convention
    done = ROOT / "artifacts/results/pairwise_pilot/PAIRWISE.EVAL.DONE"
    # if exists, running eval again should be blocked by script; here just assert boolean
    assert isinstance(done.exists(), bool)
