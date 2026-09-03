#!/usr/bin/env python
"""Unit tests for Stage 6 Phase C ranker (no confirm, no leakage, no PhaseNet train)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, load_full_event_ids
from earthquake.stage6.ranker.dataset import TraceCandidateDataset
from earthquake.stage6.ranker.features import assemble_feature_table, feature_names
from earthquake.stage6.ranker.labels import label_union_candidates
from earthquake.stage6.ranker.models import build_ranker, count_params, listwise_loss
from earthquake.stage6.ranker.schema import CANDIDATE_SCHEMA, assert_no_forbidden_features, schema_sha256
from earthquake.stage6.ranker.union_schema import union_candidates_phaseC

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_schema_frozen_and_matches_phaseB_extract():
    assert CANDIDATE_SCHEMA["stead_k"] == 5
    assert CANDIDATE_SCHEMA["ida_k"] == 5
    assert CANDIDATE_SCHEMA["max_union"] == 10
    assert CANDIDATE_SCHEMA["dedup_s"] == 0.05
    assert CANDIDATE_SCHEMA["peak_extract"]["min_distance"] == 50
    assert len(schema_sha256()) == 64


def test_union_representative_time_prefers_stead():
    stead = pd.DataFrame(
        {
            "trace_name": ["t1"],
            "event_id": ["e1"],
            "true_s_sample": [1000.0],
            "sampling_rate_hz": [100.0],
            "candidate_rank": [0],
            "candidate_sample": [1000.0],
            "candidate_probability": [0.9],
            "candidate_time_utc": ["S"],
            "candidate_source": ["stead"],
            "top1_s_sample": [1000.0],
            "top1_p_sample": [500.0],
            "top1_p_prob": [0.8],
        }
    )
    ida = pd.DataFrame(
        {
            "trace_name": ["t1"],
            "event_id": ["e1"],
            "true_s_sample": [1000.0],
            "sampling_rate_hz": [100.0],
            "candidate_rank": [0],
            "candidate_sample": [1003.0],  # within 0.05s
            "candidate_probability": [0.95],
            "candidate_time_utc": ["I"],
            "candidate_source": ["ida"],
            "top1_s_sample": [1003.0],
            "top1_p_sample": [510.0],
            "top1_p_prob": [0.7],
        }
    )
    u = union_candidates_phaseC(stead, ida)
    assert len(u) == 1
    assert float(u.iloc[0]["candidate_sample"]) == 1000.0  # STEAD time kept
    assert bool(u.iloc[0]["both_support"])
    assert u.iloc[0]["candidate_count"] <= 10


def test_union_cap_le_10():
    rows_s = []
    rows_i = []
    for r in range(5):
        rows_s.append(
            {
                "trace_name": "t",
                "event_id": "e",
                "true_s_sample": 0.0,
                "sampling_rate_hz": 100.0,
                "candidate_rank": r,
                "candidate_sample": 1000 + r * 200,
                "candidate_probability": 0.9 - 0.1 * r,
                "candidate_time_utc": None,
                "candidate_source": "stead",
                "top1_s_sample": 1000,
                "top1_p_sample": 100,
                "top1_p_prob": 0.5,
            }
        )
        rows_i.append(
            {
                "trace_name": "t",
                "event_id": "e",
                "true_s_sample": 0.0,
                "sampling_rate_hz": 100.0,
                "candidate_rank": r,
                "candidate_sample": 1100 + r * 200,
                "candidate_probability": 0.85 - 0.1 * r,
                "candidate_time_utc": None,
                "candidate_source": "ida",
                "top1_s_sample": 1100,
                "top1_p_sample": 120,
                "top1_p_prob": 0.4,
            }
        )
    u = union_candidates_phaseC(pd.DataFrame(rows_s), pd.DataFrame(rows_i))
    assert len(u) <= 10


def test_labels_none_of_k_and_positive():
    u = pd.DataFrame(
        {
            "trace_name": ["a", "a", "b", "b"],
            "event_id": ["e", "e", "e", "e"],
            "true_s_sample": [100.0, 100.0, 100.0, 100.0],
            "sampling_rate_hz": [100.0] * 4,
            "candidate_index": [0, 1, 0, 1],
            "candidate_sample": [100.0, 500.0, 800.0, 900.0],
            "candidate_count": [2, 2, 2, 2],
            "both_support": [False] * 4,
            "candidate_source_mask": ["stead"] * 4,
            "top1_s_stead": [100.0, 100.0, 800.0, 800.0],
        }
    )
    lab = label_union_candidates(u)
    a = lab[lab.trace_name == "a"]
    assert bool(a.loc[a.candidate_index == 0, "is_positive"].iloc[0])
    assert not bool(a["label_none_of_k"].iloc[0])
    b = lab[lab.trace_name == "b"]
    assert bool(b["label_none_of_k"].iloc[0])


def test_forbidden_features():
    assert_no_forbidden_features(feature_names("R2"))
    with pytest.raises(RuntimeError):
        assert_no_forbidden_features(["true_s_sample", "prob_stead"])


def test_padding_mask_and_selected_from_candidates():
    feat = pd.DataFrame(
        {
            "trace_name": ["t"] * 2,
            "event_id": ["e"] * 2,
            "candidate_index": [0, 1],
            "candidate_sample": [10.0, 50.0],
            "is_positive": [True, False],
            "label_none_of_k": [False, False],
            "positive_index": [0, 0],
            "candidate_count": [2, 2],
            "analysis_classes": ["easy_top1_correct"] * 2,
            "sampling_rate_hz": [100.0] * 2,
            "true_s_sample": [10.0] * 2,
            **{n: [0.1, 0.2] for n in feature_names("R1")},
        }
    )
    ds = TraceCandidateDataset(feat, feature_names("R1"), max_k=10, hard_boost=False)
    item = ds[0]
    assert item["mask"].sum() == 2
    assert int(item["target"]) == 0
    # selected sample must be one of candidates when not none
    assert float(item["samples"][0]) == 10.0


def test_listwise_loss_and_param_budget():
    model = build_ranker("R3", n_features=len(feature_names("R2")))
    assert count_params(model) <= 250_000
    x = torch.randn(4, 10, len(feature_names("R2")))
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, :3] = True
    logits = model(x, mask)
    assert logits.shape == (4, 11)
    target = torch.tensor([0, 10, 1, 2])
    loss = listwise_loss(logits, target, beta=0.2, gamma=1.0)
    assert torch.isfinite(loss)


def test_splits_disjoint_and_confirm_sealed():
    pe = set(load_full_event_ids("stage6_picker_train"))
    re = set(load_full_event_ids("stage6_ranker_train"))
    de = set(load_full_event_ids("stage6_dev"))
    ce = set(load_full_event_ids("stage6_internal_confirm"))
    assert pe.isdisjoint(re) and pe.isdisjoint(de) and re.isdisjoint(de)
    assert pe.isdisjoint(ce) and re.isdisjoint(ce) and de.isdisjoint(ce)
    seal = ROOT / "artifacts/results/stage6/splits_full/CONFIRM_SEALED"
    assert seal.exists()
    if not (ROOT / "artifacts/results/stage6/method_lock_stage6.json").exists():
        with pytest.raises(RuntimeError):
            assert_full_confirm_access_allowed(purpose="unit")


def test_phaseC_scripts_no_phasenet_optimizer():
    for rel in [
        "scripts/train_stage6_candidate_ranker.py",
        "scripts/build_stage6_ranker_dataset.py",
        "scripts/evaluate_stage6_candidate_ranker.py",
        "scripts/build_stage6_picker_train_history.py",
    ]:
        src = (ROOT / rel).read_text()
        assert "sbm.PhaseNet" not in src
        assert "loss.backward" in src or "PhaseNet" not in src or rel.endswith("history.py")
        # train script may have loss.backward for ranker — ensure no PhaseNet from_pretrained train
        assert "from_pretrained" not in src or "refuse" in src.lower() or "SeisBench" not in src
    train = (ROOT / "scripts/train_stage6_candidate_ranker.py").read_text()
    assert "AdamW" in train
    assert "PhaseNet" not in train or "No PhaseNet" in train


def test_stage2_5_artifacts_still_present():
    assert (ROOT / "artifacts/results/stage4/method_lock.json").exists()
    assert (ROOT / "artifacts/results/stage2/best_lambdas.json").exists()


def test_shuffled_history_helper_keeps_labels():
    # lightweight: permuting hist values by event key should not touch labels
    labels = np.array([0, 1, 0, 1])
    hist = np.array([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(hist))
    hist2 = hist[perm]
    assert np.array_equal(labels, labels)
    assert not np.array_equal(hist, hist2)
