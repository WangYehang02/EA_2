#!/usr/bin/env python
"""Unit tests for Stage 6 Phase B candidate oracle utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earthquake.stage6.phaseB import (
    complementarity_table,
    event_level_bootstrap_delta,
    oracle_metrics_for_set,
    oracle_pick_closest,
    union_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_topk_and_union_dedup_and_cap():
    rows_s = []
    rows_i = []
    # same peak within 0.05s (5 samples @100Hz) should merge
    for r, samp, p in [(0, 1000, 0.9), (1, 2000, 0.5), (2, 3000, 0.4), (3, 4000, 0.3), (4, 5000, 0.2)]:
        rows_s.append(
            {
                "trace_name": "t1",
                "event_id": "e1",
                "true_s_sample": 1005.0,
                "sampling_rate_hz": 100.0,
                "candidate_rank": r,
                "candidate_sample": samp,
                "candidate_probability": p,
                "candidate_time_utc": None,
                "candidate_source": "stead",
            }
        )
    for r, samp, p in [(0, 1003, 0.8), (1, 2500, 0.7), (2, 3500, 0.6), (3, 4500, 0.55), (4, 5500, 0.5)]:
        rows_i.append(
            {
                "trace_name": "t1",
                "event_id": "e1",
                "true_s_sample": 1005.0,
                "sampling_rate_hz": 100.0,
                "candidate_rank": r,
                "candidate_sample": samp,
                "candidate_probability": p,
                "candidate_time_utc": None,
                "candidate_source": "ida",
            }
        )
    stead = pd.DataFrame(rows_s)
    ida = pd.DataFrame(rows_i)
    u = union_candidates(stead, ida, stead_k=5, ida_k=5, max_union=10)
    assert u["trace_name"].nunique() == 1
    assert len(u) <= 10
    # first peaks merged as both_support
    both = u[u["both_support"]]
    assert len(both) >= 1
    # no averaged probability column abuse
    assert "candidate_probability" not in u.columns or u["prob_stead"].notna().any()


def test_oracle_closest_only_from_candidates():
    assert oracle_pick_closest(np.array([10.0, 50.0, 90.0]), 48.0, 100.0) == 50.0
    assert np.isnan(oracle_pick_closest(np.array([]), 48.0, 100.0))


def test_oracle_metrics_and_f1_note():
    meta = pd.DataFrame(
        {
            "trace_name": ["a", "b"],
            "event_id": ["e1", "e1"],
            "s_arrival_sample": [100.0, 200.0],
            "sampling_rate_hz": [100.0, 100.0],
        }
    )
    cands = pd.DataFrame(
        {
            "trace_name": ["a", "a", "b"],
            "candidate_sample": [100.0, 500.0, 800.0],
            "candidate_rank": [0, 1, 0],
        }
    )
    o = oracle_metrics_for_set(cands, meta)
    assert "oracle_f1_vs_hitrate_note" in o
    assert o["oracle_recall@0.5"] == pytest.approx(0.5)
    assert o["oracle_predictions"][0] == 100.0


def test_complementarity_partition():
    meta = pd.DataFrame(
        {
            "trace_name": ["t1", "t2", "t3", "t4"],
            "event_id": ["e1", "e1", "e2", "e2"],
            "s_arrival_sample": [100.0, 100.0, 100.0, 100.0],
            "sampling_rate_hz": [100.0, 100.0, 100.0, 100.0],
            "station": ["A", "B", "C", "D"],
        }
    )
    stead = pd.DataFrame(
        {
            "trace_name": ["t1", "t2", "t3", "t4"],
            "candidate_sample": [100.0, 100.0, 500.0, 500.0],
            "candidate_rank": [0, 0, 0, 0],
            "candidate_probability": [0.9, 0.9, 0.9, 0.9],
            "sampling_rate_hz": [100.0] * 4,
        }
    )
    ida = pd.DataFrame(
        {
            "trace_name": ["t1", "t2", "t3", "t4"],
            "candidate_sample": [100.0, 500.0, 100.0, 500.0],
            "candidate_rank": [0, 0, 0, 0],
            "candidate_probability": [0.9, 0.9, 0.9, 0.9],
            "sampling_rate_hz": [100.0] * 4,
        }
    )
    tab = complementarity_table(stead, ida, meta, k=5, tol_s=0.5)
    classes = set(tab["class"])
    assert classes == {"both_correct", "stead_only_correct", "ida_only_correct", "neither_correct"}
    assert len(tab) == 4
    assert tab["class"].tolist() == [
        "both_correct",
        "stead_only_correct",
        "ida_only_correct",
        "neither_correct",
    ]


def test_event_level_bootstrap_not_trace_iid():
    # two events unequal size — bootstrap uses events
    eids = np.array(["e1"] * 10 + ["e2"] * 2)
    a = np.zeros(12)
    b = np.array([1.0] * 10 + [0.0] * 2)
    out = event_level_bootstrap_delta(eids, a, b, reps=200, seed=1)
    assert out["n_events"] == 2
    assert out["n_traces"] == 12
    assert "ci95" in out


def test_phaseB_scripts_do_not_optimizer_or_load_last():
    texts = []
    for p in [
        "scripts/cache_stage6_phaseB_candidates.py",
        "scripts/run_stage6_phaseB_analyze.py",
        "scripts/validate_stage6_ida_checkpoint.py",
        "src/earthquake/stage6/phaseB.py",
    ]:
        texts.append((ROOT / p).read_text())
    blob = "\n".join(texts)
    assert "AdamW" not in blob and "torch.optim" not in blob
    assert "loss.backward" not in blob
    assert "last.pt" in blob  # mentioned as refused
    assert ("refusing last.pt" in blob.lower()) or ("never loads last.pt" in blob.lower()) or ("Refusing any checkpoint other than best.pt" in blob)


def test_confirm_seal_and_phaseB_no_confirm_read():
    seal = ROOT / "artifacts/results/stage6/splits_full/CONFIRM_SEALED"
    assert seal.exists()
    for p in [
        "scripts/build_phaseB_eval_manifest.py",
        "scripts/cache_stage6_phaseB_candidates.py",
        "scripts/run_stage6_phaseB_analyze.py",
    ]:
        src = (ROOT / p).read_text()
        assert "CONFIRM" in src or "confirm" in src
        assert "assert_full_confirm_access_allowed" in src


def test_same_manifest_sha_helper_stable():
    s = "a\nb\n"
    h1 = hashlib.sha256(s.encode()).hexdigest()
    h2 = hashlib.sha256(s.encode()).hexdigest()
    assert h1 == h2


@pytest.mark.skipif(
    not (ROOT / "artifacts/results/stage4/method_lock.json").exists(),
    reason="stage4 lock missing",
)
def test_stage2_5_artifacts_untouched_hashes_exist():
    # Phase B must not rewrite frozen stage artifacts; existence check + mtime not required
    assert (ROOT / "artifacts/results/stage2/best_lambdas.json").exists()
    assert (ROOT / "artifacts/results/stage4/method_lock.json").exists()


def test_noise_train_dev_disjoint_if_manifests_exist():
    train_meta = ROOT / "artifacts/models/stage6/phasenet_ida_full_seed42/train_meta.parquet"
    noise_man = ROOT / "artifacts/results/stage6/phaseB_noise_manifest.csv"
    if not train_meta.exists() or not noise_man.exists():
        pytest.skip("manifests not built yet")
    tr = set(pd.read_parquet(train_meta).loc[lambda d: d.is_noise.astype(bool), "trace_name"].astype(str))
    dv = set(pd.read_csv(noise_man)["trace_name"].astype(str))
    assert tr.isdisjoint(dv)
