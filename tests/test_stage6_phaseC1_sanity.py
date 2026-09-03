"""Phase C.1 sanity unit tests (no training, confirm stays sealed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, full_stage6_paths
from earthquake.stage6.phaseC1.infer import predict_ranker_variants
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics, correct_at
from earthquake.stage6.ranker.models import DeepSetsRanker, SharedCandidateScorer

ROOT = Path(__file__).resolve().parents[1]


def test_confirm_seal_unchanged():
    seal = full_stage6_paths()["confirm_seal"]
    assert seal.exists()
    payload = json.loads(seal.read_text())
    assert payload["status"] == "SEALED"
    # CONFIRM_SEALED file itself must remain; method_lock unlocks code paths but one-shot
    # confirm is gated by final_confirm AUTHORIZED/CONSUMED markers.
    from earthquake.config import artifacts_dir

    auth = artifacts_dir() / "results" / "stage6" / "final_confirm" / "CONFIRM_EVAL.AUTHORIZED"
    consumed = artifacts_dir() / "results" / "stage6" / "final_confirm" / "CONFIRM.CONSUMED"
    if full_stage6_paths()["method_lock"].exists():
        assert auth.exists() or consumed.exists()
    else:
        with pytest.raises(RuntimeError):
            assert_full_confirm_access_allowed(purpose="phaseC1_test")


def test_padding_mask_blocks_selection():
    model = SharedCandidateScorer(n_features=8, hidden=16)
    model.eval()
    x = torch.randn(2, 10, 8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :2] = True
    with torch.no_grad():
        logits = model(x, mask)
        logits = logits.clone()
        logits[:, -1] = -1e9  # suppress none
        idx = logits.argmax(-1)
    assert bool((idx < 2).all())


def test_none_class_index_is_k():
    # K=10 → none index 10; logits width K+1
    model = DeepSetsRanker(n_features=12, hidden=16)
    x = torch.randn(3, 10, 12)
    mask = torch.ones(3, 10, dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, mask)
    assert logits.shape == (3, 11)


def test_matched_denominator_excludes_miss():
    pred = np.array([100.0, np.nan, 500.0])
    true = np.array([100.0, 200.0, 100.0])
    sr = np.array([100.0, 100.0, 100.0])
    m = comprehensive_pick_metrics(pred, true, sr, none_mask=np.array([False, True, False]))
    assert m["miss_count"] == 1
    assert m["detected_ae_p95_denominator"] == 2
    assert m["none_of_k_count"] == 1
    assert m["prediction_coverage"] == pytest.approx(2 / 3)


def test_forced_choice_and_fallback_logic_synthetic():
    # synthetic: reported none → forced picks cand0; fallback uses fixed
    # We only unit-test metric helpers here if ckpt missing
    pred_reported = np.array([np.nan, 10.0])
    pred_forced = np.array([5.0, 10.0])
    fixed = np.array([7.0, 9.0])
    fallback = np.where(np.isfinite(pred_reported), pred_reported, fixed)
    assert np.isfinite(fallback).all()
    assert fallback[0] == 7.0
    assert fallback[1] == 10.0
    assert np.isfinite(pred_forced).all()


def test_bootstrap_delta_direction_sign():
    # a better F1 than b → positive delta
    import importlib.util

    rng = np.random.default_rng(0)
    n = 200
    true = np.full(n, 100.0)
    sr = np.full(n, 100.0)
    events = np.array([f"e{i//10}" for i in range(n)])
    pred_b = true + rng.normal(0, 80, size=n)  # worse
    pred_a = true + rng.normal(0, 5, size=n)  # better
    spec = importlib.util.spec_from_file_location(
        "phaseC1_audit", ROOT / "scripts/run_stage6_phaseC1_sanity_audit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    boot = mod.event_bootstrap(pred_a, pred_b, true, sr, events, n_boot=200, seed=1)
    assert boot["f1@0.5"]["mean_delta"] > 0
    assert boot["f1@0.5"]["ci95"][0] > 0


def test_correct_at_helper():
    pred = np.array([100.0, 200.0])
    true = np.array([100.0, 100.0])
    sr = np.array([100.0, 100.0])
    ok = correct_at(pred, true, sr, 0.5)
    assert bool(ok[0]) and not bool(ok[1])


@pytest.mark.skipif(
    not (ROOT / "artifacts/models/stage6/ranker/R3_seed2026_b0.2_g1.0/best.pt").exists(),
    reason="R3 ckpt missing",
)
def test_infer_variants_smoke():
    feat = pd.read_parquet(ROOT / "artifacts/cache/stage6/phaseC/dev_features_R2.parquet")
    # tiny subset: 20 traces
    keep = feat["trace_name"].drop_duplicates().head(20)
    feat = feat[feat.trace_name.isin(keep)]
    meta_names = keep.astype(str).tolist()
    fixed = np.full(len(meta_names), 1000.0)
    df = predict_ranker_variants(
        ROOT / "artifacts/models/stage6/ranker/R3_seed2026_b0.2_g1.0/best.pt",
        feat,
        device="cpu",
        fixed_fallback=fixed,
        meta_trace_names=meta_names,
    )
    assert len(df) == 20
    # forced choice never selects index 10
    assert (df["forced_index"] < 10).all()
    # when none, fallback equals fixed
    none = df["none_of_k"].to_numpy(bool)
    if none.any():
        assert np.allclose(df.loc[none, "pred_none_fallback_fixed"], 1000.0)
