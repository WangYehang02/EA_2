"""Future blind runner tests: synthetic + Stage-6 dev caches. No confirm I/O."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earthquake.config import artifacts_dir
from earthquake.stage10.future_blind_runner import (
    ALLOWED_PRIMARY_METHODS,
    BlindRunnerError,
    atomic_freeze_predictions,
    expected_s_maps,
    predict_fixed_rescore_union,
    predict_stead_top1,
    refuse_confirm_or_diting_path,
    rescore_one_trace,
    run_preregistered_diagnostics,
    run_primary,
    verify_final_method_lock,
    wrap_missing_channel_read,
)
from earthquake.stage10.preconfirm_readiness import sha256_file

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "artifacts" / "results" / "stage10"
RUNNER_SRC = ROOT / "src" / "earthquake" / "stage10" / "future_blind_runner.py"


def test_frozen_preconfirm_locks_unchanged():
    for name, side in (
        ("FINAL_METHOD.LOCK.json", "FINAL_METHOD.LOCK.sha256"),
        ("CONFIRM_ANALYSIS.LOCK.json", "CONFIRM_ANALYSIS.LOCK.sha256"),
        ("CONFIRM_READINESS.json", "CONFIRM_READINESS.sha256"),
    ):
        assert sha256_file(LOCK_DIR / name) == (LOCK_DIR / side).read_text().strip()


def test_verify_final_method_lock_and_checkpoints():
    lock = verify_final_method_lock(LOCK_DIR)
    assert lock["primary_lock_name"] == "fixed_rescore_UNION"
    assert lock["primary_baseline"]["name"] == "STEAD_top1"
    assert set(ALLOWED_PRIMARY_METHODS) == {"fixed_rescore_UNION", "STEAD_top1"}


def test_runner_source_has_no_search_and_forces_lock():
    src = RUNNER_SRC.read_text()
    tree = ast.parse(src)
    assert "verify_final_method_lock" in src
    assert "FINAL_METHOD.LOCK" in src
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert "glob" not in calls
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "verify_final_method_lock" in names
    assert "predict_fixed_rescore_union" in names


def test_refuse_confirm_and_diting_paths():
    with pytest.raises(BlindRunnerError):
        refuse_confirm_or_diting_path("/x/final_confirm/foo.npy")
    with pytest.raises(BlindRunnerError):
        refuse_confirm_or_diting_path("/data/diting/wave.h5")
    refuse_confirm_or_diting_path(str(ROOT / "artifacts/cache/stage6/phaseC/dev_union.parquet"))


def test_fail_closed_nan_inf_empty_and_missing_channel():
    empty = pd.DataFrame(
        columns=["candidate_sample", "stead_probability", "ida_probability", "candidate_index"]
    )
    with pytest.raises(BlindRunnerError, match="no UNION candidates"):
        rescore_one_trace(
            empty,
            expected_s_sample=100.0,
            sigma_samples=10.0,
            history_available=True,
            lw=0.5,
            lh=2.0,
            lp=0.0,
        )
    bad = pd.DataFrame(
        {
            "candidate_sample": [10.0],
            "stead_probability": [np.nan],
            "ida_probability": [np.inf],
            "candidate_index": [0],
        }
    )
    with pytest.raises(BlindRunnerError, match="no finite source probability"):
        rescore_one_trace(
            bad,
            expected_s_sample=10.0,
            sigma_samples=5.0,
            history_available=True,
            lw=0.5,
            lh=2.0,
            lp=0.0,
        )
    with pytest.raises(BlindRunnerError, match="nonfinite expected"):
        rescore_one_trace(
            pd.DataFrame(
                {
                    "candidate_sample": [10.0],
                    "stead_probability": [0.9],
                    "ida_probability": [np.nan],
                    "candidate_index": [0],
                }
            ),
            expected_s_sample=float("nan"),
            sigma_samples=5.0,
            history_available=True,
            lw=0.5,
            lh=2.0,
            lp=0.0,
        )

    def boom(_name=None):
        raise KeyError("Missing component Z")

    with pytest.raises(BlindRunnerError, match="missing/corrupt"):
        wrap_missing_channel_read(boom, "trace")
    with pytest.raises(BlindRunnerError, match="nonfinite waveform"):
        wrap_missing_channel_read(lambda: np.array([[1.0, np.nan], [0.0, 0.0], [0.0, 0.0]]))


def test_diagnostics_only_after_freeze_and_oracle_rejected(tmp_path):
    lock = verify_final_method_lock(LOCK_DIR)
    names = ["t0"]
    union = pd.DataFrame(
        {
            "trace_name": ["t0", "t0"],
            "candidate_sample": [50.0, 200.0],
            "stead_probability": [0.9, 0.8],
            "ida_probability": [np.nan, 0.7],
            "candidate_index": [0, 1],
        }
    )
    out = tmp_path / "primary.npy"
    freeze = run_primary(
        "fixed_rescore_UNION",
        lock=lock,
        union=union,
        stead_cache=None,
        names=names,
        exp_s={"t0": 200.0},
        sigma={"t0": 20.0},
        hist_ok={"t0": True},
        out_path=out,
    )
    assert freeze.frozen
    assert out.exists()
    diag = run_preregistered_diagnostics(primary=freeze)
    assert diag["oracle_used_for_prediction"] is False
    with pytest.raises(BlindRunnerError, match="oracle"):
        run_preregistered_diagnostics(primary=freeze, oracle_fn=lambda: None)
    unfinished = freeze
    unfinished.frozen = False
    with pytest.raises(BlindRunnerError, match="before primary freeze"):
        run_preregistered_diagnostics(primary=unfinished)


def test_atomic_freeze_refuses_confirm_path(tmp_path):
    with pytest.raises(BlindRunnerError):
        atomic_freeze_predictions(tmp_path / "final_confirm" / "x.npy", np.array([1.0]))


def test_dev_fixed_rescore_and_stead_top1_match_frozen_npy():
    """Trace-by-trace vs Stage-6 full-dev caches/npy. No waveforms."""
    lock = verify_final_method_lock(LOCK_DIR)
    art = artifacts_dir()
    refuse_confirm_or_diting_path(art / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    meta = pd.read_csv(art / "results" / "stage6" / "phaseB_eval_manifest.csv")
    names = meta["trace_name"].astype(str).to_numpy()
    union = pd.read_parquet(art / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    stead = pd.read_parquet(art / "cache" / "stage6" / "phaseB" / "stead_top10" / "stead_top10.parquet")
    hist = pd.read_parquet(
        art / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"
    )
    hist = hist[hist["subset"] == "stage6_dev"].drop_duplicates("trace_name")
    fr = lock["fixed_rescore"]
    exp_s, sigma, hist_ok = expected_s_maps(
        meta,
        hist,
        fr["global_residual"],
        shrink_k=float(fr["shrinkage_k"]),
        min_history=int(fr["min_history"]),
        mad_disable_s=float(fr["mad_disable_s"]),
        min_sigma_s=float(fr["min_sigma_s"]),
        max_sigma_s=float(fr["max_sigma_s"]),
    )
    new_union = predict_fixed_rescore_union(
        union,
        names,
        exp_s=exp_s,
        sigma=sigma,
        hist_ok=hist_ok,
        lw=float(fr["lambdas_s"]["lw"]),
        lh=float(fr["lambdas_s"]["lh"]),
        lp=float(fr["lambdas_s"]["lp"]),
    )
    old_union = np.load(art / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy")
    assert old_union.shape == new_union.shape
    assert np.isfinite(old_union).all()
    np.testing.assert_array_equal(new_union, old_union)

    new_stead = predict_stead_top1(stead, names)
    old_stead = np.load(art / "results" / "stage6" / "phaseC" / "baseline_preds" / "STEAD_top1.npy")
    np.testing.assert_array_equal(new_stead, old_stead)


def test_pytest_does_not_import_confirm_loaders():
    tree = ast.parse(Path(__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "earthquake.data.hdf5_reader" not in imported
