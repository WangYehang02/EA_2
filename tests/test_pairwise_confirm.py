"""Confirm protocol tests (structural; no full confirm IO required for most)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

EXPECTED = "1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b"


def test_fulldev_method_lock_hash_matches_preregistered():
    p = ROOT / "artifacts/results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256"
    assert p.read_text().strip() == EXPECTED


def test_execution_lock_if_present():
    p = ROOT / "artifacts/results/pairwise_confirm/SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.json"
    if not p.exists():
        pytest.skip("execution lock not written")
    d = json.loads(p.read_text())
    assert d["method_lock_hash"] == EXPECTED
    assert d["tau"] == 0.5
    assert d["confirm_is_evaluation_only"] is True
    assert "no parameter selection" in d["declaration"]


def test_confirm_watchdog_command_is_confirm_only():
    import watch_pairwise_confirm_gpu as w

    cmd = " ".join(w.EVAL_CMD).lower()
    assert "eval_scalar_pairwise_confirm" in cmd
    assert "fulldev" not in Path(w.EVAL_CMD[1]).name
    assert "waveform" not in cmd


def test_tau_frozen():
    from earthquake.pairwise.fulldev import TAU

    assert TAU == 0.5


def test_select_gpu_confirm_watchdog():
    import watch_pairwise_confirm_gpu as w

    gpus = [
        w.GPUInfo(0, "a", free_mb=1000, used_mb=20000, util=0),
        w.GPUInfo(7, "b", free_mb=20000, used_mb=20, util=0),
    ]
    g = w.select_gpu(gpus, min_free_mb=4000, max_util=10, max_used_mb=1500)
    assert g is not None and g.index == 7
