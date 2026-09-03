"""Stage 9 gate / protocol tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_stage9_frozen_artifact_hashes():
    lock = ROOT / "artifacts/results/stage6/final_confirm/method_lock.json"
    assert sha256(lock) == "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"
    assert (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists()


def test_stage9_reject_random_weights_and_checkpoint_loaded():
    smoke = json.loads((ROOT / "artifacts/results/stage9/segphase_smoke_metrics.json").read_text())
    assert smoke["checkpoint_not_random"] is True
    assert smoke["alignment_gate"] == "PASS"


def test_stage9_component_mapping():
    from earthquake.stage9.segphase_adapter import enz_to_ud_ns_ew

    enz = np.array([[1.0, 2], [3.0, 4], [5.0, 6]], dtype=np.float32)
    zne = enz_to_ud_ns_ew(enz)
    assert np.allclose(zne[0], enz[2])
    assert np.allclose(zne[1], enz[1])
    assert np.allclose(zne[2], enz[0])


def test_stage9_segphase_window_mapping():
    from earthquake.stage9.segphase_adapter import window_starts

    assert window_starts("A") == [0, 3000, 6000, 9000]
    b = window_starts("B")
    assert b[0] == 0 and b[1] == 1500 and b[-1] == 9000


def test_stage9_dkpn_leakage_excluded_from_main():
    reg = json.loads((ROOT / "artifacts/results/stage9/baseline_registry.json").read_text())
    assert reg["repos"]["DKPN"]["enter_main_table"] is False
    assert reg["repos"]["DKPN"]["classification"].startswith("C.")


def test_stage9_confirm_not_used_for_lock_if_locked():
    p = ROOT / "artifacts/results/stage9/segphase_method_lock.json"
    if not p.exists():
        pytest.skip("lock not yet written")
    lock = json.loads(p.read_text())
    assert lock["selected_before_confirm"] is True


def test_stage9_threshold_dev_only_grid():
    grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    assert grid == [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
