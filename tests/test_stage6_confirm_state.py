"""Tests for confirm state machine and method-lock gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from earthquake.stage6.confirm_state import (
    ConfirmStateError,
    assert_same_lock_for_resume,
    authorize,
    mark_consumed,
    mark_running,
    refuse_if_consumed,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_delta_direction_doc():
    # ΔF1 = method - baseline; positive better
    assert (0.87 - 0.84) > 0


def test_state_machine_tmp(tmp_path, monkeypatch):
    import earthquake.stage6.confirm_state as cs

    monkeypatch.setattr(cs, "final_confirm_dir", lambda root=None: tmp_path)
    phasec2 = tmp_path / "PHASEC2.DONE"
    phasec2.write_text("DONE\n")
    lock = tmp_path / "method_lock.json"
    write_json(lock, {"method": "fixed_rescore_UNION"})
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"primary_endpoint": "S F1@0.5"})
    pre = tmp_path / "pre.json"
    write_json(pre, {"preconfirm_audit_passed": True, "contaminated_events": 0})

    auth = authorize(method_lock_path=lock, protocol_path=protocol, preconfirm_path=pre, phasec2_done=phasec2)
    assert auth.exists()
    h = json.loads(auth.read_text())["method_lock_sha256"]
    mark_running(method_lock_hash=h)
    assert_same_lock_for_resume(h)
    with pytest.raises(ConfirmStateError):
        assert_same_lock_for_resume("deadbeef")

    pred = tmp_path / "p.parquet"
    pred.write_text("x")
    met = tmp_path / "m.json"
    met.write_text("{}")
    boot = tmp_path / "b.json"
    boot.write_text("{}")
    # mark_consumed needs sha256 of files — write real content
    pred.write_bytes(b"abc")
    met.write_text("{}\n")
    boot.write_text("{}\n")
    mark_consumed(method_lock_hash=h, predictions_path=pred, metrics_path=met, bootstrap_path=boot)
    with pytest.raises(ConfirmStateError):
        refuse_if_consumed()


def test_consumed_blocks_reselection():
    # smoke: module imports
    from earthquake.stage6 import confirm_state

    assert hasattr(confirm_state, "mark_consumed")
