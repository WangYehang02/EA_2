"""Confirm evaluation state machine: SEALED → AUTHORIZED → RUNNING → CONSUMED."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from earthquake.stage6.phaseB import sha256_file


class ConfirmStateError(RuntimeError):
    pass


def final_confirm_dir(root: Path | None = None) -> Path:
    from earthquake.config import artifacts_dir

    return (root or artifacts_dir()) / "results" / "stage6" / "final_confirm"


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return sha256_file(path)


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ConfirmStateError(f"missing required files: {missing}")


def authorize(
    *,
    method_lock_path: Path,
    protocol_path: Path,
    preconfirm_path: Path,
    phasec2_done: Path,
    pytest_ok_marker: Path | None = None,
) -> Path:
    d = final_confirm_dir()
    d.mkdir(parents=True, exist_ok=True)
    require_files([method_lock_path, protocol_path, preconfirm_path, phasec2_done])
    pre = json.loads(preconfirm_path.read_text())
    if not pre.get("preconfirm_audit_passed"):
        raise ConfirmStateError("preconfirm audit not passed")
    if pre.get("contaminated_events", 1) != 0:
        raise ConfirmStateError("contaminated confirm events")
    if (d / "CONFIRM.CONSUMED").exists():
        raise ConfirmStateError("confirm already consumed")
    if (d / "CONFIRM_EVAL.AUTHORIZED").exists():
        raise ConfirmStateError("already authorized")
    payload = {
        "status": "AUTHORIZED",
        "one_shot": True,
        "method_lock_sha256": sha256_file(method_lock_path),
        "protocol_sha256": sha256_file(protocol_path),
        "preconfirm_audit_sha256": sha256_file(preconfirm_path),
        "phasec2_done": str(phasec2_done),
        "authorized_utc": datetime.now(timezone.utc).isoformat(),
        "pytest_ok_marker": str(pytest_ok_marker) if pytest_ok_marker else None,
    }
    path = d / "CONFIRM_EVAL.AUTHORIZED"
    write_json(path, payload)
    return path


def mark_running(*, method_lock_hash: str) -> Path:
    d = final_confirm_dir()
    auth = d / "CONFIRM_EVAL.AUTHORIZED"
    if not auth.exists():
        raise ConfirmStateError("not authorized")
    if (d / "CONFIRM.CONSUMED").exists():
        raise ConfirmStateError("already consumed")
    auth_doc = json.loads(auth.read_text())
    if auth_doc["method_lock_sha256"] != method_lock_hash:
        raise ConfirmStateError("method lock hash mismatch vs AUTHORIZED")
    payload = {
        "status": "RUNNING",
        "method_lock_sha256": method_lock_hash,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot": True,
        "metrics_hidden_until_consumed": True,
    }
    path = d / "CONFIRM_EVAL.RUNNING"
    write_json(path, payload)
    return path


def assert_same_lock_for_resume(method_lock_hash: str) -> None:
    d = final_confirm_dir()
    running = d / "CONFIRM_EVAL.RUNNING"
    if not running.exists():
        return
    doc = json.loads(running.read_text())
    if doc.get("method_lock_sha256") != method_lock_hash:
        raise ConfirmStateError("resume forbidden: different method lock hash")


def mark_consumed(
    *,
    method_lock_hash: str,
    predictions_path: Path,
    metrics_path: Path,
    bootstrap_path: Path,
) -> Path:
    d = final_confirm_dir()
    if not (d / "CONFIRM_EVAL.RUNNING").exists():
        raise ConfirmStateError("not running")
    if (d / "CONFIRM.CONSUMED").exists():
        raise ConfirmStateError("already consumed")
    payload = {
        "status": "CONSUMED",
        "method_lock_sha256": method_lock_hash,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "confirm_may_not_be_reused_for_model_selection": True,
        "one_shot": True,
    }
    path = d / "CONFIRM.CONSUMED"
    write_json(path, payload)
    return path


def refuse_if_consumed() -> None:
    if (final_confirm_dir() / "CONFIRM.CONSUMED").exists():
        raise ConfirmStateError("CONFIRM.CONSUMED — method selection / re-eval forbidden")
