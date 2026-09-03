#!/usr/bin/env python
"""Authorize one-shot confirm evaluation after locks/audits/tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir
from earthquake.stage6.confirm_state import authorize


def main() -> None:
    marker = artifacts_dir() / "results" / "stage6" / "final_confirm" / "PYTEST_PRECONFIRM.OK"
    path = authorize(
        method_lock_path=artifacts_dir() / "results" / "stage6" / "final_confirm" / "method_lock.json",
        protocol_path=artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_evaluation_protocol.json",
        preconfirm_path=artifacts_dir() / "results" / "stage6" / "final_confirm" / "preconfirm_contamination_audit.json",
        phasec2_done=artifacts_dir() / "results" / "stage6" / "phaseC2" / "PHASEC2.DONE",
        pytest_ok_marker=marker if marker.exists() else None,
    )
    print({"authorized": str(path)})


if __name__ == "__main__":
    main()
