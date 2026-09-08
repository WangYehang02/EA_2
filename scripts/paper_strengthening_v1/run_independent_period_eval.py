#!/usr/bin/env python
"""Independent-period evaluation runner (blocked until external data is prepared).

Does NOT invent data. Writes a LOCKED protocol stub and exits with status explaining
what is missing. When a prepared package appears at the expected path, this script
will run A–E once under the frozen protocol.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "paper_strengthening_v1"
REPORT = ROOT / "reports" / "paper_strengthening_v1"
EXPECTED_PACKAGE = OUT / "independent_period" / "READY.json"


def main() -> int:
    ensure_dir(OUT / "independent_period")
    ensure_dir(OUT / "locks")
    ensure_dir(REPORT)

    main_ctrl = json.loads((OUT / "locks" / "MAIN_STRONG_CONTROL.LOCK.json").read_text())
    protocol = {
        "name": "paper_strengthening_v1_independent_period",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "WAITING_FOR_DATA",
        "primary_comparison": "E_scalar - calibration_selected_main_strong_control",
        "main_strong_control": main_ctrl["main_strong_control"],
        "methods": ["A_fixed", "B_resid", "C_best", "D_best", "E_scalar"],
        "rules": {
            "shared_frozen_candidates": True,
            "shared_c1_c2": True,
            "no_abstain": True,
            "no_label_features": True,
            "windowing_not_centered_on_manual_S": True,
            "single_candidate_fallback_c1": True,
            "one_shot_after_freeze": True,
            "no_retune_on_test": True,
        },
        "expected_ready_file": str(EXPECTED_PACKAGE),
        "ready_file_schema": {
            "pairs_parquet": "path to pairs with same schema as pairs_ranker_train (c1/c2 features + true_s for eval only)",
            "manifest_csv": "trace/event metadata",
            "data_lock_sha256": "hash of frozen sample list",
            "independence_audit_json": "overlap audit vs all historical splits",
            "info_access_audit_json": "catalog-assisted fields disclosure",
        },
    }
    save_json(protocol, OUT / "locks" / "INDEPENDENT_PERIOD_PROTOCOL.WAITING.json")

    if not EXPECTED_PACKAGE.exists():
        msg = {
            "status": "INDEPENDENT_VALIDATION_NOT_COMPLETED",
            "reason": f"missing {EXPECTED_PACKAGE}",
            "do_not_substitute": ["confirm", "heldout_eval", "fulldev_phaseB"],
        }
        save_json(msg, OUT / "independent_period" / "STATUS.json")
        (REPORT / "INDEPENDENT_VALIDATION_NOT_COMPLETED.md").write_text(
            "# Independent validation NOT completed\n\n"
            "No prepared independent-period package was found.\n\n"
            f"Expected: `{EXPECTED_PACKAGE}`\n\n"
            "Do **not** treat confirm / heldout / full-dev as a substitute.\n",
            encoding="utf-8",
        )
        print(json.dumps(msg, indent=2))
        return 2

    # Future: load READY.json and run once. Intentionally not implemented with fake data.
    print("READY package present but full external pipeline not yet wired for this environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
