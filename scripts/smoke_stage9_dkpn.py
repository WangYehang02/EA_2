#!/usr/bin/env python
"""DKPN diagnostic smoke — INSTANCE weights; not for main table."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.stage9.dkpn_adapter import DKPNConfig, DKPNWeightAudit


def main() -> int:
    out = ROOT / "artifacts/results/stage9"
    reports = ROOT / "reports/stage9"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    audit = DKPNWeightAudit(DKPNConfig())
    summary = audit.summary()
    summary.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "C. possible_INSTANCE_confirm_leakage_diagnostic_only",
            "enter_main_table": False,
            "enter_formal_bootstrap": False,
            "full_annotate_smoke": "skipped_main_path_blocked_by_leakage_policy",
            "note": "Weights load successfully as tensors; no Stage-6 confirm comparator run.",
            "alignment_gate": "DIAGNOSTIC_ONLY_NO_FULL_EVAL",
            "clean_retrain_plan_only": True,
        }
    )
    (out / "dkpn_smoke_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = f"""# Stage 9 — DKPN Alignment / Diagnostic Smoke

**Role:** `diagnostic_only_possible_leakage`  
**alignment_gate:** `DIAGNOSTIC_ONLY_NO_FULL_EVAL`  
**enter_main_table:** false

## Checkpoint

- path: `{summary['checkpoint']}`
- sha256: `{summary['sha256']}`
- n_tensors: {summary['n_tensors']}
- n_params: {summary['n_params']}
- loaded (not random init): `{summary['loaded']}`

## Why not main Table A

All official paper weights are trained on **INSTANCE** (`TrainDataset_INSTANCE` in filenames).
Cannot prove exclusion of Stage-6 confirm events → **no formal Ours−DKPN ranking / bootstrap**.

## Clean retrain (not started)

Would require Stage-6 picker_train event set, same labels/noise, confirm held out, then user approval.
"""
    (reports / "dkpn_alignment_sanity.md").write_text(md)
    print(json.dumps({"dkpn": "diagnostic_only", "loaded": summary["loaded"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
