#!/usr/bin/env python
"""Write pre-registered confirm evaluation protocol (before AUTHORIZED)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "final_confirm")
    protocol = {
        "primary_comparison": "fixed_rescore_UNION vs STEAD_top1",
        "primary_endpoint": "S F1@0.5",
        "secondary_comparisons": [
            "fixed_rescore_STEAD vs STEAD_top1",
            "fixed_rescore_UNION vs fixed_rescore_STEAD",
        ],
        "diagnostic_only": ["IDA_top1", "UNION_candidate_oracle"],
        "forbidden_on_confirm": ["R1", "R2", "R3", "forced_choice", "none_fallback_fixed", "new_gate", "new_ranker"],
        "bootstrap": {"n": 5000, "seed": 20260817, "unit": "event"},
        "decision_rules": {
            "strong_confirmed": "dF1_05>=0.01 and CI_lo>0 and dF1_01>=-0.003 and dP95<=0 and no selective recall collapse",
            "modest_confirmed": "dF1_05>0 and CI_lo>0 and gain<0.01 and safety ok",
            "direction_only_underpowered": "point positive but CI includes 0",
            "not_confirmed": "dF1_05<=0 or safety failure",
        },
        "sota_claim_allowed": False,
        "blind_picker_claim_allowed": False,
        "multistation_may_start": False,
        "claim_scope": "catalog-assisted S-phase candidate re-picking/refinement",
        "no_midrun_metrics": True,
        "population": "all_s_labelled_internal_confirm_no_head_cap",
    }
    path = out / "confirm_evaluation_protocol.json"
    save_json(protocol, path)
    print(json.dumps({"protocol_sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
