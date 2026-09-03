#!/usr/bin/env python
"""Stage 7A: write comparator registry + reproducibility audit (no GPU)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage7")
    reports = ensure_dir(ROOT / "reports" / "stage7")
    pn = Path.home() / ".seisbench" / "models" / "v3" / "phasenet"
    eq = Path.home() / ".seisbench" / "models" / "v3" / "eqtransformer"

    registry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seisbench_version": "0.12.3",
        "threshold_grid_preregistered": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
        "models": {
            "PhaseNet-STEAD": {
                "official_source": "seisbench.models.PhaseNet.from_pretrained('stead')",
                "weight_name": "stead",
                "weight_path": str(pn / "stead.pt.v2"),
                "weight_sha256": sha256_file(pn / "stead.pt.v2") if (pn / "stead.pt.v2").exists() else None,
                "training_dataset": "STEAD",
                "possible_confirm_leakage": False,
                "sampling_rate": 100,
                "component_order": "ZNE",
                "phases": "PSN",
                "valid_comparator": True,
                "role": "frozen_stage6_baseline",
                "run_in_stage7A": False,
                "reuse_stage6_confirm_preds": True,
                "exclusion_reason": None,
            },
            "PhaseNet-ETHZ": {
                "official_source": "seisbench.models.PhaseNet.from_pretrained('ethz')",
                "weight_name": "ethz",
                "weight_path": str(pn / "ethz.pt.v2"),
                "weight_sha256": sha256_file(pn / "ethz.pt.v2"),
                "training_dataset": "ETHZ",
                "possible_confirm_leakage": False,
                "sampling_rate": 100,
                "component_order": "ZNE",
                "phases": "PSN",
                "valid_comparator": True,
                "role": "new_external_comparator",
                "run_in_stage7A": True,
                "default_S_threshold_official": 0.34493361542513395,
                "threshold_selection": "preregistered_grid_on_stage6_dev_only",
                "exclusion_reason": None,
            },
            "PhaseNet-SCEDC": {
                "official_source": "seisbench.models.PhaseNet.from_pretrained('scedc')",
                "weight_name": "scedc",
                "weight_path": str(pn / "scedc.pt.v2"),
                "weight_sha256": sha256_file(pn / "scedc.pt.v2"),
                "training_dataset": "SCEDC",
                "possible_confirm_leakage": False,
                "sampling_rate": 100,
                "component_order": "ZNE",
                "phases": "PSN",
                "valid_comparator": True,
                "role": "new_external_comparator",
                "run_in_stage7A": True,
                "default_S_threshold_official": 0.4039903966026159,
                "threshold_selection": "preregistered_grid_on_stage6_dev_only",
                "exclusion_reason": None,
            },
            "EQTransformer-STEAD": {
                "official_source": "seisbench.models.EQTransformer.from_pretrained('stead')",
                "weight_name": "stead",
                "valid_comparator": False,
                "run_in_stage7A": False,
                "exclusion_reason": "official_weight_download_failed_SSLError_network; local cache empty",
            },
            "EQTransformer-ETHZ": {
                "official_source": "seisbench.models.EQTransformer.from_pretrained('ethz')",
                "valid_comparator": False,
                "run_in_stage7A": False,
                "exclusion_reason": "official_weight_download_failed_SSLError_network; local cache empty",
            },
            "PhaseNet-instance": {
                "valid_comparator": False,
                "role": "diagnostic_only_data_leakage",
                "run_in_stage7A": False,
                "possible_confirm_leakage": True,
                "exclusion_reason": "INSTANCE-trained weights may include confirm events; leakage diagnostic only",
            },
            "EQTransformer-instance": {
                "valid_comparator": False,
                "role": "diagnostic_only_data_leakage",
                "run_in_stage7A": False,
                "possible_confirm_leakage": True,
                "exclusion_reason": "INSTANCE-trained weights may include confirm events; leakage diagnostic only; weight unavailable locally",
            },
            "LFTNet": {
                "valid_comparator": False,
                "LFTNet_status": "not_reproducible_from_official_release",
                "run_in_stage7A": False,
                "exclusion_reason": "no local official package/checkpoint; incomplete reproducibility gates",
            },
            "PhaseNO": {
                "valid_comparator": False,
                "exclusion_reason": "multi-station input; incompatible with single-station Stage-6 protocol",
            },
            "SegPhase": {
                "valid_comparator": False,
                "exclusion_reason": "region/sampling/training protocol incompatible; no fully compatible official checkpoint verified",
            },
            "GreenPhase": {
                "valid_comparator": False,
                "exclusion_reason": "no compatible official public implementation verified in this environment",
            },
            "PhaseNet-DAS": {
                "valid_comparator": False,
                "exclusion_reason": "DAS modality incompatible",
            },
            "PhaseNet+": {
                "valid_comparator": False,
                "exclusion_reason": "multi-task objectives; not a dedicated picking comparator under our protocol",
            },
        },
    }
    new_valid = [k for k, v in registry["models"].items() if v.get("run_in_stage7A")]
    registry["n_new_valid_comparators"] = len(new_valid)
    registry["new_valid_comparators"] = new_valid
    registry["proceed_gpu_batch"] = len(new_valid) >= 2
    save_json(registry, out / "comparator_registry.json")

    md = f"""# Stage 7A — Comparator Reproducibility Audit

**SeisBench:** 0.12.3  
**New valid comparators runnable:** {len(new_valid)} → `{new_valid}`  
**Proceed GPU batch:** `{registry['proceed_gpu_batch']}`

## Valid / runnable

| Model | Dataset | Leakage risk | Notes |
|--|--|--|--|
| PhaseNet-STEAD | STEAD | no | Frozen Stage-6 baseline (reuse confirm preds) |
| PhaseNet-ETHZ | ETHZ | no | Local official weight; Stage-7A run |
| PhaseNet-SCEDC | SCEDC | no | Local official weight; Stage-7A run |

## Excluded

| Model | Reason |
|--|--|
| EQTransformer-STEAD/ETHZ | Official weight download failed (SSL/network); empty local cache |
| PhaseNet/EQT-instance | `diagnostic_only_data_leakage` — not run by default |
| LFTNet | `not_reproducible_from_official_release` |
| PhaseNO / SegPhase / GreenPhase / PhaseNet-DAS / PhaseNet+ | incompatible protocol/modality/implementation |

## Threshold policy

Preregistered grid on Stage-6 **dev only**: `[0.05, 0.10, 0.20, 0.30, 0.50, 0.70]`.  
Primary: S F1@0.5. Tie-break: F1@0.1 → miss → P95 → higher threshold.
"""
    (reports / "comparator_reproducibility_audit.md").write_text(md)
    print(json.dumps({"proceed": registry["proceed_gpu_batch"], "new_valid": new_valid}, indent=2))


if __name__ == "__main__":
    main()
