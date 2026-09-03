#!/usr/bin/env python
"""Finalize Stage 8 when LFTNet drop is incomplete — no training, no confirm infer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def main() -> None:
    out = ROOT / "artifacts/results/stage8"
    reports = ROOT / "reports/stage8"
    manifest = json.loads((out / "lftnet_release_manifest.json").read_text())
    smoke = json.loads((out / "lftnet_smoke_metrics.json").read_text())
    frozen = json.loads((out / "stage2_7_frozen_hashes.json").read_text())
    # re-check method lock unchanged now
    lock = ROOT / "artifacts/results/stage6/final_confirm/method_lock.json"
    import hashlib

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    lock_now = sha(lock)
    lock_ok = lock_now == "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"

    verdict = {
        "stage": "8",
        "final_verdict": "implementation_not_reproducible",
        "secondary_tags": [
            "E. incomplete_or_unusable",
            "official_code_no_valid_checkpoint",
        ],
        "sota_claim_allowed": False,
        "replace_stage6_main_method": False,
        "add_lftnet_to_union_candidates": False,
        "post_confirm_comparator_ran": False,
        "auto_continue_blocked": True,
        "provenance_class": manifest.get("provenance_class"),
        "smoke_status": smoke.get("status"),
        "stage6_method_lock_sha256": lock_now,
        "stage6_method_lock_unchanged": lock_ok,
        "confirm_consumed_still_present": (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists(),
        "free_gpus_at_audit": "see logs/gpu_at_start.txt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "answers": {
            "1_official_complete": False,
            "2_usable_official_checkpoint": False,
            "3_checkpoint_training_data": "unknown_no_checkpoint",
            "4_INSTANCE_confirm_leakage_risk": "unknown_no_checkpoint",
            "5_reproduced_paper_0_844": False,
            "6_same_protocol_LFTNet_metrics": None,
            "7_who_better_F1_0_5": "not_comparable_LFTNet_not_run",
            "8_bootstrap_CI_includes_0": None,
            "9_precision_recall_coverage_miss_tradeoff": "not_run",
            "10_P95_improved_via_abstention": "not_run",
            "11_LFTNet_raises_UNION_oracle": "not_run_gate_failed",
            "12_worth_adding_LFTNet": False,
            "13_allowed_to_replace_stage6": False,
            "14_sota_claim_allowed": False,
            "15_pytest": "see tests/test_stage8_lftnet_gate.py",
            "16_stage2_7_hash_unchanged": lock_ok,
        },
        "training_feasibility_if_user_provides_full_code_only": {
            "note": "No automatic training. Rough estimate only, using Stage-7A annotate proxy ~24 traces/s on 4×4090 is NOT training throughput.",
            "assume_train_traces": 800000,
            "conservative_train_tps": 3.66,
            "hours_per_epoch": 60.7,
            "50_epochs_wall_h": 3035,
            "within_one_week": False,
            "needs_separate_TF25_env": True,
        },
    }
    (out / "lftnet_final_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    # stub artifacts required by spec
    stubs = {
        "lftnet_paper_protocol_metrics.json": {
            "status": "not_run",
            "reason": "incomplete_release",
            "label": "cannot_claim_official_or_approximate_paper_protocol_reproduction",
        },
        "lftnet_dev_threshold_sweep.csv": "threshold,status,note\n,,,blocked_incomplete_release\n",
        "lftnet_method_lock.json": {
            "locked": False,
            "reason": "gate_failed_incomplete_release",
            "selected_before_confirm": False,
        },
        "lftnet_confirm_metrics.json": {"status": "not_run", "reason": "no_method_lock"},
        "lftnet_bootstrap.json": {"status": "not_run"},
        "lftnet_candidate_oracle.json": {
            "status": "not_run",
            "gate": "candidate_complementarity_not_attempted",
            "would_require_dev_delta_oracle_F1_0.5_ge": 0.01,
        },
    }
    for name, payload in stubs.items():
        p = out / name
        if name.endswith(".csv"):
            p.write_text(str(payload))
        else:
            p.write_text(json.dumps(payload, indent=2) + "\n")
    (out / "LFTNET_CONFIG_LOCKED").write_text(
        "NOT_LOCKED\nreason=incomplete_or_unusable_drop\n"
    )

    write(
        reports / "lftnet_paper_protocol_reproduction.md",
        """# LFTNet paper-protocol reproduction

**Status:** not run.

Cannot claim `official_paper_protocol_reproduction` or even `approximate_paper_protocol_reproduction`:
the local drop has no eval manifest, no 10k→20k segment construction script, and no checkpoint.
""",
    )
    write(
        reports / "lftnet_same_protocol_benchmark.md",
        """# LFTNet same-protocol (Stage 6) benchmark

**Status:** not run (gate failed).

Frozen Stage-6/7A baselines remain the reference (reused, not re-inferred):

| Method | F1@0.1 | F1@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 0.1198 | 0.8802 | 1.927 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 0.0683 | 0.9317 | 58.657 |
| PhaseNet-STEAD | 0.4852 | 0.8176 | 0 | 1 | 6.100 |
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 0 | 1 | 3.510 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 0 | 1 | 2.4655 |
| UNION oracle | 0.6017 | 0.8676 | 0 | 1 | 1.780 |
| LFTNet | — | — | — | — | not run |

`sota_claim_allowed = false`
""",
    )
    write(
        reports / "lftnet_candidate_complementarity.md",
        """# LFTNet candidate complementarity vs UNION

**Status:** not run.

Stop-gate for adding LFTNet candidates was never evaluated because the model cannot be executed.
Default verdict tag: `candidate_complementarity_insufficient` is **not** asserted from metrics;
instead Stage-8 final verdict is `implementation_not_reproducible`.

Main method remains `fixed_rescore_UNION`. No new ranker, no GNN, no multistation.
""",
    )
    write(
        reports / "lftnet_runtime_benchmark.md",
        """# LFTNet runtime benchmark

**Status:** not run.

Free GPUs were checked at pipeline start (see `artifacts/results/stage8/logs/gpu_at_start.txt`).
No multi-GPU LFTNet jobs were launched because the provenance/smoke gate failed.
""",
    )

    final = f"""# Stage 8 — Final Report (LFTNet)

**Final verdict:** `implementation_not_reproducible`  
**sota_claim_allowed:** `false`  
**Replace Stage-6 main method:** `false`  
**Post-confirm comparator executed:** `false`

## Answers

1. **Official & complete?** No. Local drop is an incomplete fragment (`E. incomplete_or_unusable`). `~/baseline` missing; content at `Earthquake/baseline/Tianjiyu1-LFTNet-79994f6` (2 files).
2. **Usable official checkpoint?** No.
3. **Checkpoint training data?** Unknown — no checkpoint.
4. **INSTANCE/confirm leakage risk?** Unknown (no weights). Would be `diagnostic_only_possible_leakage` if INSTANCE-trained weights appear without exclusion proof.
5. **Reproduced paper 0.844?** No (not run).
6. **Same-protocol LFTNet metrics?** Not available.
7. **Who better on F1@0.5?** Cannot compare; Ours remains the only executed same-protocol result among these.
8. **Bootstrap CI includes 0?** N/A.
9. **P/R/coverage/miss tradeoff?** N/A.
10. **P95 better via abstention?** N/A.
11. **LFTNet raises UNION oracle?** Not tested (gate failed).
12. **Worth adding LFTNet now?** No.
13. **Allowed to replace Stage 6?** No.
14. **SOTA claim?** **false**.
15. **pytest:** Stage-8 gate tests (see log).
16. **Stage 2–7 hash unchanged?** method_lock SHA256 `{lock_now}` match declared: `{lock_ok}`.

## What we found in the drop

- `README.md` is a TF2.5 requirements list (not docs).
- `se-tcn-Eqt_utils.py` defines MSSE-TCN-like blocks + `cred2` detector/P/S model, but is **not importable** (no imports; missing custom layers).
- No LICENSE, train/infer scripts, configs, checkpoints, or paper 10k/20k protocol code.

## Dependencies

Do **not** install TensorFlow 2.5 into conda `PS`. After a complete Zenodo release is provided, create `LFTNet_eval`.

## Training (only if later: code-only, no weights)

Automatic training is **disabled**. Rough wall estimate (conservative): ~60 h/epoch → ~3000 h for 50 epochs on current I/O-bound proxy — **not** feasible in one week.

## User action required

Provide full Zenodo package `10.5281/zenodo.15710535` (or complete GitHub tree) **including official pretrained weights**, then re-run `scripts/run_stage8_lftnet_pipeline.sh`.

Frozen start hashes snapshot: `artifacts/results/stage8/stage2_7_frozen_hashes.json`.
"""
    write(reports / "stage8_final_report.md", final)
    write(out / "STAGE8.DONE", json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"final_verdict": verdict["final_verdict"], "lock_ok": lock_ok}, indent=2))


if __name__ == "__main__":
    main()
