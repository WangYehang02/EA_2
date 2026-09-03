#!/usr/bin/env python
"""Finalize Phase C: ablations, full-dev eval, bootstrap, verdict. Stops without unlock."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.utils import ensure_dir


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_finalize")
        raise SystemExit("method_lock")
    except RuntimeError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-ckpt", required=True)
    parser.add_argument("--baselines-json", default="artifacts/results/stage6/phaseC/phaseC_baselines.json")
    parser.add_argument("--ranker-metrics", default="artifacts/results/stage6/phaseC/ranker_dev_metrics.json")
    parser.add_argument("--bootstrap-json", default="artifacts/results/stage6/phaseC/phaseC_bootstrap.json")
    parser.add_argument("--ablation-json", default="artifacts/results/stage6/phaseC/phaseC_ablations.json")
    args = parser.parse_args()
    _guard()
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC")

    baselines = load_json(ROOT / args.baselines_json)
    ranker = load_json(ROOT / args.ranker_metrics)
    boot = load_json(ROOT / args.bootstrap_json) if (ROOT / args.bootstrap_json).exists() else {}
    abl = load_json(ROOT / args.ablation_json) if (ROOT / args.ablation_json).exists() else {}

    # strongest non-learning baseline by F1@0.5
    nonlearn = {k: v for k, v in baselines.items() if k != "oracle_UNION"}
    strongest_name = max(nonlearn, key=lambda k: nonlearn[k].get("f1@0.5", -1))
    strong = nonlearn[strongest_name]
    oracle_f1 = float(baselines.get("oracle_UNION", {}).get("f1@0.5", 0.8917))

    f05 = float(ranker["f1@0.5"])
    f01 = float(ranker["f1@0.1"])
    p95 = float(ranker["detected_ae_p95"])
    d05 = f05 - float(strong["f1@0.5"])
    d01 = f01 - float(strong["f1@0.1"])
    dp95 = p95 - float(strong.get("detected_ae_p95", p95))
    denom = oracle_f1 - float(strong["f1@0.5"])
    gap = (f05 - float(strong["f1@0.5"])) / denom if abs(denom) > 1e-9 else float("nan")

    boot_ci = None
    direction_stable = False
    key = f"ranker_vs_{strongest_name}"
    if key in boot and "f1@0.5" in boot[key]:
        boot_ci = boot[key]["f1@0.5"].get("ci95")
        md = boot[key]["f1@0.5"].get("mean_delta", 0)
        if boot_ci:
            direction_stable = (md >= 0 and boot_ci[0] >= 0) or (md <= 0 and boot_ci[1] <= 0)

    history_beats = None
    if "R2" in abl and "R2_shuffled_history" in abl:
        history_beats = abl["R2"]["f1@0.5"] > abl["R2_shuffled_history"]["f1@0.5"] + 0.002

    seed_consistent = abl.get("seed_consistent")

    # stop-gate
    if (
        d05 >= 0.01
        and direction_stable
        and d01 >= -0.003
        and dp95 <= 0.5  # not clearly worse
        and seed_consistent is not False
    ):
        verdict = "strong_pass"
        multi = True
    elif (0.005 <= d05 < 0.01 or dp95 <= -0.15) and direction_stable:
        verdict = "marginal_pass"
        multi = False
    else:
        verdict = "fail"
        multi = False

    blob = torch.load(args.best_ckpt, map_location="cpu", weights_only=False)
    final = {
        "candidate_source": "UNION_STEAD5_IDA5",
        "candidate_k_max": 10,
        "ranker_train_events": 8058,
        "ranker_train_traces": 206413,
        "dev_events": 5341,
        "dev_s_traces": 87293,
        "strongest_baseline": strongest_name,
        "strongest_baseline_f1_01": strong.get("f1@0.1"),
        "strongest_baseline_f1_05": strong.get("f1@0.5"),
        "best_ranker": blob.get("variant"),
        "best_ranker_params": {
            "seed": blob.get("seed"),
            "beta": blob.get("beta"),
            "gamma": blob.get("gamma"),
            "n_params": blob.get("n_params"),
            "ckpt": str(args.best_ckpt),
        },
        "best_ranker_f1_01": f01,
        "best_ranker_f1_05": f05,
        "best_ranker_detected_ae_p95": p95,
        "delta_f1_01": d01,
        "delta_f1_05": d05,
        "delta_p95": dp95,
        "bootstrap_ci_f1_05": boot_ci,
        "oracle_f1_05": oracle_f1,
        "oracle_gap_recovered": gap,
        "ida_only_recovery_rate": abl.get("ida_only_recovery_rate"),
        "history_beats_shuffled": history_beats,
        "seed_consistent": seed_consistent,
        "verdict": verdict,
        "multistation_may_start": bool(multi and verdict == "strong_pass"),
        "confirm_remains_sealed": True,
    }
    save_json(final, out / "phaseC_final_verdict.json")

    md = f"""# Stage 6 Phase C — Candidate Ranker Report

**Verdict:** `{verdict}`  
**Best ranker:** `{final['best_ranker']}` {final['best_ranker_params']}  
**Strongest baseline:** `{strongest_name}` F1@0.5={strong.get('f1@0.5'):.4f}  
**Ranker F1@0.5 / @0.1:** {f05:.4f} / {f01:.4f}  
**ΔF1@0.5 / @0.1:** {d05:+.4f} / {d01:+.4f}  
**detected_ae_p95:** {p95:.3f} (Δ {dp95:+.3f})  
**Oracle gap recovered:** {gap:.3f} (oracle={oracle_f1:.4f})  
**Multistation may start:** `{final['multistation_may_start']}`  
**Confirm sealed:** `true`

## Notes

- Candidate source frozen: UNION_STEAD5_IDA5 (Phase B schema).
- History: picker_train-only snapshot; shrinkage_k=50 frozen.
- Selected pick always from union candidates or none_of_k.

## Artifacts

- `artifacts/results/stage6/phaseC/phaseC_final_verdict.json`
- `artifacts/results/stage6/phaseC/phaseC_baselines.json`
- `artifacts/results/stage6/phaseC/phaseC_bootstrap.json`
"""
    (ROOT / "reports/stage6/phaseC_candidate_ranker_report.md").write_text(md)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
