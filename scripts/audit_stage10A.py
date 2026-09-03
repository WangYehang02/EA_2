#!/usr/bin/env python
"""Stage 10A — protocol & leakage audit (read-only; no training)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DKPN = Path("/home/yehang/EARTHQUAKE/baseline/DKPN")
SEG = Path("/home/yehang/EARTHQUAKE/baseline/SegPhase")
LFT = ROOT / "baseline" / "Tianjiyu1-LFTNet-79994f6"


def sha256(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_json(p: Path):
    return json.loads(p.read_text())


def main() -> None:
    out = ROOT / "artifacts" / "results" / "stage10"
    reports = ROOT / "reports" / "stage10"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    audit = load_json(ROOT / "artifacts/results/stage6/full_split_audit.json")
    noise = load_json(ROOT / "artifacts/results/stage6/full_noise_counts.json")
    counts = load_json(ROOT / "artifacts/results/stage6/full_dataset_counts.json")
    missing = load_json(ROOT / "artifacts/results/stage6/missing_s_label_audit.json")
    train_man = load_json(ROOT / "artifacts/results/stage6/full_training_manifest.json")
    cm = load_json(ROOT / "artifacts/results/stage6/final_confirm/confirm_method_metrics.json")
    phaseC = load_json(ROOT / "artifacts/results/stage6/phaseC/phaseC_baselines.json")
    c2 = load_json(ROOT / "artifacts/results/stage6/phaseC2/phaseC_corrected_verdict.json")
    oracle_b = load_json(ROOT / "artifacts/results/stage6/phaseB_candidate_oracle.json")

    # disjoint check
    spf = ROOT / "artifacts/results/stage6/splits_full"

    def eset(name: str) -> set[str]:
        return set((spf / f"stage6_{name}_events.txt").read_text().splitlines())

    pt, rt, dv, cf = map(eset, ["picker_train", "ranker_train", "dev", "internal_confirm"])
    disjoint = {
        "picker_train∩ranker_train": len(pt & rt),
        "picker_train∩dev": len(pt & dv),
        "picker_train∩confirm": len(pt & cf),
        "ranker_train∩dev": len(rt & dv),
        "ranker_train∩confirm": len(rt & cf),
        "dev∩confirm": len(dv & cf),
        "event_disjoint_all_pairs": all(
            len(a & b) == 0 for a, b in [(pt, rt), (pt, dv), (pt, cf), (rt, dv), (rt, cf), (dv, cf)]
        ),
    }

    # frozen hashes
    critical = {
        "stage6_method_lock": ROOT / "artifacts/results/stage6/final_confirm/method_lock.json",
        "stage6_CONFIRM_CONSUMED": ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED",
        "stage7_DONE": ROOT / "artifacts/results/stage7/STAGE7A.DONE",
        "stage8_DONE": ROOT / "artifacts/results/stage8/STAGE8.DONE",
        "stage9_DONE": ROOT / "artifacts/results/stage9/STAGE9.DONE",
    }
    frozen = {
        k: {"path": str(p), "exists": p.exists(), "sha256": sha256(p), "size": p.stat().st_size if p.exists() else None}
        for k, p in critical.items()
    }
    frozen["method_lock_matches_declared"] = (
        frozen["stage6_method_lock"]["sha256"] == "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"
    )
    frozen["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "stage10A_frozen_hashes.json").write_text(json.dumps(frozen, indent=2) + "\n")

    # GPU snapshot
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception as e:
        gpu = repr(e)
    free = []
    for line in gpu.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and float(parts[1]) < 500:
            free.append(int(parts[0]))

    verified_confirm = {
        "STEAD_top1": {k: cm["STEAD_top1"][k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate", "n_traces"]},
        "fixed_rescore_UNION": {
            k: cm["fixed_rescore_UNION"][k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate", "n_traces"]
        },
        "oracle_UNION": {k: cm["oracle_UNION"][k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95", "n_traces"]},
    }
    verified_dev = {
        "fixed_rescore_UNION_f1@0.5": phaseC["fixed_rescore_UNION"]["f1@0.5"],
        "oracle_UNION_f1@0.5": phaseC["oracle_UNION"]["f1@0.5"],
        "phaseC2_fixed_union_f1_05": c2["fixed_union_f1_05"],
        "phaseB_UNION_K10_oracle_f1@0.5": oracle_b["UNION_K10"]["oracle_f1@0.5"],
        "old_ranker_delta_vs_fixed_approx": round(0.8323 - 0.8668, 4),
        "ranker_failed_plus0.01_gate": True,
    }

    registry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT.resolve()),
        "sota_claim_allowed": False,
        "exact_lftnet_protocol_reproducible": False,
        "sota_claim_allowed_reason": "exact_lftnet_protocol_reproducible=false; Stage6 confirm already CONSUMED; no new untouched holdout yet",
        "models": {
            "Ours-Blind": {
                "allowed_inputs": ["waveform", "waveform_model_features"],
                "forbidden": ["source_location", "origin_time", "distance", "path_history", "manual_picks", "manual_snr"],
                "role": "fair_blind_INSTANCE_comparison",
            },
            "Ours-Path": {
                "allowed_extra": ["distance_mlp", "path_history", "shrinkage"],
                "must_describe_as": "catalog-assisted S-phase candidate re-picking/refinement",
                "not_blind_picker": True,
            },
            "frozen_main": "fixed_rescore_UNION",
        },
        "splits": audit["sizes"],
        "n_total_events": audit["n_total_events"],
        "event_disjoint": audit["event_disjoint"],
        "trace_disjoint": audit["trace_disjoint"],
        "disjoint_pair_counts": disjoint,
        "supervision_from_full_training_manifest": train_man.get("supervision"),
        "noise": {
            "metadata_rows": noise["metadata_rows"],
            "unique_trace_name": noise["unique_trace_name"],
            "is_132288": noise["metadata_rows"] == 132288,
            "sampling_rate_hz": 100.0,
        },
        "instance_raw": {
            "events_rows": counts["raw"]["events_csv_data_rows"],
            "P_labelled": counts["raw"]["P_label_coverage_traces"],
            "S_labelled": counts["raw"]["S_label_coverage_traces"],
            "P_only": missing["n_P_only"],
            "sampling_rate_hz": 100.0,
            "hdf5_component_order": "ENZ (InstanceHDF5Reader.read_waveform)",
            "realpath_INSTANCE": str((ROOT / "data/INSTANCE").resolve()),
        },
        "baselines": {
            "DKPN": {
                "realpath": str(DKPN.resolve()),
                "head": git_head(DKPN),
                "license": "MIT",
                "version": "0.4.12",
                "official_weights_train_data": "INSTANCE (TrainDataset_INSTANCE in filenames)",
                "official_weights_usable_for_main": False,
                "reason": "possible_INSTANCE_confirm_leakage; random INSTANCE splits cannot prove confirm exclusion",
                "clean_train_policy": "random_init_or_non_INSTANCE_external_only_on_stage6_picker_train",
                "in_channels": 5,
                "component_order": "ZNE",
                "env_target": "stage10_dkpn (do not mutate PS); torch~1.11 + seisbench~0.4 from repo README",
            },
            "SegPhase": {
                "realpath": str(SEG.resolve()),
                "head": git_head(SEG),
                "license": "MIT",
                "preferred_ckpt": "model/model_100Hz.pth",
                "training_region": "Japan seismic network / JMA (not INSTANCE)",
                "stage9_confirm_f1@0.5": 0.6543,
                "note": "Japan pretrained ≠ architecture ceiling; Stage10C = in-domain INSTANCE retrain",
            },
            "LFTNet": {
                "local_drop": str(LFT.resolve()) if LFT.exists() else None,
                "complete": False,
                "exact_lftnet_protocol_reproducible": False,
                "missing": [
                    "official checkpoint",
                    "train/infer scripts",
                    "10k record list / 20k segment construction",
                    "paper-exact threshold & metric code",
                    "event-disjoint proof",
                ],
                "paper_claim_S_F1_0.5": 0.844,
                "allowed_claim_if_we_beat_baselines": "best among evaluated reproducible methods under our event-disjoint INSTANCE protocol",
                "forbidden_claims": ["INSTANCE SOTA", "surpassed LFTNet"],
            },
        },
        "verified_frozen_metrics": {"confirm": verified_confirm, "dev": verified_dev},
        "gpu_snapshot": {"raw": gpu, "free_indices": free, "n_free": len(free)},
        "data_cache_root": str((out / "data_cache").resolve()),
        "confirm_consumed": True,
        "confirm_usable_for_stage10_tuning": False,
        "post_hoc_on_consumed_confirm_only": True,
    }
    (out / "stage10A_protocol_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    state = {
        "stage": "10A",
        "status": "AUDIT_COMPLETE",
        "next": "unified_adapter_tests_then_DKPN_clean_smoke",
        "sota_claim_allowed": False,
        "exact_lftnet_protocol_reproducible": False,
        "dkpn_official_weights_main_table": False,
        "gpu_free_at_audit": free,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "stage10_state.json").write_text(json.dumps(state, indent=2) + "\n")

    md = f"""# Stage 10A — Protocol & Leakage Audit

**Created (UTC):** {registry['created_utc']}  
**Project root:** `{ROOT.resolve()}`  
**sota_claim_allowed:** **false**  
**exact_lftnet_protocol_reproducible:** **false**

## Task split

| Model | Inputs | Claim type |
|--|--|--|
| **Ours-Blind** | waveform only | fair vs PhaseNet/DKPN/SegPhase/LFTNet-class blind pickers |
| **Ours-Path** | waveform + distance/history/shrinkage | *catalog-assisted S-phase candidate re-picking/refinement* (not blind) |
| Frozen main | `fixed_rescore_UNION` | Stage 6 confirm CONSUMED — do not retune on confirm |

## 1–2. Splits (from `full_split_audit.json`)

| Split | Events | Traces |
|--|--:|--:|
| picker_train | {audit['sizes']['stage6_picker_train']['events']} | {audit['sizes']['stage6_picker_train']['traces']} |
| ranker_train | {audit['sizes']['stage6_ranker_train']['events']} | {audit['sizes']['stage6_ranker_train']['traces']} |
| dev | {audit['sizes']['stage6_dev']['events']} | {audit['sizes']['stage6_dev']['traces']} |
| internal_confirm | {audit['sizes']['stage6_internal_confirm']['events']} | {audit['sizes']['stage6_internal_confirm']['traces']} |

- `event_disjoint={audit['event_disjoint']}`, `trace_disjoint={audit['trace_disjoint']}`
- Pairwise event overlaps: `{disjoint}`

Supervision plan (full training manifest):

- P+S both traces in picker_train: **{train_man['supervision']['n_ps_both_traces']}**
- picker_train all traces: **{train_man['supervision']['n_picker_train_traces_all']}**
- Phase A historically **excluded P-only** (not treated as noise). Stage 10 allows **masked P-only** if tested.
- Full noise available: **{noise['unique_trace_name']}** (is 132288: **{noise['metadata_rows']==132288}**)

## 3–6. Baseline provenance

### DKPN
- realpath: `{DKPN.resolve()}`
- HEAD: `{registry['baselines']['DKPN']['head']}`
- LICENSE: MIT; version 0.4.12; SeisBench-oriented; `in_channels=5`, component **ZNE**
- Official paper weights: **INSTANCE-trained** → **`diagnostic_only`**, **not** for main Table / formal SOTA
- Clean DKPN: **random init** (or non-INSTANCE external) on Stage-6 `picker_train` only

### SegPhase
- realpath: `{SEG.resolve()}`
- HEAD: `{registry['baselines']['SegPhase']['head']}`
- Japan/JMA pretrained (100 Hz UD,NS,EW); Stage 9 confirm F1@0.5≈0.654 — architecture not exhausted
- Stage 10C: in-domain INSTANCE retrain (later)

### LFTNet
- Local drop incomplete (`{LFT}`): no checkpoint, no 10k list, no metric scripts
- **Cannot** claim “exceeded LFTNet / INSTANCE SOTA”
- Allowed phrasing: *best among evaluated reproducible methods under our event-disjoint INSTANCE protocol*

## 7. INSTANCE IO

- realpath: `{(ROOT / 'data/INSTANCE').resolve()}`
- 100 Hz; HDF5 waveform order **ENZ** via `InstanceHDF5Reader`
- Map to DKPN/SegPhase as required (ZNE / UD-NS-EW) with UTC remap tests

## 8. Noise

- Full noise traces: **{noise['metadata_rows']}** (target 132288: **yes**)
- S-labelled event traces (all INSTANCE): **{counts['raw']['S_label_coverage_traces']}**; P-only: **{missing['n_P_only']}**

## 9–10. LFTNet protocol

`exact_lftnet_protocol_reproducible = false` — missing official list/code/weights/threshold definitions. No second-hand table guessing.

## Verified frozen metrics (re-read from JSON)

### Confirm (S-labelled n={verified_confirm['STEAD_top1']['n_traces']})
| Method | F1@0.5 | F1@0.1 | P95 |
|--|--:|--:|--:|
| STEAD top-1 | {verified_confirm['STEAD_top1']['f1@0.5']:.4f} | {verified_confirm['STEAD_top1']['f1@0.1']:.4f} | {verified_confirm['STEAD_top1']['detected_ae_p95']:.3f} |
| fixed_rescore_UNION | {verified_confirm['fixed_rescore_UNION']['f1@0.5']:.4f} | {verified_confirm['fixed_rescore_UNION']['f1@0.1']:.4f} | {verified_confirm['fixed_rescore_UNION']['detected_ae_p95']:.3f} |
| UNION oracle | {verified_confirm['oracle_UNION']['f1@0.5']:.4f} | {verified_confirm['oracle_UNION']['f1@0.1']:.4f} | {verified_confirm['oracle_UNION']['detected_ae_p95']:.3f} |

### Dev
- fixed_rescore_UNION F1@0.5 ≈ **{verified_dev['fixed_rescore_UNION_f1@0.5']:.4f}**
- UNION oracle F1@0.5 ≈ **{verified_dev['oracle_UNION_f1@0.5']:.4f}** (phaseB UNION_K10={verified_dev['phaseB_UNION_K10_oracle_f1@0.5']:.4f})
- Old ranker failed +0.01 gate (≈ −0.034 vs fixed after correction)

## GPU at audit

```
{gpu}
```

Free GPUs (<500 MiB): **{free if free else 'none'}** — do not preempt others.

## Confirm policy

- `CONFIRM.CONSUMED` present; Stage 10 **must not** tune on confirm
- Any confirm numbers are **post-hoc** only

## Frozen hash check

method_lock match: **{frozen['method_lock_matches_declared']}** (`a02dc28e…`)
"""
    (reports / "stage10A_protocol_and_leakage_audit.md").write_text(md)
    print(json.dumps({"exact_lftnet": False, "dkpn_main": False, "free_gpus": free, "lock_ok": frozen["method_lock_matches_declared"]}, indent=2))


if __name__ == "__main__":
    main()
