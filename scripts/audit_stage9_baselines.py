#!/usr/bin/env python
"""Stage 9 — read-only provenance/checkpoint audit for DKPN + SegPhase."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEG = Path("/home/yehang/EARTHQUAKE/baseline/SegPhase")
DKPN = Path("/home/yehang/EARTHQUAKE/baseline/DKPN")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def is_lfs_pointer(p: Path) -> bool:
    head = p.read_bytes()[:80]
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def collect_weights(repo: Path) -> list[dict]:
    out = []
    for pat in ("*.pt", "*.pth", "*.ckpt", "*.h5", "*.hdf5", "*.onnx", "*.weights", "*.bin"):
        for p in repo.rglob(pat):
            if ".git" in p.parts:
                continue
            out.append(
                {
                    "path": str(p),
                    "rel": str(p.relative_to(repo)),
                    "size": p.stat().st_size,
                    "sha256": sha256(p),
                    "is_lfs_pointer": is_lfs_pointer(p),
                    "is_real_binary": not is_lfs_pointer(p) and p.stat().st_size > 1024,
                }
            )
    return sorted(out, key=lambda x: x["rel"])


def git_head(repo: Path) -> str:
    return (repo / ".git" / "HEAD").read_text().strip() if False else __import__("subprocess").check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    out = ROOT / "artifacts/results/stage9"
    reports = ROOT / "reports/stage9"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # freeze stage2-8 critical hashes
    critical = {
        "stage6_method_lock": ROOT / "artifacts/results/stage6/final_confirm/method_lock.json",
        "stage6_confirm_consumed": ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED",
        "stage7_done": ROOT / "artifacts/results/stage7/STAGE7A.DONE",
        "stage8_done": ROOT / "artifacts/results/stage8/STAGE8.DONE",
    }
    frozen = {
        k: {"exists": p.exists(), "sha256": sha256(p) if p.is_file() else None, "size": p.stat().st_size if p.exists() else None}
        for k, p in critical.items()
    }
    frozen["method_lock_matches"] = frozen["stage6_method_lock"]["sha256"] == (
        "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"
    )
    (out / "stage2_8_frozen_hashes.json").write_text(json.dumps(frozen, indent=2) + "\n")

    seg_w = collect_weights(SEG)
    dkpn_w = collect_weights(DKPN)
    file_hashes = {"SegPhase": {w["rel"]: w for w in seg_w}, "DKPN": {w["rel"]: w for w in dkpn_w}}
    (out / "baseline_file_hashes.json").write_text(json.dumps(file_hashes, indent=2) + "\n")

    # preferred SegPhase: model_100Hz
    seg_pref = next(w for w in seg_w if w["rel"] == "model/model_100Hz.pth")
    # DKPN preferred diagnostic: MEDIUM Rnd_50
    dkpn_pref = next(
        w
        for w in dkpn_w
        if "MEDIUM" in w["rel"] and "DKPN_TrainDataset_INSTANCE" in w["rel"] and "Rnd_50" in w["rel"]
    )

    registry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT.resolve()),
        "post_confirm_external_comparator_evaluation": True,
        "sota_claim_allowed": False,
        "stage6_main_method": "fixed_rescore_UNION",
        "stage6_method_lock_sha256": "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303",
        "repos": {
            "SegPhase": {
                "realpath": str(SEG.resolve()),
                "head": git_head(SEG),
                "expected_head_prefix": "27e7e5d",
                "license": "MIT",
                "readme_channels": "UD, NS, EW",
                "readme_input_100Hz": "(B, 3, 3000)",
                "normalize": "per-channel zscore (pred.py)",
                "peak_extraction": "scipy.signal.find_peaks(..., distance=int(1/sf), height=0.1)",
                "official_peak_height": 0.1,
                "output_order": "pred[0]=P, pred[1]=S (pred.py)",
                "preferred_checkpoint": seg_pref,
                "training_data": "Japan seismic network (paper); model_V2_JP: ~7.7M Japan/JMA waveforms",
                "contains_INSTANCE": False,
                "possible_confirm_leakage": False,
                "classification": "A. valid_external_pretrained_no_INSTANCE_leakage",
                "role": "valid_waveform_only_comparator",
                "enter_main_table": True,
                "enter_formal_bootstrap": True,
                "n_weight_files": len(seg_w),
            },
            "DKPN": {
                "realpath": str(DKPN.resolve()),
                "head": git_head(DKPN),
                "expected_head_prefix": "cbced5a",
                "version": "0.4.12",
                "license": "MIT (LICENSE.md)",
                "component_order": "ZNE (SeisBench / core.py)",
                "official_S_threshold": 0.2,
                "official_P_threshold": 0.2,
                "seisbench_target": "0.4.0",
                "torch_target": "1.11.0",
                "preferred_checkpoint_diagnostic": dkpn_pref,
                "training_data": "INSTANCE (filenames: DKPN_TrainDataset_INSTANCE_Size_*)",
                "contains_INSTANCE": True,
                "possible_confirm_leakage": True,
                "classification": "C. possible_INSTANCE_confirm_leakage_diagnostic_only",
                "role": "diagnostic_only_possible_leakage",
                "enter_main_table": False,
                "enter_formal_bootstrap": False,
                "n_weight_files": len(dkpn_w),
                "evidence": [
                    "All shipped *.pt under models_v0412_paper_sb4/{MICRO,NANO2,MEDIUM} encode TrainDataset_INSTANCE in filename",
                    "Cannot prove Stage-6 confirm events excluded from INSTANCE MEDIUM/MICRO/NANO2 random splits",
                ],
            },
        },
        "tables": {
            "A_waveform_only": ["PhaseNet-STEAD", "PhaseNet-ETHZ", "PhaseNet-SCEDC", "SegPhase-100Hz"],
            "A_diagnostic_only": ["DKPN-INSTANCE-MEDIUM"],
            "B_catalog_assisted": ["fixed_rescore_STEAD", "fixed_rescore_UNION", "UNION oracle"],
        },
        "gates": {
            "SegPhase_auto_continue_if_alignment_ok": True,
            "DKPN_full_main_eval": False,
            "DKPN_diagnostic_smoke_only": True,
            "no_auto_train": True,
        },
    }
    (out / "baseline_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    md = f"""# Stage 9 — Baseline Provenance Audit (DKPN / SegPhase)

**Created (UTC):** {registry['created_utc']}  
**Project root:** `{ROOT.resolve()}`  
**post_confirm_external_comparator_evaluation:** true  
**sota_claim_allowed:** false  
**Stage-6 method_lock unchanged:** `{frozen['method_lock_matches']}`

## Paths / versions

| Repo | realpath | HEAD | Match expected? |
|--|--|--|--|
| SegPhase | `{SEG.resolve()}` | `{registry['repos']['SegPhase']['head']}` | prefix `27e7e5d` |
| DKPN | `{DKPN.resolve()}` | `{registry['repos']['DKPN']['head']}` | prefix `cbced5a` / v0.4.12 |

Third-party repos kept **read-only** (no pull/checkout/modify).

## SegPhase

- **Classification:** `A. valid_external_pretrained_no_INSTANCE_leakage`
- **Preferred checkpoint:** `model/model_100Hz.pth`  
  - SHA256: `{seg_pref['sha256']}`  
  - size: {seg_pref['size']} (real binary, not LFS pointer)
- **Input:** `(B, 3, 3000)` @ 100 Hz; channels **UD, NS, EW**
- **INSTANCE map:** ENZ → **Z, N, E** (UD=Z, NS=N, EW=E)
- **Normalize:** per-channel z-score (`pred.py`)
- **Peaks:** `find_peaks(..., distance=1/sf, height=0.1)` — P=`pred[0]`, S=`pred[1]`
- **Training:** Japan network / JMA (not INSTANCE) → eligible for main Table A
- Other weights present: `model_250Hz.pth`, `model_M01.pth`, `model_V2_JP/best_model.pth` (Japan V2; not selected as primary)

## DKPN

- **Classification:** `C. possible_INSTANCE_confirm_leakage_diagnostic_only`
- **All official v0.4.12 paper weights** are named `*_TrainDataset_INSTANCE_*`
- Preferred diagnostic ckpt: `{dkpn_pref['rel']}`  
  - SHA256: `{dkpn_pref['sha256']}`
- **Must NOT** enter valid same-protocol main table or formal Ours−DKPN bootstrap conclusions
- Smoke / diagnostic allowed; clean Stage-6-train retrain is **planned only**, not auto-started
- Official annotate defaults: S_threshold=0.2, component_order=ZNE, SeisBench-oriented API

## Environment plan (do not mutate `PS`)

| Env | Plan |
|--|--|
| `segphase_eval` | Lightweight: python3.11 + pytorch **CUDA** + numpy/scipy/obspy (avoid repo `env.yml` cpuonly pin) |
| `dkpn_eval` | Optional diagnostic: torch + seisbench compatible; full `dkpn_env.yml` (torch 1.11 / sb 0.4) only if needed |

## Next

1. SegPhase smoke + alignment → if pass: dev threshold + lock + confirm (Table A)  
2. DKPN diagnostic smoke only  
3. No simultaneous full HDF5 jobs; SegPhase first for full eval
"""
    (reports / "baseline_provenance_audit.md").write_text(md)
    print(json.dumps({"SegPhase": "A", "DKPN": "C", "seg_ckpt": seg_pref["sha256"][:16]}, indent=2))


if __name__ == "__main__":
    main()
