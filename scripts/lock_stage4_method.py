#!/usr/bin/env python
"""Freeze Stage-4 method lock BEFORE any holdout inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, save_json
from earthquake.utils import ensure_dir


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/confirmatory.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage4")

    best = load_json(artifacts_dir() / "results" / "stage2" / "best_lambdas.json")
    rh = load_json(artifacts_dir() / "results" / "stage2" / "residual_history_meta.json")
    tt = load_json(artifacts_dir() / "results" / "stage2" / "travel_time_baseline_selection.json")

    locked_files = [
        "configs/fusion_fixed.yaml",
        "configs/gate_catalog.yaml",
        "configs/stage4/confirmatory.yaml",
        "artifacts/results/stage2/best_lambdas.json",
        "artifacts/results/stage2/residual_history_meta.json",
        "artifacts/results/stage2/travel_time_baseline_selection.json",
        "artifacts/history/residual_history_features_frozen.parquet",
        "artifacts/history/travel_time_baseline.pkl",
        "src/earthquake/fusion/candidate_rescorer.py",
        "src/earthquake/fusion/peak_candidates.py",
        "src/earthquake/history/residual_prior.py",
        "src/earthquake/models/seisbench_reference.py",
        "src/earthquake/picking.py",
        "src/earthquake/metrics.py",
    ]
    for seed in cfg.get("gate_seeds", [42, 123, 2026]):
        locked_files.append(f"artifacts/models/learned_gate/scalar_gate_seed{seed}/best.pt")

    file_hashes = {}
    for rel in locked_files:
        p = ROOT / rel
        file_hashes[rel] = {"sha256": sha256_file(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}

    # git (optional)
    git_info = {"is_repo": False, "commit": None, "status": None}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            git_info["is_repo"] = True
            git_info["commit"] = r.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            git_info["status"] = st.stdout
        else:
            git_info["note"] = "not a git repository; recorded workspace status via file hashes only"
    except Exception as e:
        git_info["error"] = str(e)

    versions = {}
    for mod in ("torch", "numpy", "pandas", "seisbench", "scipy", "sklearn", "zarr"):
        try:
            m = __import__(mod if mod != "sklearn" else "sklearn")
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    versions["python"] = sys.version

    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze catalog-assisted S-phase candidate re-picking/refinement BEFORE Stage-4 holdout inference.",
        "method_name": "fixed_catalog_rescore",
        "method_role": "catalog-assisted S-phase candidate re-picking/refinement",
        "not_a_blind_detector": True,
        "phasenet": {
            "weight": "stead",
            "pick_threshold": float(cfg["pick_threshold"]),
            "utc_remap": True,
            "annotate_reference": "SeisBenchPhaseNetReference.predict_row(remap_to_waveform=True)",
        },
        "peak_detection": {
            "candidate_k": int(cfg["candidate_k"]),
            "min_peak_distance": int(cfg["min_peak_distance"]),
            "min_peak_prominence": float(cfg["min_peak_prominence"]),
            "min_peak_probability": float(cfg["min_peak_probability"]),
        },
        "travel_time_baseline": {
            "selected_kind": tt.get("selected_kind"),
            "artifact": "artifacts/history/travel_time_baseline.pkl",
            "selection_note": tt.get("note"),
        },
        "residual_history": {
            "artifact": "artifacts/history/residual_history_features_frozen.parquet",
            "shrinkage_k": float(rh.get("best_shrinkage_k", 50.0)),
            "min_history": int(cfg["min_history"]),
            "mad_disable_s": float(cfg["mad_disable_s"]),
            "min_sigma_s": float(cfg["min_sigma_s"]),
            "max_sigma_s": float(cfg["max_sigma_s"]),
            "global_residual": rh.get("global_residual"),
            "protocol": "frozen temporal store / train-only past events",
        },
        "fixed_rescore_lambdas": {
            "source": "artifacts/results/stage2/best_lambdas.json",
            "selected_on": "Stage-2 chronological VAL subset of the mixed fixed-eval set (NOT Stage-3 test-only, NOT Stage-4 holdout)",
            "selected_using_stage3_test": False,
            "best_p": best["best_p"],
            "best_s": best["best_s"],
            "shrinkage_k": best.get("shrinkage_k", 50.0),
            "config_override": {
                "lambda_p": cfg.get("lambda_p"),
                "lambda_s": cfg.get("lambda_s"),
            },
        },
        "hyperparameter_provenance_disclosure": {
            "shrinkage_k": "Selected on train residual MAE grid in Stage 2 (residual_history_meta); frozen at 50.",
            "lambdas_lw_lh_lp": "Selected by grid search on Stage-2 VAL traces within mixed fixed_eval (val split), maximizing S/P F1@0.1 then F1@0.5 then -matched P95. Not tuned on Stage-3 test-only or Stage-4 holdout.",
            "stage3_test_viewed": True,
            "stage3_test_used_for_param_search": False,
            "warning": "Stage-3 test metrics were inspected for method selection (gate vs fixed). Stage-4 holdout must remain unused for any further tuning.",
        },
        "learned_gate_ablation_only": {
            "seeds": list(cfg.get("gate_seeds", [42, 123, 2026])),
            "checkpoints": [f"artifacts/models/learned_gate/scalar_gate_seed{s}/best.pt" for s in cfg.get("gate_seeds", [])],
            "not_main_method": True,
        },
        "sampling_rate_hz_nominal": 100.0,
        "file_hashes": file_hashes,
        "git": git_info,
        "versions": versions,
        "platform": {"system": platform.platform(), "machine": platform.machine()},
        "forbidden_on_holdout": [
            "retune alpha/lambdas/thresholds/K/shrinkage",
            "retrain gate",
            "GNN",
            "overwrite stage1-3 artifacts",
        ],
    }
    path = out / "method_lock.json"
    save_json(lock, path)
    # also a compact hash of the lock content itself
    lock_bytes = json.dumps(lock, sort_keys=True).encode()
    save_json({"method_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "path": str(path)}, out / "method_lock_meta.json")
    print({"saved": str(path), "method_lock_sha256": hashlib.sha256(lock_bytes).hexdigest()}, flush=True)


if __name__ == "__main__":
    main()
