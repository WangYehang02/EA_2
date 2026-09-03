#!/usr/bin/env python
"""Lock Stage-6 final method = fixed_rescore_UNION (before confirm)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.full_splits import full_stage6_paths, load_full_event_ids, load_full_trace_names
from earthquake.stage6.phaseB import CAND_EXTRACT_CFG, UNION_DEDUP_S, sha256_file
from earthquake.stage6.ranker.schema import CANDIDATE_SCHEMA, schema_sha256
from earthquake.utils import ensure_dir


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc}"


def _large_file_meta(path: Path) -> dict:
    st = path.stat() if path.exists() else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": st.st_size if st else None,
        "mtime": st.st_mtime if st else None,
        "sha256": None,
        "note": "large HDF5 — path/size/mtime only",
    }


def main() -> None:
    assert (artifacts_dir() / "results" / "stage6" / "phaseC2" / "PHASEC2.DONE").exists()
    seal = full_stage6_paths()["confirm_seal"]
    assert seal.exists() and json.loads(seal.read_text()).get("status") == "SEALED"

    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "final_confirm")
    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    phaseb_man = load_json(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.json")
    full_audit = load_json(artifacts_dir() / "results" / "stage6" / "full_split_audit.json")
    ida_val = load_json(artifacts_dir() / "results" / "stage6" / "phaseB_ida_checkpoint_validation.json")
    corrected = load_json(artifacts_dir() / "results" / "stage6" / "phaseC2" / "phaseC_corrected_verdict.json")

    ida_ckpt = artifacts_dir() / "models" / "stage6" / "phasenet_ida_full_seed42" / "checkpoints" / "best.pt"
    mlp = artifacts_dir() / "models" / "stage6" / "history_picker_train" / "travel_time_baseline_mlp.pkl"
    hist_feat = artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"
    store = artifacts_dir() / "models" / "stage6" / "history_picker_train" / "temporal_history_store.pkl"

    import seisbench
    import torch
    import numpy
    import scipy
    import pandas as pd_mod

    source_files = [
        ROOT / "src/earthquake/stage6/keyed_align.py",
        ROOT / "src/earthquake/gating/cache_io.py",
        ROOT / "src/earthquake/metrics.py",
        ROOT / "src/earthquake/fusion/peak_candidates.py",
        ROOT / "src/earthquake/models/seisbench_reference.py",
        ROOT / "src/earthquake/stage6/ranker/union_schema.py",
        ROOT / "src/earthquake/stage6/phaseB.py",
        ROOT / "scripts/evaluate_stage6_candidate_ranker.py",
        ROOT / "scripts/cache_stage6_phaseB_candidates.py",
        ROOT / "scripts/run_phaseC_baselines_fast.py",
    ]
    registry_rows = []
    for p in source_files:
        registry_rows.append({"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p) if p.exists() else None, "kind": "source"})

    small_arts = {
        "phaseB_eval_manifest_csv": artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv",
        "history_manifest": artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json",
        "ida_best_pt": ida_ckpt,
        "mlp_pkl": mlp,
        "hist_features": hist_feat,
        "phaseC2_verdict": artifacts_dir() / "results" / "stage6" / "phaseC2" / "phaseC_corrected_verdict.json",
        "confirm_events": artifacts_dir() / "results" / "stage6" / "splits_full" / "stage6_internal_confirm_events.txt",
        "confirm_traces": artifacts_dir() / "results" / "stage6" / "splits_full" / "stage6_internal_confirm_traces.txt",
        "CONFIRM_SEALED": seal,
    }
    for k, p in small_arts.items():
        registry_rows.append({"path": str(p), "sha256": sha256_file(p) if p.exists() else None, "kind": k})

    conf_ev = load_full_event_ids("stage6_internal_confirm")
    conf_tr = load_full_trace_names("stage6_internal_confirm")

    lock = {
        "method": "fixed_rescore_UNION",
        "locked_before_confirm": True,
        "ranker_used": False,
        "forced_choice_used": False,
        "hyperparameter_search_on_confirm": False,
        "confirm_seen_before_lock": False,
        "catalog_assisted": True,
        "blind_picker": False,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "phaseC_corrected_verdict": corrected.get("verdict"),
        "candidate_sources": {
            "stead_weight": "stead",
            "ida_ckpt": str(ida_ckpt),
            "ida_epoch": ida_val.get("epoch", 14),
            "ida_sha256": sha256_file(ida_ckpt),
            "stead_k": 5,
            "ida_k": 5,
            "union_k_max": 10,
            "dedup_s": UNION_DEDUP_S,
            "representative_time": "STEAD_preferred_when_both_support",
            "probability_combination": "never_average; max available only for heuristic baseline (not used for fixed_rescore)",
            "peak_extract": dict(CAND_EXTRACT_CFG),
            "component_order": "ENZ_HDF5_to_ZNE_SeisBench",
            "label_order": "PSN",
            "sampling_rate_hz": 100.0,
            "utc_remap": "seisbench_reference.remap_trace_to_waveform_grid",
            "schema_version": CANDIDATE_SCHEMA["version"],
            "schema_sha256": schema_sha256(),
            "no_candidate_behavior": "nan_prediction",
        },
        "fixed_rescore": {
            "lambdas_s": hist_man["lambdas_s_frozen"],
            "shrinkage_k": hist_man["shrinkage_k"],
            "min_history": hist_man["min_history"],
            "grid_size": hist_man["grid_size"],
            "mad_disable_s": 1.0,
            "min_sigma_s": 0.05,
            "max_sigma_s": 1.0,
            "global_residual": hist_man["global_residual"],
            "distance_mlp_path": str(mlp),
            "distance_mlp_sha256": sha256_file(mlp),
            "history_features_path": str(hist_feat),
            "history_features_sha256": hist_man.get("features_sha256") or sha256_file(hist_feat),
            "history_store_path": str(store),
            "history_store_sha256": sha256_file(store),
            "history_protocol": hist_man["protocol"],
            "history_fit_split": "stage6_picker_train_only",
            "history_query_splits_pre_confirm": ["stage6_ranker_train", "stage6_dev"],
            "confirm_history_query": "past_only_from_frozen_picker_train_store_never_update",
        },
        "data_eval": {
            "picker_train_events": len(load_full_event_ids("stage6_picker_train")),
            "ranker_train_events": len(load_full_event_ids("stage6_ranker_train")),
            "dev_events": len(load_full_event_ids("stage6_dev")),
            "confirm_events": len(conf_ev),
            "confirm_traces": len(conf_tr),
            "confirm_events_sha256": full_audit["manifests"]["stage6_internal_confirm"]["events_sha256"],
            "confirm_traces_sha256": full_audit["manifests"]["stage6_internal_confirm"]["traces_sha256"],
            "dev_eval_manifest_sha256": phaseb_man.get("csv_sha256"),
            "match_windows_s": [0.1, 0.2, 0.5],
            "bootstrap_reps": 5000,
            "bootstrap_seed": 20260817,
            "metrics_module_sha256": sha256_file(ROOT / "src/earthquake/metrics.py"),
            "keyed_align_sha256": sha256_file(ROOT / "src/earthquake/stage6/keyed_align.py"),
            "detected_ae_p95_definition": "finite_pred_and_label_only_excludes_none_miss",
        },
        "large_files": {
            "events_hdf5": _large_file_meta(Path("/data/mnt_data/INSTANCE/events/Instance_events_counts.hdf5") if Path("/data/mnt_data/INSTANCE/events/Instance_events_counts.hdf5").exists() else artifacts_dir().parent),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seisbench": getattr(seisbench, "__version__", "unknown"),
            "numpy": numpy.__version__,
            "pandas": pd_mod.__version__,
            "scipy": scipy.__version__,
            "git_commit": _run(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "git_diff_hash": hashlib.sha256(_run(["git", "diff"]).encode()).hexdigest(),
            "conda_explicit_hash": hashlib.sha256(_run(["conda", "list", "--explicit"]).encode()).hexdigest(),
            "gpu": _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        },
    }

    # resolve INSTANCE path properly
    from earthquake.config import resolve_instance_root

    lock["large_files"]["events_hdf5"] = _large_file_meta(resolve_instance_root() / "events" / "Instance_events_counts.hdf5")
    lock["large_files"]["noise_hdf5"] = _large_file_meta(resolve_instance_root() / "noise" / "Instance_noise.hdf5")

    lock_path = out / "method_lock.json"
    # also write stage6 method_lock expected by full_splits
    save_json(lock, lock_path)
    digest = sha256_file(lock_path)
    (out / "method_lock.sha256").write_text(digest + "\n")
    # Canonical Stage-6 gate file
    save_json(lock, artifacts_dir() / "results" / "stage6" / "method_lock_stage6.json")

    pd.DataFrame(registry_rows).to_csv(out / "source_hash_registry.csv", index=False)
    env_txt = "\n".join(f"{k}={v}" for k, v in lock["environment"].items()) + "\n"
    (out / "environment_lock.txt").write_text(env_txt)

    # verify lock unchanged
    assert sha256_file(lock_path) == digest

    md = f"""# Stage 6 Final Method Lock

**Method:** `fixed_rescore_UNION`  
**Locked before confirm:** true  
**Ranker used on confirm:** false  
**method_lock.sha256:** `{digest}`

Catalog-assisted S-phase candidate re-picking/refinement. Not a blind picker. SOTA claims not allowed from this lock alone.
"""
    (ROOT / "reports/stage6/stage6_final_method_lock.md").write_text(md)
    print(json.dumps({"method_lock_sha256": digest, "path": str(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
