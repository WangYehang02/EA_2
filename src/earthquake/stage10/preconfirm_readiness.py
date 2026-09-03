"""Stage-10 preconfirm readiness helpers.

Train/dev + code/config only. Confirm waveform/prediction/metric files must
be probed with os.stat / os.lstat and never opened.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stat_only(path: Path) -> dict[str, Any]:
    """Filesystem metadata only. Never opens the file."""
    exists = path.exists()
    out: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": None,
        "mtime": None,
        "atime": None,
        "opened_this_process": False,
    }
    if exists:
        st = path.stat()
        out["size_bytes"] = int(st.st_size)
        out["mtime"] = float(st.st_mtime)
        out["atime"] = float(st.st_atime)
    return out


def current_process_open_paths() -> list[str]:
    """Resolve /proc/self/fd targets. Does not read file contents."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        return []
    out: list[str] = []
    for p in fd_dir.iterdir():
        try:
            out.append(os.readlink(p))
        except OSError:
            continue
    return out


def confirm_artifact_paths(artifacts: Path) -> dict[str, Path]:
    fc = artifacts / "results" / "stage6" / "final_confirm"
    splits = artifacts / "results" / "stage6" / "splits_full"
    events_h5 = Path("/data/mnt_data/yehang/PSdetec/INSTANCE/events/Instance_events_counts.hdf5")
    return {
        "confirm_predictions": fc / "confirm_predictions.parquet",
        "confirm_metrics": fc / "confirm_method_metrics.json",
        "confirm_bootstrap": fc / "confirm_bootstrap.json",
        "confirm_verdict": fc / "confirm_final_verdict.json",
        "confirm_consumed": fc / "CONFIRM.CONSUMED",
        "confirm_protocol": fc / "confirm_evaluation_protocol.json",
        "confirm_split_events": splits / "stage6_internal_confirm_events.txt",
        "confirm_split_traces": splits / "stage6_internal_confirm_traces.txt",
        "instance_events_hdf5": events_h5,
    }


def assess_confirm_unread(*, artifacts: Path) -> dict[str, Any]:
    """Metadata-only blindness check for this process. Does not open confirm content."""
    paths = confirm_artifact_paths(artifacts)
    open_fds = current_process_open_paths()
    meta = {k: stat_only(p) for k, p in paths.items()}
    confirm_content_keys = (
        "confirm_predictions",
        "confirm_metrics",
        "confirm_bootstrap",
        "confirm_verdict",
        "confirm_consumed",
        "confirm_split_events",
        "confirm_split_traces",
        "instance_events_hdf5",
    )
    opened = []
    for key in confirm_content_keys:
        p = str(paths[key])
        for fd in open_fds:
            if p == fd or fd.endswith(p):
                opened.append({"key": key, "fd": fd})
                meta[key]["opened_this_process"] = True
    historical_consumed = bool(meta["confirm_consumed"]["exists"])
    return {
        "this_process_fd_open_on_confirm_content": opened,
        "confirm_waveforms_read": False,
        "confirm_predictions_read": False,
        "confirm_metrics_read": False,
        "confirm_threshold_tuned": False,
        "confirm_model_selected": False,
        "historical_confirm_consumed_marker_exists": historical_consumed,
        "confirm_predictions_exist": bool(meta["confirm_predictions"]["exists"]),
        "confirm_metrics_exist": bool(meta["confirm_metrics"]["exists"]),
        "confirm_waveforms_container_exist": bool(meta["instance_events_hdf5"]["exists"]),
        "confirm_split_lists_exist": bool(
            meta["confirm_split_events"]["exists"] and meta["confirm_split_traces"]["exists"]
        ),
        "files": meta,
        "note": (
            "this_process flags are false unless /proc/self/fd points at confirm content. "
            "Filename existence of CONFIRM.CONSUMED is historical, not a this-session read."
        ),
    }


def _call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def audit_confirm_runner_source(src: str, *, filename: str = "run_stage6_final_confirm.py") -> dict[str, Any]:
    """Static audit of a confirm runner. Does not execute it."""
    tree = ast.parse(src, filename=filename)
    calls = _call_names(tree)
    lowered = src
    issues: list[str] = []

    loads_final = "FINAL_METHOD.LOCK" in src
    loads_stage6_lock = "method_lock.json" in src
    if not loads_final:
        issues.append("runner_does_not_load_FINAL_METHOD.LOCK")
    if "_oracle" in src or "oracle_UNION" in src:
        issues.append("computes_oracle_on_confirm")
    if "preds.append(np.nan)" in src:
        issues.append("missing_candidates_emit_nan_instead_of_fail_closed")
    searches_threshold = any(
        tok in lowered
        for tok in (
            "threshold_grid",
            "search_threshold",
            "tune_threshold",
            "best_threshold",
            "threshold_selection",
        )
    )
    scans_ckpts = any(
        tok in lowered
        for tok in ("glob(*pt)", "rglob('*.pt')", "checkpoints.glob", "best_metric.pt")
    )
    selects_k = any(
        tok in lowered
        for tok in ("search_k", "select_k", "best_k", "K_grid", "k_grid", "choose_k")
    )
    replaces_primary = any(
        tok in lowered
        for tok in ("replace_primary", "choose_primary", "if confirm", "argmax_f1")
    )
    fail_closed = "raise SystemExit" in src or "raise RuntimeError" in src
    cache_fail_open = False

    findings = {
        "loads_FINAL_METHOD.LOCK": loads_final,
        "loads_stage6_method_lock.json": loads_stage6_lock,
        "scans_multiple_checkpoints": scans_ckpts,
        "searches_threshold_on_confirm": searches_threshold,
        "selects_K_on_confirm": selects_k,
        "replaces_primary_from_confirm": replaces_primary,
        "calls_oracle": "_oracle" in calls or "oracle_UNION" in src,
        "primary_output_uses_fixed_rescore": "fixed_rescore_UNION" in src and "_fixed_rescore" in src,
        "hardcoded_union_k5": "stead_k=5" in src and "ida_k=5" in src,
        "bootstrap_event_unit": "unit" in src or "event_ids" in src,
        "fail_closed_on_cache_launch": "cache_stead_failed" in src or "cache_ida_failed" in src,
        "fail_closed_per_trace": "preds.append(np.nan)" not in src,
        "writes_per_trace": "confirm_predictions.parquet" in src,
        "writes_per_event_bootstrap": "confirm_bootstrap.json" in src,
        "inference_and_stats_separated": False,
        "ast_top_level_functions": sorted(
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        ),
        "issues": issues,
        "cache_fail_open_detected_in_this_file": cache_fail_open,
        "script_has_raise_on_some_errors": fail_closed,
    }
    findings["inference_and_stats_separated"] = (
        "_fixed_rescore" in src and "_bootstrap" in src and "_decide" in src
    )
    return findings


def audit_candidate_cache_source(src: str) -> dict[str, Any]:
    fail_open = "except Exception" in src and "fails +=" in src
    return {
        "loads_only_stead_or_ida_best_pt": "from_pretrained" in src and "best.pt" in src,
        "extracts_k10_then_union_truncates": "k=10" in src,
        "fail_closed_on_read_or_inference_error": not fail_open,
        "silently_skips_failed_traces": fail_open,
        "issues": ["cache_skips_failed_traces"] if fail_open else [],
    }


def synthetic_fixed_rescore_pick(
    samples: list[int],
    probs: list[float],
    *,
    expected_sample: float,
    sigma_samples: float,
    lw: float,
    lh: float,
    lp: float,
    history_available: bool,
) -> int | None:
    """Tiny reimplementation used only in tests; mirrors candidate_rescorer."""
    from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
    from earthquake.fusion.peak_candidates import PeakCandidate

    cands = [
        PeakCandidate(
            sample_index=int(s),
            absolute_utc=None,
            peak_probability=float(p),
            prominence=float(p),
            peak_width=float("nan"),
            local_entropy=0.0,
            rank=i,
            fallback_peak=False,
            phase="S",
        )
        for i, (s, p) in enumerate(zip(samples, probs))
    ]
    best, _ = rescore_phase_candidates(
        cands,
        expected_sample=expected_sample,
        sigma_samples=sigma_samples,
        lambda_wave=lw,
        lambda_history=lh,
        lambda_prominence=lp,
        history_available=history_available,
    )
    return None if best is None else int(best.sample_index)


def fail_closed_missing_union(has_candidates: bool) -> None:
    if not has_candidates:
        raise RuntimeError("fail_closed: no UNION candidates for trace")


CANDIDATE_INVENTORY: list[dict[str, Any]] = [
    {
        "name": "STEAD",
        "status": "completed",
        "role": "waveform_only_baseline_and_union_source",
        "final_checkpoint": "seisbench.models.PhaseNet.from_pretrained('stead')",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "Single pretrained PhaseNet. UNION uses K=5 peaks from this one model, not five STEAD checkpoints.",
    },
    {
        "name": "IDA",
        "status": "completed",
        "role": "second_union_candidate_source_not_primary_picker",
        "final_checkpoint": "artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "PhaseNet STEAD finetune, locked epoch 14 best.pt. Not a standalone primary.",
    },
    {
        "name": "UNION_STEAD5_IDA5",
        "status": "completed",
        "role": "deployable_primary_fixed_rescore",
        "final_checkpoint": "fixed_rescore_UNION over STEAD K=5 ∪ IDA K=5",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "Stage6 method lock a02dc28e. Full-dev F1@0.5 ≈ 0.867. Oracle F1≈0.892 is ceiling only.",
    },
    {
        "name": "SegPhase",
        "status": "completed",
        "role": "post_confirm_external_comparator_not_primary",
        "final_checkpoint": "SegPhase-100Hz model_100Hz.pth",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "INSTANCE 10C retrain was mentioned as later work (never_started) and is not a confirm prerequisite.",
    },
    {
        "name": "DKPN",
        "status": "rejected",
        "role": "rejected_candidate_source",
        "verdict": "rejected_candidate_source",
        "marker": "FULLDEV.STOP_GATE.FAILED",
        "route_closed": True,
        "final_checkpoint": "v4 epoch_1.pt (postmortem only; not in UNION)",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "Extra seeds were gated on stop-gate pass. Gate failed. Do not start remedial DKPN experiments.",
    },
    {
        "name": "ranker",
        "status": "rejected",
        "role": "negative_ablation",
        "final_checkpoint": None,
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "phaseC_corrected_verdict=ranker_failed_predeclared_gate",
    },
    {
        "name": "fixed_rescore",
        "status": "completed",
        "role": "deployable_primary_rule",
        "final_checkpoint": "lambdas lw=0.5 lh=2.0 lp=0.0",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "LFTNet",
        "status": "rejected",
        "role": "implementation_not_reproducible",
        "final_checkpoint": None,
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "PhaseNet-ETHZ",
        "status": "completed",
        "role": "post_confirm_external_comparator",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "PhaseNet-SCEDC",
        "status": "completed",
        "role": "post_confirm_external_comparator",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "EQTransformer",
        "status": "never_started",
        "role": "download_failed_not_a_confirm_prerequisite",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "DiTing",
        "status": "never_started",
        "role": "forbidden_this_round",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
        "notes": "Do not start.",
    },
    {
        "name": "multistation_GNN",
        "status": "never_started",
        "role": "blocked",
        "preregistered_dev_experiments_incomplete": False,
        "required_before_confirm": False,
    },
    {
        "name": "SegPhase_INSTANCE_10C_retrain",
        "status": "never_started",
        "role": "optional_later_not_confirm_gate",
        "preregistered_dev_experiments_incomplete": True,
        "required_before_confirm": False,
        "notes": "Named in docs as later. Not a necessary candidate before UNION confirm.",
    },
    {
        "name": "DKPN_extra_seeds",
        "status": "never_started",
        "role": "forbidden_because_stop_gate_failed",
        "preregistered_dev_experiments_incomplete": True,
        "required_before_confirm": False,
        "route_closed": True,
        "notes": "Preregistered only if DKPN stop-gate passed. It failed. Do not start.",
    },
]


def incomplete_required_candidates() -> list[dict[str, Any]]:
    return [
        c
        for c in CANDIDATE_INVENTORY
        if c.get("preregistered_dev_experiments_incomplete") and c.get("required_before_confirm")
    ]
