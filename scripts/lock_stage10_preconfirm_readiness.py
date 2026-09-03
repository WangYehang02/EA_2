#!/usr/bin/env python
"""Freeze Stage-10 final method + confirm analysis; audit one-shot confirm readiness.

Does not open confirm waveforms, predictions, or metrics. Does not run confirm.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json
from earthquake.stage10.preconfirm_readiness import (
    CANDIDATE_INVENTORY,
    assess_confirm_unread,
    audit_candidate_cache_source,
    audit_confirm_runner_source,
    incomplete_required_candidates,
    sha256_file,
    stat_only,
)
from earthquake.utils import ensure_dir


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc}"


def _dump(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text)
    digest = sha256_file(path)
    os.chmod(path, 0o444)
    return digest


def main() -> None:
    art = artifacts_dir()
    out = ensure_dir(art / "results" / "stage10")
    now = datetime.now(timezone.utc).isoformat()

    stage6_lock_path = art / "results" / "stage6" / "final_confirm" / "method_lock.json"
    stage6_lock_sha_path = art / "results" / "stage6" / "final_confirm" / "method_lock.sha256"
    protocol_path = art / "results" / "stage6" / "final_confirm" / "confirm_evaluation_protocol.json"
    protocol_src = ROOT / "scripts" / "write_stage6_confirm_protocol.py"
    schema_path = art / "results" / "stage6" / "phaseC" / "candidate_schema.json"
    ida_ckpt = art / "models" / "stage6" / "phasenet_ida_full_seed42" / "checkpoints" / "best.pt"
    stead_weight = Path("/home/yehang/.seisbench/models/v3/phasenet/stead.pt.v2")
    comparator = load_json(art / "results" / "stage7" / "comparator_registry.json")
    stage6_lock = load_json(stage6_lock_path)
    protocol = load_json(protocol_path)
    phasec2 = load_json(art / "results" / "stage6" / "phaseC2" / "phaseC_corrected_verdict.json")
    phasec1 = load_json(art / "results" / "stage6" / "phaseC1" / "phaseC1_final_verdict.json")
    status = load_json(art / "results" / "stage10" / "PROJECT_STATUS.json")
    dkpn_lock_path = (
        art
        / "results"
        / "stage10"
        / "data_cache"
        / "dkpn_clean_v4_seed42_fp32"
        / "full_dev_stop_gate"
        / "METHOD.LOCK.json"
    )
    dkpn_failed = Path(
        "/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v4_seed42_fp32/FULLDEV.STOP_GATE.FAILED"
    )

    stage6_lock_sha = stage6_lock_sha_path.read_text().strip()
    assert sha256_file(stage6_lock_path) == stage6_lock_sha

    stead_sha_recorded = comparator["models"]["PhaseNet-STEAD"]["weight_sha256"]
    stead_sha_now = sha256_file(stead_weight) if stead_weight.exists() else None
    ida_sha_now = sha256_file(ida_ckpt)
    ida_sha_lock = stage6_lock["candidate_sources"]["ida_sha256"]
    assert ida_sha_now == ida_sha_lock

    protocol_sha = sha256_file(protocol_path)
    protocol_stat = stat_only(protocol_path)
    schema_sha = sha256_file(schema_path)

    source_files = [
        ROOT / "src/earthquake/fusion/candidate_rescorer.py",
        ROOT / "src/earthquake/fusion/peak_candidates.py",
        ROOT / "src/earthquake/gating/cache_io.py",
        ROOT / "src/earthquake/history/residual_prior.py",
        ROOT / "src/earthquake/models/seisbench_reference.py",
        ROOT / "src/earthquake/stage6/ranker/union_schema.py",
        ROOT / "src/earthquake/stage6/phaseB.py",
        ROOT / "src/earthquake/data/hdf5_reader.py",
        ROOT / "scripts/run_stage6_final_confirm.py",
        ROOT / "scripts/cache_stage6_phaseB_candidates.py",
        ROOT / "scripts/lock_stage6_final_method.py",
        ROOT / "scripts/write_stage6_confirm_protocol.py",
    ]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}

    runner_src = (ROOT / "scripts" / "run_stage6_final_confirm.py").read_text()
    cache_src = (ROOT / "scripts" / "cache_stage6_phaseB_candidates.py").read_text()
    runner_audit = audit_confirm_runner_source(runner_src)
    cache_audit = audit_candidate_cache_source(cache_src)
    unread = assess_confirm_unread(artifacts=art)

    k = 5
    stead_slots = [
        {
            "slot": i,
            "role": f"STEAD_peak_candidate_k{i}",
            "not_an_independent_model": True,
            "model": "PhaseNet-STEAD",
            "weight_name": "stead",
            "path": str(stead_weight),
            "sha256": stead_sha_now or stead_sha_recorded,
        }
        for i in range(k)
    ]
    ida_slots = [
        {
            "slot": i,
            "role": f"IDA_peak_candidate_k{i}",
            "not_an_independent_model": True,
            "model": "PhaseNet-IDA-best.pt-epoch14",
            "path": str(ida_ckpt),
            "sha256": ida_sha_now,
            "epoch": 14,
        }
        for i in range(k)
    ]

    env = stage6_lock["environment"]
    env_now = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status_short": _run(["git", "status", "--short"]),
    }

    final_method = {
        "lock_name": "FINAL_METHOD.LOCK",
        "locked_utc": now,
        "round": "stage10_preconfirm_readiness_audit",
        "do_not_modify_after_sha256": True,
        "primary_name": "UNION_STEAD5_IDA5 fixed-rescore",
        "primary_lock_name": "fixed_rescore_UNION",
        "deployable_primary": True,
        "contains_DKPN": False,
        "oracle_is_not_confirm_primary": True,
        "may_not_change_primary_from_confirm": True,
        "copied_from_stage6_method_lock_sha256": stage6_lock_sha,
        "stage6_method_lock_path": str(stage6_lock_path),
        "stage6_locked_before_confirm": bool(stage6_lock.get("locked_before_confirm")),
        "confirm_seen_before_stage6_lock": bool(stage6_lock.get("confirm_seen_before_lock")),
        "full_dev": {
            "source": "stage6_phaseC1_and_PROJECT_STATUS_dev_only",
            "n_traces": 87293,
            "n_events": 5341,
            "manifest_sha256": stage6_lock["data_eval"]["dev_eval_manifest_sha256"],
            "primary_f1@0.5": 0.8668048984454653,
            "primary_f1@0.5_rounded": 0.867,
            "STEAD_top1_f1@0.5": 0.8402621057816778,
            "UNION_oracle_f1@0.5": 0.8916980743014904,
            "oracle_role": "ceiling_only_not_deployable_not_confirm_primary",
            "phaseC1_fixed_rescore_UNION_f1@0.5": phasec1["fixed_rescore_UNION"]["f1@0.5"],
            "phaseC2_verdict": phasec2["verdict"],
            "PROJECT_STATUS_primary_f1@0.5": status["full_dev"]["primary_f1@0.5"],
        },
        "STEAD_candidates": {
            "n_independent_checkpoints": 1,
            "k": k,
            "definition": "top-K S peaks from one PhaseNet-STEAD, not five STEAD models",
            "checkpoint": {
                "source": "seisbench.models.PhaseNet.from_pretrained('stead')",
                "weight_name": "stead",
                "path": str(stead_weight),
                "sha256_recorded_stage7": stead_sha_recorded,
                "sha256_rehashed_this_audit": stead_sha_now,
                "sha256_match": stead_sha_now == stead_sha_recorded,
            },
            "slots": stead_slots,
        },
        "IDA_candidates": {
            "n_independent_checkpoints": 1,
            "k": k,
            "definition": "top-K S peaks from one IDA best.pt, not five IDA models",
            "checkpoint": {
                "path": str(ida_ckpt),
                "epoch": 14,
                "sha256": ida_sha_now,
                "name_must_be_best_pt": True,
                "last_pt_forbidden": True,
            },
            "slots": ida_slots,
        },
        "primary_baseline": {
            "name": "STEAD_top1",
            "checkpoint_path": str(stead_weight),
            "checkpoint_sha256": stead_sha_now or stead_sha_recorded,
            "definition": "PhaseNet-STEAD annotate argmax S peak (cache top1_s_sample)",
        },
        "K": k,
        "union": {
            "stead_k": 5,
            "ida_k": 5,
            "max_union": 10,
            "dedup_s": 0.05,
            "representative_time": "STEAD_preferred_when_both_support",
            "probability_combination": "never_average; rescore uses max(available_source_prob, 1e-6)",
            "schema_path": str(schema_path),
            "schema_sha256": schema_sha,
            "schema_sha256_in_stage6_lock": stage6_lock["candidate_sources"]["schema_sha256"],
        },
        "input": {
            "sampling_rate_hz": 100.0,
            "window": "SeisBench PhaseNet.annotate default (in_samples=3001 @ 100 Hz, overlap per SeisBench)",
            "component_order": "ENZ_HDF5_to_ZNE_SeisBench",
            "label_order": "PSN",
            "utc_remap": "seisbench_reference.remap_trace_to_waveform_grid",
        },
        "normalization_filter": {
            "normalization": "SeisBench PhaseNet.annotate default window normalization; no extra project zscore",
            "filter": "none_beyond_SeisBench_PhaseNet_annotate",
        },
        "peak_extraction": {
            "backend": "scipy.signal.find_peaks",
            "k_extracted_before_union": 10,
            "k_used_in_union": 5,
            "min_distance_samples": 50,
            "min_prominence": 0.05,
            "min_probability": 0.1,
            "phase": "S",
            "fallback": "global_argmax_if_no_peak_passes_threshold",
        },
        "base_model_thresholds": {
            "STEAD_peak_min_probability": 0.1,
            "IDA_peak_min_probability": 0.1,
            "STEAD_top1_annotate_peak_threshold": 0.0,
            "shared_CAND_EXTRACT_CFG": True,
            "not_tuned_on_confirm": True,
        },
        "fixed_rescore": {
            "formula": (
                "score = lw * log(p + eps) + lh * log(hist + eps) + lp * (prominence / max_prominence); "
                "hist = exp(-0.5 * ((candidate_sample - expected_s_sample) / sigma_samples)^2); "
                "lh := 0 if history unavailable"
            ),
            "eps": 1e-8,
            "lambdas_s": {"lw": 0.5, "lh": 2.0, "lp": 0.0},
            "shrinkage_k": 50.0,
            "min_history": 5,
            "mad_disable_s": 1.0,
            "min_sigma_s": 0.05,
            "max_sigma_s": 1.0,
            "sigma_s": "clip(1.4826 * residual_s_mad, min_sigma_s, max_sigma_s)",
            "expected_s": (
                "pred_tau_s = base_tau_s + shrunk_residual_s; "
                "expected_s_sample from origin_time + pred_tau_s on waveform grid"
            ),
            "global_residual": stage6_lock["fixed_rescore"]["global_residual"],
            "history_paths": {
                "distance_mlp_path": stage6_lock["fixed_rescore"]["distance_mlp_path"],
                "distance_mlp_sha256": stage6_lock["fixed_rescore"]["distance_mlp_sha256"],
                "history_features_path": stage6_lock["fixed_rescore"]["history_features_path"],
                "history_features_sha256": stage6_lock["fixed_rescore"]["history_features_sha256"],
                "history_store_path": stage6_lock["fixed_rescore"]["history_store_path"],
                "history_store_sha256": stage6_lock["fixed_rescore"]["history_store_sha256"],
            },
            "candidate_probability_for_score": "max(stead_probability, ida_probability, 1e-6)",
            "tie_break": (
                "strictly greater score wins; equal scores keep the first candidate in "
                "rescore iteration order. UNION pool is sorted by (-both_support, -max_prob, stead_rank) "
                "before max_union truncation. Both-support peaks keep STEAD representative time."
            ),
        },
        "missing_or_corrupt_trace_rules": {
            "preregistered": True,
            "source": "src/earthquake/data/hdf5_reader.py + stage6 method_lock no_candidate_behavior",
            "missing_E_N_Z_component": "raise KeyError; do not impute",
            "unrecognized_waveform_shape": "raise ValueError",
            "trace_not_in_hdf5": "raise KeyError",
            "nonfinite_probability": "raise RuntimeError in cache shard",
            "no_union_candidates": "stage6 lock: nan_prediction (historical). Readiness requires fail_closed instead.",
            "corrupt_trace": "do not skip after error; fail closed",
            "historical_cache_script_actually": "catches Exception and skips trace (NOT fail closed)",
        },
        "code": {
            "git_commit_at_stage6_lock": env.get("git_commit"),
            "git_diff_hash_at_stage6_lock": env.get("git_diff_hash"),
            "git_commit_this_audit": env_now["git_commit"],
            "source_file_sha256": source_hashes,
            "metrics_module_sha256": stage6_lock["data_eval"]["metrics_module_sha256"],
            "keyed_align_sha256": stage6_lock["data_eval"]["keyed_align_sha256"],
        },
        "config_hash": {
            "stage6_method_lock_sha256": stage6_lock_sha,
            "union_confirm_protocol_sha256": protocol_sha,
            "candidate_schema_sha256": schema_sha,
            "write_protocol_script_sha256": sha256_file(protocol_src),
        },
        "environment": {
            "python": env["python"],
            "torch": env["torch"],
            "seisbench": env["seisbench"],
            "cuda": env["cuda"],
            "numpy": env["numpy"],
            "pandas": env["pandas"],
            "scipy": env["scipy"],
            "conda_explicit_hash": env["conda_explicit_hash"],
            "audit_python": env_now["python"],
            "audit_platform": env_now["platform"],
        },
        "full_dev_manifest_sha256": stage6_lock["data_eval"]["dev_eval_manifest_sha256"],
        "confirm_split_hashes_from_existing_records_only": {
            "confirm_events_sha256": stage6_lock["data_eval"]["confirm_events_sha256"],
            "confirm_traces_sha256": stage6_lock["data_eval"]["confirm_traces_sha256"],
            "n_confirm_events_recorded": stage6_lock["data_eval"]["confirm_events"],
            "n_confirm_traces_recorded": stage6_lock["data_eval"]["confirm_traces"],
            "content_not_rehashed_this_round": True,
        },
        "DKPN_excluded": {
            "status": "rejected_candidate_source",
            "marker": "FULLDEV.STOP_GATE.FAILED",
            "route_closed": True,
            "in_primary": False,
        },
    }

    analysis = {
        "lock_name": "CONFIRM_ANALYSIS.LOCK",
        "locked_utc": now,
        "copied_from_existing_union_confirm_protocol": True,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "protocol_mtime": protocol_stat["mtime"],
        "protocol_atime": protocol_stat["atime"],
        "must_not_modify_existing_union_gate": True,
        "primary_comparison": {
            "method": "UNION_STEAD5_IDA5 fixed-rescore",
            "method_lock_name": "fixed_rescore_UNION",
            "baseline": "STEAD_top1",
            "protocol_string": protocol["primary_comparison"],
        },
        "primary_metric": {
            "name": "F1@0.5s",
            "protocol_endpoint": protocol["primary_endpoint"],
        },
        "secondary_metrics": [
            "Precision@0.1s",
            "Recall@0.1s",
            "F1@0.1s",
            "miss",
            "coverage",
            "picks_per_trace",
            "no_S_peak_rate",
            "detected_AE_median",
            "detected_AE_MAE",
            "detected_AE_P95",
            "wrong_peak_rate",
            "inference_runtime",
        ],
        "noise_FPR": {
            "include": False,
            "reason": "UNION confirm protocol population is all_s_labelled_internal_confirm; no preregistered confirm noise set in confirm_evaluation_protocol.json",
        },
        "statistics": {
            "pairing_unit": "event",
            "bootstrap": "paired_event_bootstrap",
            "n": 5000,
            "seed": 20260817,
            "protocol_bootstrap": protocol["bootstrap"],
            "report": "ΔF1@0.5 point estimate and 95% CI (method minus STEAD_top1)",
            "trace_bootstrap_forbidden": True,
        },
        "subgroups": {
            "union_confirm_protocol_lists_none": True,
            "preregistered_for_UNION_blind_confirm": [],
            "do_not_add_after_seeing_confirm": True,
            "user_named_families": [
                "P-S interval",
                "distance",
                "SNR",
                "station",
                "channel",
                "network",
            ],
            "those_families_preregistered_in": (
                "DKPN v4 FULLDEV stop-gate METHOD.LOCK grouping only; "
                "NOT the UNION confirm protocol. Do not import DKPN grouping as UNION confirm subgroups."
            ),
        },
        "diagnostic_only": protocol["diagnostic_only"],
        "forbidden_on_confirm": protocol["forbidden_on_confirm"],
        "oracle_union": {
            "allowed_as_diagnostic": True,
            "allowed_as_primary_output": False,
            "actual_per_trace_prediction_must_be": "fixed_rescore_UNION",
        },
        "claim_scope": protocol["claim_scope"],
        "sota_claim_allowed": False,
        "blind_picker_claim_allowed": False,
    }

    union_gate = {
        "exists": True,
        "kind": "UNION_blind_confirm_gate",
        "path": str(protocol_path),
        "mtime": protocol_stat["mtime"],
        "sha256": protocol_sha,
        "source_script": str(protocol_src),
        "quoted_verbatim_decision_rules": protocol["decision_rules"],
        "modified_this_round": False,
        "not_the_dkpn_stop_gate": True,
    }
    dkpn_gate = {
        "exists": dkpn_lock_path.exists(),
        "kind": "DKPN_candidate_FULLDEV_stop_gate",
        "path": str(dkpn_lock_path),
        "sha256": sha256_file(dkpn_lock_path) if dkpn_lock_path.exists() else None,
        "failed_marker_exists": dkpn_failed.exists(),
        "failed_marker_stat": stat_only(dkpn_failed) if dkpn_failed.exists() else None,
        "quoted_stop_gate": {
            "A": "DKPN top1 vs strongest waveform-only ΔF1@0.5 >= +0.01",
            "B": "F1 not down AND detected AE P95 improve >= 0.15s AND miss not worse",
            "C": "UNION+DKPN5 oracle ΔF1@0.5 >= +0.01 AND event-bootstrap CI lo > 0",
        },
        "must_not_be_used_as_UNION_confirm_gate": True,
    }

    incomplete_required = incomplete_required_candidates()
    block_reasons: list[str] = []
    if incomplete_required:
        block_reasons.append("CONFIRM.BLOCKED:incomplete_required_preregistered_candidates")
    if not union_gate["exists"]:
        block_reasons.append("CONFIRM.GATE.MISSING")
    if unread["historical_confirm_consumed_marker_exists"]:
        block_reasons.append("confirm_already_consumed_not_a_first_blind_look")
    if not runner_audit["loads_FINAL_METHOD.LOCK"]:
        block_reasons.append("confirm_runner_does_not_load_FINAL_METHOD.LOCK")
    if not runner_audit["fail_closed_per_trace"]:
        block_reasons.append("confirm_runner_nan_skip_not_fail_closed")
    if cache_audit["silently_skips_failed_traces"]:
        block_reasons.append("candidate_cache_skips_failed_traces")
    if runner_audit["calls_oracle"]:
        block_reasons.append("confirm_runner_computes_oracle_on_confirm")

    # Gate exists and is UNION; missing-gate path is not taken.
    verdict = "CONFIRM.BLOCKED" if block_reasons else "CONFIRM.READY"

    readiness = {
        "index_name": "CONFIRM_READINESS",
        "locked_utc": now,
        "verdict": verdict,
        "allow_one_shot_confirm_execution": False,
        "auto_ran_confirm": False,
        "await_explicit_user_approval": True,
        "CONFIRM.GATE.MISSING": False,
        "block_reasons": block_reasons,
        "incomplete_required_preregistered_candidates": incomplete_required,
        "candidate_inventory": CANDIDATE_INVENTORY,
        "DKPN": {
            "status": "rejected",
            "verdict": "rejected_candidate_source",
            "marker": "FULLDEV.STOP_GATE.FAILED",
            "route_closed": True,
            "do_not_start_remedial_experiments": True,
        },
        "primary": "UNION_STEAD5_IDA5 fixed-rescore",
        "baseline": "STEAD_top1",
        "primary_metric": "F1@0.5s",
        "union_confirm_gate": union_gate,
        "dkpn_stop_gate": dkpn_gate,
        "confirm_unread_this_process": {
            "confirm_waveforms_read": unread["confirm_waveforms_read"],
            "confirm_predictions_read": unread["confirm_predictions_read"],
            "confirm_metrics_read": unread["confirm_metrics_read"],
            "confirm_threshold_tuned": unread["confirm_threshold_tuned"],
            "confirm_model_selected": unread["confirm_model_selected"],
            "this_process_fd_open_on_confirm_content": unread["this_process_fd_open_on_confirm_content"],
        },
        "confirm_historical": {
            "CONFIRM.CONSUMED_exists": unread["historical_confirm_consumed_marker_exists"],
            "confirm_predictions_exist": unread["confirm_predictions_exist"],
            "confirm_metrics_exist": unread["confirm_metrics_exist"],
            "one_shot_already_executed": unread["historical_confirm_consumed_marker_exists"],
            "content_not_opened_this_round": True,
        },
        "script_audit": {
            "runner": str(ROOT / "scripts" / "run_stage6_final_confirm.py"),
            "cache": str(ROOT / "scripts" / "cache_stage6_phaseB_candidates.py"),
            "runner_findings": runner_audit,
            "cache_findings": cache_audit,
        },
        "pytest": {
            "file": "tests/test_stage10_preconfirm_readiness.py",
            "uses_confirm_waveforms": False,
            "uses_confirm_predictions": False,
            "uses_confirm_metrics": False,
        },
    }

    method_path = out / "FINAL_METHOD.LOCK.json"
    analysis_path = out / "CONFIRM_ANALYSIS.LOCK.json"
    ready_path = out / "CONFIRM_READINESS.json"
    for p in (method_path, analysis_path, ready_path):
        if p.exists():
            os.chmod(p, 0o644)

    method_sha = _dump(method_path, final_method)
    analysis_sha = _dump(analysis_path, analysis)
    readiness["FINAL_METHOD.LOCK.sha256"] = method_sha
    readiness["CONFIRM_ANALYSIS.LOCK.sha256"] = analysis_sha
    ready_sha = _dump(ready_path, readiness)
    (out / "FINAL_METHOD.LOCK.sha256").write_text(method_sha + "\n")
    (out / "CONFIRM_ANALYSIS.LOCK.sha256").write_text(analysis_sha + "\n")
    (out / "CONFIRM_READINESS.sha256").write_text(ready_sha + "\n")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "FINAL_METHOD.LOCK.sha256": method_sha,
                "CONFIRM_ANALYSIS.LOCK.sha256": analysis_sha,
                "CONFIRM_READINESS.sha256": ready_sha,
                "block_reasons": block_reasons,
                "protocol_sha256": protocol_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
