"""Preconfirm readiness tests: synthetic/dev mocks only. Never open confirm content."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.stage10.preconfirm_readiness import (
    CANDIDATE_INVENTORY,
    audit_candidate_cache_source,
    audit_confirm_runner_source,
    fail_closed_missing_union,
    incomplete_required_candidates,
    sha256_file,
    stat_only,
    synthetic_fixed_rescore_pick,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE10 = ROOT / "artifacts" / "results" / "stage10"


def _load_lock(name: str) -> dict:
    path = STAGE10 / name
    assert path.exists(), path
    return json.loads(path.read_text())


def test_no_incomplete_required_candidates():
    assert incomplete_required_candidates() == []
    dkpn = next(c for c in CANDIDATE_INVENTORY if c["name"] == "DKPN")
    assert dkpn["status"] == "rejected"
    assert dkpn["verdict"] == "rejected_candidate_source"
    assert dkpn["marker"] == "FULLDEV.STOP_GATE.FAILED"
    assert dkpn["route_closed"] is True
    extra = next(c for c in CANDIDATE_INVENTORY if c["name"] == "DKPN_extra_seeds")
    assert extra["required_before_confirm"] is False
    assert extra["route_closed"] is True


def test_final_method_lock_primary_not_oracle_not_dkpn():
    lock = _load_lock("FINAL_METHOD.LOCK.json")
    assert lock["primary_name"] == "UNION_STEAD5_IDA5 fixed-rescore"
    assert lock["primary_lock_name"] == "fixed_rescore_UNION"
    assert lock["contains_DKPN"] is False
    assert lock["oracle_is_not_confirm_primary"] is True
    assert lock["K"] == 5
    assert lock["full_dev"]["primary_f1@0.5_rounded"] == 0.867
    assert lock["full_dev"]["UNION_oracle_f1@0.5"] == pytest.approx(0.8916980743014904)
    assert lock["STEAD_candidates"]["n_independent_checkpoints"] == 1
    assert lock["IDA_candidates"]["n_independent_checkpoints"] == 1
    assert len(lock["STEAD_candidates"]["slots"]) == 5
    assert len(lock["IDA_candidates"]["slots"]) == 5
    assert lock["primary_baseline"]["name"] == "STEAD_top1"
    assert lock["fixed_rescore"]["lambdas_s"] == {"lh": 2.0, "lp": 0.0, "lw": 0.5}
    sha_path = STAGE10 / "FINAL_METHOD.LOCK.sha256"
    assert sha256_file(STAGE10 / "FINAL_METHOD.LOCK.json") == sha_path.read_text().strip()


def test_confirm_analysis_lock_event_bootstrap_and_no_new_subgroups():
    lock = _load_lock("CONFIRM_ANALYSIS.LOCK.json")
    protocol = ROOT / "artifacts" / "results" / "stage6" / "final_confirm" / "confirm_evaluation_protocol.json"
    proto = json.loads(protocol.read_text())
    assert lock["protocol_sha256"] == sha256_file(protocol)
    assert lock["primary_comparison"]["baseline"] == "STEAD_top1"
    assert lock["primary_metric"]["name"] == "F1@0.5s"
    assert lock["statistics"]["pairing_unit"] == "event"
    assert lock["statistics"]["n"] == 5000
    assert lock["statistics"]["seed"] == 20260817
    assert lock["statistics"]["trace_bootstrap_forbidden"] is True
    assert lock["subgroups"]["preregistered_for_UNION_blind_confirm"] == []
    assert "DKPN" in lock["subgroups"]["those_families_preregistered_in"]
    assert proto["decision_rules"]["strong_confirmed"].startswith("dF1_05>=0.01")
    assert lock["oracle_union"]["allowed_as_primary_output"] is False


def test_confirm_analysis_quotes_protocol_rules():
    lock = _load_lock("CONFIRM_ANALYSIS.LOCK.json")
    protocol = json.loads(
        (ROOT / "artifacts" / "results" / "stage6" / "final_confirm" / "confirm_evaluation_protocol.json").read_text()
    )
    ready = _load_lock("CONFIRM_READINESS.json")
    assert ready["union_confirm_gate"]["quoted_verbatim_decision_rules"] == protocol["decision_rules"]
    assert ready["union_confirm_gate"]["not_the_dkpn_stop_gate"] is True
    assert ready["dkpn_stop_gate"]["must_not_be_used_as_UNION_confirm_gate"] is True
    assert ready["CONFIRM.GATE.MISSING"] is False
    assert lock["protocol_sha256"] == ready["union_confirm_gate"]["sha256"]


def test_readiness_blocked_and_does_not_authorize_execution():
    ready = _load_lock("CONFIRM_READINESS.json")
    assert ready["verdict"] == "CONFIRM.BLOCKED"
    assert ready["allow_one_shot_confirm_execution"] is False
    assert ready["auto_ran_confirm"] is False
    assert ready["DKPN"]["route_closed"] is True
    unread = ready["confirm_unread_this_process"]
    assert unread["confirm_waveforms_read"] is False
    assert unread["confirm_predictions_read"] is False
    assert unread["confirm_metrics_read"] is False
    assert unread["confirm_threshold_tuned"] is False
    assert unread["confirm_model_selected"] is False
    assert ready["confirm_historical"]["one_shot_already_executed"] is True
    reasons = ready["block_reasons"]
    assert "confirm_already_consumed_not_a_first_blind_look" in reasons
    assert "candidate_cache_skips_failed_traces" in reasons


def test_static_audit_existing_runner_has_no_threshold_or_k_search():
    src = (ROOT / "scripts" / "run_stage6_final_confirm.py").read_text()
    findings = audit_confirm_runner_source(src)
    assert findings["searches_threshold_on_confirm"] is False
    assert findings["selects_K_on_confirm"] is False  # K is hardcoded from lock, not searched
    assert findings["scans_multiple_checkpoints"] is False
    assert findings["replaces_primary_from_confirm"] is False
    assert findings["hardcoded_union_k5"] is True
    assert findings["primary_output_uses_fixed_rescore"] is True
    assert findings["loads_FINAL_METHOD.LOCK"] is False
    assert findings["calls_oracle"] is True
    assert findings["fail_closed_per_trace"] is False
    cache = (ROOT / "scripts" / "cache_stage6_phaseB_candidates.py").read_text()
    cf = audit_candidate_cache_source(cache)
    assert cf["silently_skips_failed_traces"] is True
    ast.parse(src)
    ast.parse(cache)


def test_synthetic_fixed_rescore_and_fail_closed_mock():
    pick = synthetic_fixed_rescore_pick(
        [50, 200],
        [0.9, 0.8],
        expected_sample=200.0,
        sigma_samples=20.0,
        lw=0.5,
        lh=2.0,
        lp=0.0,
        history_available=True,
    )
    assert pick == 200
    with pytest.raises(RuntimeError, match="fail_closed"):
        fail_closed_missing_union(False)
    fail_closed_missing_union(True)
    cands = [
        PeakCandidate(10, None, 0.4, 0.4, float("nan"), 0.0, 0, False, "S"),
        PeakCandidate(10, None, 0.4, 0.4, float("nan"), 0.0, 1, False, "S"),
    ]
    best, rows = rescore_phase_candidates(
        cands,
        expected_sample=10.0,
        sigma_samples=5.0,
        lambda_wave=0.5,
        lambda_history=2.0,
        lambda_prominence=0.0,
        history_available=True,
    )
    assert best is not None
    assert best.rank == 0
    assert np.isfinite(rows[0]["score"])


def test_stat_only_does_not_need_open(tmp_path):
    p = tmp_path / "mock_confirm_metrics.json"
    p.write_text('{"f1": 0.0}\n')
    meta = stat_only(p)
    assert meta["exists"] is True
    assert meta["opened_this_process"] is False
    assert meta["size_bytes"] == len('{"f1": 0.0}\n')


def test_pytest_module_has_no_confirm_io_imports():
    tree = ast.parse(Path(__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.add(node.module)
    assert "pandas" not in imported
    assert "earthquake.data" not in imported
    assert "hdf5_reader" not in imported
