#!/usr/bin/env python
"""Audit frozen evidence and write FINAL_EVIDENCE_LOCK (no new experiments)."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "final_evidence"
REPORT = ROOT / "reports" / "paper"
EXPECTED_METHOD_LOCK = "1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b"
EXPECTED_CONFIRM_EXEC = "0839e1a8f70123c196867f1d2da417e9f913d8359e917c557959b9848f620a15"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def round4(x: float) -> float:
    return float(f"{x:.4f}")


def round3(x: float) -> float:
    return float(f"{x:.3f}")


def check(name: str, observed: float, reported: float, tol: float, failures: list) -> dict:
    ok = abs(float(observed) - float(reported)) <= tol
    row = {"name": name, "observed": observed, "reported": reported, "tol": tol, "ok": ok}
    if not ok:
        failures.append(row)
    return row


def main() -> int:
    ensure_dir(OUT)
    ensure_dir(REPORT)
    failures: list[dict] = []
    checks: list[dict] = []

    # --- paths ---
    method_sha_p = artifacts_dir() / "results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256"
    confirm_sha_p = artifacts_dir() / "results/pairwise_confirm/SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.sha256"
    method_lock_p = artifacts_dir() / "results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    confirm_lock_p = artifacts_dir() / "results/pairwise_confirm/SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.json"
    confirm_fixed = artifacts_dir() / "results/pairwise_confirm/confirm_fixed_metrics.json"
    confirm_resid = artifacts_dir() / "results/pairwise_confirm/confirm_resid_metrics.json"
    confirm_scalar = artifacts_dir() / "results/pairwise_confirm/confirm_scalar_metrics.json"
    boot_sf = artifacts_dir() / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_fixed.json"
    boot_sr = artifacts_dir() / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_resid.json"
    confirm_final = artifacts_dir() / "results/pairwise_confirm/confirm_final.json"
    confirm_status = artifacts_dir() / "results/pairwise_confirm/SCALAR_PAIRWISE.CONFIRM.CONFIRMED"
    fulldev_passed = artifacts_dir() / "results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV.PASSED"
    fulldev_final = artifacts_dir() / "results/pairwise_fulldev/fulldev_final.json"
    pilot_ablation = artifacts_dir() / "results/pairwise_pilot/ablation_metrics.json"
    rank_dist = artifacts_dir() / "results/multistation_moveout/ranking_gap_rank_distribution.json"
    resid_audit = artifacts_dir() / "results/pairwise_pilot/residual_control_audit.json"
    wf_boot = artifacts_dir() / "results/pairwise_pilot/waveform_bootstrap.json"
    ckpt = artifacts_dir() / "results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt"
    metrics_py = ROOT / "src/earthquake/metrics.py"
    effect_csv = artifacts_dir() / "results/pairwise_confirm/effect_replication_summary.csv"

    required = [
        method_sha_p,
        confirm_sha_p,
        method_lock_p,
        confirm_lock_p,
        confirm_fixed,
        confirm_resid,
        confirm_scalar,
        boot_sf,
        boot_sr,
        confirm_final,
        confirm_status,
        fulldev_passed,
        fulldev_final,
        pilot_ablation,
        rank_dist,
        resid_audit,
        wf_boot,
        ckpt,
        metrics_py,
        effect_csv,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        save_json({"missing": missing}, OUT / "FINAL_EVIDENCE.BLOCKED.json")
        (OUT / "FINAL_EVIDENCE.BLOCKED").write_text("missing artifacts\n", encoding="utf-8")
        print("FINAL_EVIDENCE.BLOCKED missing", missing)
        return 2

    method_sha = method_sha_p.read_text().strip()
    confirm_exec_sha = confirm_sha_p.read_text().strip()
    checks.append(check("method_lock_hash", 0 if method_sha == EXPECTED_METHOD_LOCK else 1, 0, 0, failures))
    if method_sha != EXPECTED_METHOD_LOCK:
        failures[-1]["observed_str"] = method_sha
        failures[-1]["reported_str"] = EXPECTED_METHOD_LOCK
    checks.append(check("confirm_exec_hash", 0 if confirm_exec_sha == EXPECTED_CONFIRM_EXEC else 1, 0, 0, failures))
    if confirm_exec_sha != EXPECTED_CONFIRM_EXEC:
        failures[-1]["observed_str"] = confirm_exec_sha
        failures[-1]["reported_str"] = EXPECTED_CONFIRM_EXEC

    cf = json.loads(confirm_fixed.read_text())
    cr = json.loads(confirm_resid.read_text())
    cs = json.loads(confirm_scalar.read_text())
    bsf = json.loads(boot_sf.read_text())
    bsr = json.loads(boot_sr.read_text())
    cfin = json.loads(confirm_final.read_text())
    fd = json.loads(fulldev_final.read_text())
    ab = json.loads(pilot_ablation.read_text())
    rd = json.loads(rank_dist.read_text())
    ra = json.loads(resid_audit.read_text())
    wb = json.loads(wf_boot.read_text())

    # confirm core
    checks.append(check("confirm_fixed_f1@0.5_round4", round4(cf["f1@0.5"]), 0.8373, 5e-5, failures))
    checks.append(check("confirm_resid_f1@0.5_round4", round4(cr["f1@0.5"]), 0.8459, 5e-5, failures))
    checks.append(check("confirm_scalar_f1@0.5_round4", round4(cs["f1@0.5"]), 0.8535, 5e-5, failures))
    checks.append(check("confirm_scalar_fixed_delta_round4", round4(cs["delta_f1@0.5_vs_fixed"]), 0.0162, 5e-5, failures))
    checks.append(check("confirm_scalar_resid_delta_round4", round4(cs["delta_f1@0.5_vs_resid"]), 0.0076, 5e-5, failures))
    checks.append(check("confirm_fixes", cs["fixes"], 907, 0, failures))
    checks.append(check("confirm_breaks", cs["breaks"], 210, 0, failures))
    checks.append(check("confirm_net", cs["net_fixes"], 697, 0, failures))
    checks.append(check("confirm_switch_precision_round3", round3(cs["switch_precision"]), 0.812, 5e-4, failures))
    checks.append(check("confirm_q2_recovery_round3", round3(cs["Q2_recovery_rate"]), 0.773, 5e-4, failures))
    checks.append(check("confirm_fixed_f1@0.1_round4", round4(cf["f1@0.1"]), 0.5035, 5e-5, failures))
    checks.append(check("confirm_p95_round2", round(cf["detected_ae_p95"], 2), 2.47, 0.015, failures))

    # bootstrap CIs
    checks.append(check("boot_sf_mean_round4", round4(bsf["delta_f1@0.5"]["mean"]), 0.0162, 5e-5, failures))
    checks.append(check("boot_sf_ci_lo_round4", round4(bsf["delta_f1@0.5"]["ci95"][0]), 0.0146, 5e-4, failures))
    checks.append(check("boot_sf_ci_hi_round4", round4(bsf["delta_f1@0.5"]["ci95"][1]), 0.0178, 5e-4, failures))
    checks.append(check("boot_sr_mean_round4", round4(bsr["delta_f1@0.5"]["mean"]), 0.0076, 5e-5, failures))
    checks.append(check("boot_sr_ci_lo_round4", round4(bsr["delta_f1@0.5"]["ci95"][0]), 0.0065, 5e-4, failures))
    checks.append(check("boot_sr_ci_hi_round4", round4(bsr["delta_f1@0.5"]["ci95"][1]), 0.0088, 5e-4, failures))
    if not (bsf["delta_f1@0.5"]["ci95"][0] > 0 and bsr["delta_f1@0.5"]["ci95"][0] > 0):
        failures.append({"name": "bootstrap_ci_gt_0", "ok": False})
    if cfin.get("status") != "CONFIRMED":
        failures.append({"name": "confirm_status", "observed": cfin.get("status"), "reported": "CONFIRMED", "ok": False})

    # replication
    checks.append(check("pilot_fixed_round4", round4(ab["fixed_rescore_UNION"]["f1@0.5"]), 0.9196, 5e-5, failures))
    checks.append(check("pilot_resid_round4", round4(ab["resid_s_control"]["f1@0.5"]), 0.9350, 5e-5, failures))
    checks.append(check("pilot_scalar_round4", round4(ab["scalar_pairwise"]["f1@0.5"]), 0.9403, 5e-5, failures))
    checks.append(check("pilot_delta_fixed_round4", round4(ab["scalar_pairwise"]["delta_f1@0.5"]), 0.0207, 5e-5, failures))
    checks.append(check("fd_fixed_round4", round4(fd["fixed"]["f1@0.5"]), 0.8668, 5e-5, failures))
    checks.append(check("fd_resid_round4", round4(fd["resid"]["f1@0.5"]), 0.8732, 5e-5, failures))
    checks.append(check("fd_scalar_round4", round4(fd["scalar"]["f1@0.5"]), 0.8805, 5e-5, failures))

    # ranking forensic
    checks.append(check("n_recoverable", rd["n_recoverable"], 2173, 0, failures))
    checks.append(check("rank2", rd["rank_good_eq_2"], 1820, 0, failures))
    checks.append(check("rank3", rd["rank_good_eq_3"], 307, 0, failures))
    checks.append(check("rank4", rd["rank_good_eq_4"], 40, 0, failures))
    checks.append(check("rank5", rd["rank_good_eq_5"], 6, 0, failures))
    checks.append(check("P_le2_round3", round3(rd["P_rank_good_le_2"]), 0.838, 5e-4, failures))
    checks.append(check("P_le3_round3", round3(rd["P_rank_good_le_3"]), 0.979, 5e-4, failures))

    # residual corr
    corr = ra.get("interpretation", {}).get("corr_resid_expected_vs_base")
    if corr is None:
        corr = ra.get("geometry_comparison", {}).get("corr_resid_expected_vs_base")
    checks.append(check("resid_corr_round3", round3(float(corr)), 0.998, 5e-4, failures))

    # waveform increment
    d_wf = float(wb.get("delta_waveform_point", math.nan))
    ci = wb["waveform_plus_scalar_vs_scalar_pairwise"]["delta_f1@0.5"]["ci95"]
    checks.append(check("waveform_delta_lt_0.001", 1.0 if abs(d_wf) < 0.001 else 0.0, 1.0, 0, failures))
    if not (ci[0] < 0 < ci[1] or abs(d_wf) < 0.001):
        # CI crosses 0 is required narrative; allow if crosses
        if not (ci[0] < 0 < ci[1]):
            failures.append({"name": "waveform_ci_crosses_0_or_immaterial", "ci": ci, "delta": d_wf, "ok": False})

    # coverage
    cov = cfin["coverage"]
    checks.append(check("n_traces", cov["n_traces"], 43090, 0, failures))
    checks.append(check("n_events", cov["n_events"], 2669, 0, failures))
    checks.append(check("n_ge2", cov["n_with_2plus_candidates"], 27560, 0, failures))

    method_lock = json.loads(method_lock_p.read_text())
    if abs(float(method_lock["calibration_tau"]) - 0.5) > 1e-12:
        failures.append({"name": "tau", "observed": method_lock["calibration_tau"], "ok": False})

    audit = {
        "n_checks": len(checks),
        "n_failures": len(failures),
        "checks": checks,
        "failures": failures,
        "passed": len(failures) == 0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(audit, OUT / "final_evidence_audit.json")
    (REPORT / "final_evidence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if failures:
        (OUT / "FINAL_EVIDENCE.BLOCKED").write_text("blocked\n", encoding="utf-8")
        save_json({"failures": failures}, OUT / "FINAL_EVIDENCE.BLOCKED.json")
        print("FINAL_EVIDENCE.BLOCKED", len(failures), "failures")
        for f in failures:
            print(f)
        return 2

    # LOCK
    artifacts = {
        "fulldev_method_lock": str(method_lock_p),
        "confirm_execution_lock": str(confirm_lock_p),
        "confirm_fixed_metrics": str(confirm_fixed),
        "confirm_resid_metrics": str(confirm_resid),
        "confirm_scalar_metrics": str(confirm_scalar),
        "confirm_bootstrap_vs_fixed": str(boot_sf),
        "confirm_bootstrap_vs_resid": str(boot_sr),
        "confirm_final": str(confirm_final),
        "fulldev_final": str(fulldev_final),
        "pilot_ablation": str(pilot_ablation),
        "ranking_gap_rank_distribution": str(rank_dist),
        "residual_control_audit": str(resid_audit),
        "waveform_bootstrap": str(wf_boot),
        "scalar_checkpoint": str(ckpt),
        "metrics_py": str(metrics_py),
        "effect_replication_csv": str(effect_csv),
        "SPLIT.LOCK": str(artifacts_dir() / "results/pairwise_pilot/SPLIT.LOCK.json"),
        "phaseB_manifest": str(artifacts_dir() / "results/stage6/phaseB_eval_manifest.csv"),
        "confirm_manifest": str(artifacts_dir() / "results/stage6/final_confirm/confirm_s_eval_manifest.csv"),
    }
    hashes = {k: sha_file(Path(v)) for k, v in artifacts.items()}
    feature_schema = json.dumps(method_lock.get("scalar_feature_names", []), separators=(",", ":")).encode()
    lock = {
        "declaration": "All primary method development ended before final evidence packaging.",
        "primary_method": "scalar_pairwise",
        "task": "catalog-assisted S-phase candidate reranking / re-picking",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "fulldev_method_lock_hash": EXPECTED_METHOD_LOCK,
        "confirm_execution_lock_hash": EXPECTED_CONFIRM_EXEC,
        "tau": 0.50,
        "feature_schema_hash": hashlib.sha256(feature_schema).hexdigest(),
        "metric_implementation_sha256": hashes["metrics_py"],
        "scalar_checkpoint_sha256": hashes["scalar_checkpoint"],
        "confirm_result_status": "CONFIRMED",
        "confirm_final_sha256": hashes["confirm_final"],
        "artifact_sha256": hashes,
        "no_new_experiments": True,
        "audit_passed": True,
    }
    body = json.dumps(lock, sort_keys=True, indent=2)
    digest = hashlib.sha256(body.encode()).hexdigest()
    lock["final_evidence_lock_body_sha256"] = digest
    save_json(lock, OUT / "FINAL_EVIDENCE_LOCK.json")
    (OUT / "FINAL_EVIDENCE_LOCK.sha256").write_text(digest + "\n", encoding="utf-8")
    (OUT / "FINAL_EVIDENCE.LOCKED").write_text("locked\n", encoding="utf-8")
    print("FINAL_EVIDENCE.LOCKED", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
