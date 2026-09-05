#!/usr/bin/env python
"""Create CONFIRM_EXECUTION_LOCK referencing frozen FULLDEV method lock (no new method)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.pairwise.fulldev import FIXED_SCORE, RESID_CONTROL, SCALAR_FEATURE_NAMES, TAU
from earthquake.utils import ensure_dir

EXPECTED_METHOD_LOCK_HASH = "1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b"
OUT = artifacts_dir() / "results" / "pairwise_confirm"
FULLDEV = artifacts_dir() / "results" / "pairwise_fulldev"
FC = artifacts_dir() / "results" / "stage6" / "final_confirm"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ensure_dir(OUT)
    if not (FULLDEV / "SCALAR_PAIRWISE.FULLDEV.PASSED").exists():
        raise SystemExit("FULLDEV.PASSED missing — refuse confirm")
    method_lock_path = FULLDEV / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    method_sha_path = FULLDEV / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256"
    got = method_sha_path.read_text().strip()
    if got != EXPECTED_METHOD_LOCK_HASH:
        raise SystemExit(f"METHOD LOCK HASH mismatch: got {got} expected {EXPECTED_METHOD_LOCK_HASH}")
    method = json.loads(method_lock_path.read_text())
    if method.get("lock_body_sha256") != EXPECTED_METHOD_LOCK_HASH:
        raise SystemExit("method lock body hash mismatch")
    if abs(float(method["calibration_tau"]) - 0.50) > 1e-12:
        raise SystemExit("tau != 0.50 in method lock")

    ckpt = Path(method["checkpoint_path"])
    if _sha_file(ckpt) != method["checkpoint_sha256"]:
        raise SystemExit("checkpoint sha mismatch vs method lock")

    man = FC / "confirm_s_eval_manifest.csv"
    union = FC / "confirm_union.parquet"
    hist = FC / "confirm_history_features.parquet"
    frozen_preds = FC / "confirm_predictions.parquet"
    metrics_py = ROOT / "src" / "earthquake" / "metrics.py"
    feature_schema = json.dumps(SCALAR_FEATURE_NAMES, separators=(",", ":")).encode()

    lock = {
        "confirm_is_evaluation_only": True,
        "declaration": "confirm is evaluation-only; no parameter selection is allowed",
        "method_definition_source": "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json",
        "method_lock_hash": EXPECTED_METHOD_LOCK_HASH,
        "method_lock_path": str(method_lock_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git(),
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": method["checkpoint_sha256"],
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "scalar_feature_schema_hash": hashlib.sha256(feature_schema).hexdigest(),
        "c1_c2_definition": method["c1_c2_definition"],
        "tau": TAU,
        "fixed_score": FIXED_SCORE,
        "residual_control": RESID_CONTROL,
        "metric_implementation_path": str(metrics_py),
        "metric_implementation_sha256": _sha_file(metrics_py),
        "confirm_manifest_path": str(man),
        "confirm_manifest_sha256": _sha_file(man),
        "confirm_union_path": str(union),
        "confirm_union_sha256": _sha_file(union),
        "confirm_history_path": str(hist),
        "confirm_history_sha256": _sha_file(hist),
        "confirm_frozen_predictions_path": str(frozen_preds),
        "confirm_frozen_predictions_sha256": _sha_file(frozen_preds),
        "confirm_artifact_provenance": {
            "stage6_final_confirm_dir": str(FC),
            "CONFIRM.CONSUMED_present": (FC / "CONFIRM.CONSUMED").exists(),
            "note": "Stage-6 historical confirm already consumed for fixed_rescore_UNION; this lock is a separate pairwise confirm evaluation reading frozen artifacts only",
        },
        "scoring_implementation": "stage6 attach_expected_s + rescore_phase_candidates (locked lambdas)",
        "bootstrap": {"n": 5000, "seed": 20260817, "unit": "event"},
        "forbidden": [
            "retrain",
            "recalibrate_tau",
            "retune_resid",
            "change_features",
            "change_c1_c2",
            "waveform",
            "auto_next_stage",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "fulldev_reference_preregistered": {
            "fixed_f1@0.5": 0.8668,
            "resid_f1@0.5": 0.8732,
            "scalar_f1@0.5": 0.8805,
            "delta_fixed": 0.0137,
            "delta_resid": 0.0073,
        },
    }
    body = json.dumps(lock, sort_keys=True, indent=2)
    digest = hashlib.sha256(body.encode()).hexdigest()
    lock["execution_lock_body_sha256"] = digest
    save_json(lock, OUT / "SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.json")
    (OUT / "SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.sha256").write_text(digest + "\n", encoding="utf-8")
    print("CONFIRM_EXECUTION_LOCK", digest)
    print("method_lock_hash", EXPECTED_METHOD_LOCK_HASH)
    print("tau", TAU, "manifest_sha", lock["confirm_manifest_sha256"][:16])


if __name__ == "__main__":
    main()
