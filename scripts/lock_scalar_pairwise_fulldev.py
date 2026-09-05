#!/usr/bin/env python
"""Freeze scalar_pairwise method lock BEFORE phaseB full-dev evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.pairwise.fulldev import FIXED_SCORE, RESID_CONTROL, SCALAR_FEATURE_NAMES, TAU
from earthquake.pairwise.guards import assert_no_confirm_path
from earthquake.pairwise.model import PairwiseScorer
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_fulldev"
PILOT = artifacts_dir() / "results" / "pairwise_pilot"


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
    # provenance must remain
    for p in [
        PILOT / "PAIRWISE.PILOT.FAILED",
        PILOT / "PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX",
        PILOT / "PAIRWISE.PILOT.FAILED_AFTER_RESUME",
    ]:
        if not p.exists():
            raise SystemExit(f"missing provenance marker: {p}")

    ckpt = PILOT / "ckpt_scalar_pairwise_seed42.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    split_lock = PILOT / "SPLIT.LOCK.json"
    phaseb_man = artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv"
    enriched = artifacts_dir() / "results" / "multistation_moveout" / "phaseB_enriched_candidate_table.parquet"
    metrics_py = ROOT / "src" / "earthquake" / "metrics.py"
    stage6_lock = artifacts_dir() / "results" / "stage6" / "method_lock_stage6.json"

    # confirm must not be used as inputs
    for p in [ckpt, split_lock, phaseb_man, enriched]:
        assert_no_confirm_path(p)

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if blob.get("use_waveform", False):
        raise SystemExit("checkpoint is not scalar-only")
    model = PairwiseScorer(int(blob.get("n_scalar", 10)), use_waveform=False, use_scalar=True)
    model.load_state_dict(blob["state_dict"])
    n_params = model.n_parameters()

    split = json.loads(split_lock.read_text())
    lock = {
        "method": "scalar_pairwise",
        "locked_before_phaseB_full_dev_evaluation": True,
        "declaration": "locked before phaseB full-dev evaluation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git(),
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": _sha_file(ckpt),
        "architecture": {
            "class": "PairwiseScorer",
            "use_waveform": False,
            "use_scalar": True,
            "n_scalar": int(blob.get("n_scalar", 10)),
            "head": "Linear(n_scalar,64)->ReLU->Linear(64,1)",
            "pairwise": "P(c2>c1)=sigmoid(s2-s1)",
        },
        "n_parameters": n_params,
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "feature_preprocessing": {
            "resid_nan_to_0": True,
            "prob_nan_to_0": True,
            "source_one_hot": ["stead", "ida", "stead+ida"],
            "no_trace_norm": True,
            "no_waveform": True,
        },
        "candidate_source": "UNION_STEAD5_IDA5 via phaseB enriched candidate table",
        "c1_c2_definition": {
            "c1": "argmax fixed_score",
            "c2": "second by fixed_score",
            "tie_break": "candidate_index ascending",
            "forbidden": ["true_S", "true_P", "STEAD_rank", "IDA_rank", "fulldev_label"],
            "lt2_candidates": "prediction=c1",
        },
        "fixed_score": FIXED_SCORE,
        "calibration_tau": TAU,
        "tau_note": "frozen from pilot calibration; full-dev must not re-tune",
        "training_split_event_sha256": split["event_sha256"]["train"],
        "calibration_split_event_sha256": split["event_sha256"]["calibration"],
        "pilot_heldout_event_sha256": split["event_sha256"]["heldout_eval"],
        "split_lock_path": str(split_lock),
        "split_lock_sha256": _sha_file(split_lock),
        "phaseB_manifest_path": str(phaseb_man),
        "phaseB_manifest_sha256": _sha_file(phaseb_man),
        "phaseB_enriched_candidates_path": str(enriched),
        "phaseB_enriched_sha256": _sha_file(enriched),
        "metric_implementation_path": str(metrics_py),
        "metric_implementation_sha256": _sha_file(metrics_py),
        "residual_control": RESID_CONTROL,
        "stage6_method_lock_sha256": _sha_file(stage6_lock) if stage6_lock.exists() else None,
        "pilot_heldout_reference": {
            "fixed_f1@0.5": 0.9196,
            "resid_f1@0.5": 0.9350,
            "scalar_f1@0.5": 0.9403,
            "delta_vs_fixed": 0.0207,
        },
        "forbidden_in_fulldev": [
            "waveform",
            "recalibrate_tau",
            "confirm",
            "retune_resid_kernel",
            "modify_c1_c2",
            "new_candidates",
            "abstain",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "provenance_pairwise_pilot_markers_preserved": [
            "PAIRWISE.PILOT.FAILED",
            "PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX",
            "PAIRWISE.PILOT.FAILED_AFTER_RESUME",
        ],
    }
    body = json.dumps(lock, sort_keys=True, indent=2)
    digest = hashlib.sha256(body.encode()).hexdigest()
    lock["lock_body_sha256"] = digest
    out_json = OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    save_json(lock, out_json)
    (OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256").write_text(digest + "\n", encoding="utf-8")
    # also write canonical body hash of file after save
    file_sha = _sha_file(out_json)
    (OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.file.sha256").write_text(file_sha + "\n", encoding="utf-8")
    print("LOCKED", out_json)
    print("body_sha256", digest)
    print("file_sha256", file_sha)
    print("tau", TAU, "n_params", n_params, "ckpt", lock["checkpoint_sha256"][:16])


if __name__ == "__main__":
    main()
