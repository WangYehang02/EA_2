#!/usr/bin/env python
"""Write METHOD.LOCK for v4 Stage-6 full-dev stop gate BEFORE inference.

Does not train, does not read confirm waveforms/metrics, does not overwrite
PILOT.FAILED / PILOT.ADJUDICATION.FAILED / protocol amendment.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage10.dkpn_picks import extract_picks
from earthquake.stage10.fp32_guard import state_dict_tensor_sha256
from earthquake.stage10.full_dev_annotate import (
    BLINDING,
    DKPN_K10,
    DKPN_K5,
    EXTRACT_MIN_DISTANCE,
    EXTRACT_SMOOTH,
    OVERLAP,
    STACKING,
    STRIDE,
    IN_SAMPLES,
)
from earthquake.stage10.full_dev_stop_gate import (
    BOOT_SEED,
    DELTA_F1,
    DISTANCE_BINS_KM,
    HEIGHT,
    MARK,
    N_BOOT,
    NOISE_N,
    P95_IMPROVE_S,
    PS_INTERVAL_BINS,
    SNR_BINS_DB,
    STRONGEST_WAVEFORM_ONLY,
    V4,
    WAVEFORM_ONLY_DEV_F1_LOCKED,
    sha256_file,
)
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT
from earthquake.utils import ensure_dir

CONFIRM_PRED = ROOT / "artifacts/results/stage6/final_confirm/confirm_predictions.parquet"
CONFIRM_METRICS = ROOT / "artifacts/results/stage6/final_confirm/confirm_method_metrics.json"
CONFIRM_MAN = ROOT / "artifacts/results/stage6/final_confirm/confirm_s_eval_manifest.csv"
PILOT_FAILED = V4 / "PILOT.FAILED"
PILOT_ADJ = V4 / "PILOT.ADJUDICATION.FAILED"
AMENDMENT = ROOT / "configs/stage10/protocol_amendment_20260902_noninferiority.json"
FAILED_SHA = "b544aba50775be366bd0d49f23c5e3c3f6f00120f1d54f0c4871f9f0f7994e67"


def _stat_no_open(p: Path) -> dict | None:
    if not p.exists():
        return None
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime": st.st_mtime, "atime": st.st_atime, "opened": False}


def _ckpt_info(path: Path) -> dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "param_sha256": state_dict_tensor_sha256(sd),
        "epoch": None if not isinstance(ck, dict) else ck.get("epoch"),
        "virtual_epoch": None if not isinstance(ck, dict) else ck.get("virtual_epoch"),
        "val_s_f1_fixed0p2": None if not isinstance(ck, dict) else ck.get("val_s_f1_fixed0p2"),
        "precision_mode": None if not isinstance(ck, dict) else ck.get("precision_mode"),
    }


def main() -> int:
    if os.access(PILOT_FAILED, os.W_OK) or sha256_file(PILOT_FAILED) != FAILED_SHA:
        raise SystemExit("PILOT.FAILED must remain frozen")
    if not PILOT_ADJ.is_file():
        raise SystemExit("PILOT.ADJUDICATION.FAILED missing")
    if not AMENDMENT.is_file():
        raise SystemExit("protocol amendment missing")

    art = artifacts_dir()
    man_csv = art / "results" / "stage6" / "phaseB_eval_manifest.csv"
    man_json = art / "results" / "stage6" / "phaseB_eval_manifest.json"
    schema = art / "results" / "stage6" / "phaseC" / "candidate_schema.json"
    union_pq = art / "cache" / "stage6" / "phaseC" / "dev_union.parquet"
    stead_pq = art / "cache" / "stage6" / "phaseB" / "stead_top10" / "stead_top10.parquet"
    ida_pq = art / "cache" / "stage6" / "phaseB" / "ida_top10" / "ida_top10.parquet"
    pred_dir = art / "results" / "stage6" / "phaseC" / "baseline_preds"
    ck = V4 / "checkpoints"

    epoch0 = _ckpt_info(ck / "epoch_0.pt")
    epoch1 = _ckpt_info(ck / "epoch_1.pt")
    epoch2 = _ckpt_info(ck / "epoch_2.pt")
    best_metric_file = _ckpt_info(ck / "best_metric.pt")
    best_loss = _ckpt_info(ck / "best_loss.pt")
    last = _ckpt_info(ck / "last.pt")

    if epoch1["epoch"] != 1:
        raise SystemExit(f"epoch_1.pt epoch={epoch1['epoch']}")
    if best_metric_file["param_sha256"] != epoch2["param_sha256"]:
        raise SystemExit("expected best_metric.pt weights to match epoch_2 (overwrite bug)")
    if epoch1["param_sha256"] == epoch2["param_sha256"]:
        raise SystemExit("epoch_1 and epoch_2 weights unexpectedly identical")

    extract_src = inspect.getsource(extract_picks)
    lock = {
        "marker": MARK,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "postmortem candidate-value eval of frozen v4; not PILOT.PASSED",
        "not_training": True,
        "not_creating_v5": True,
        "not_rerunning_seed42": True,
        "not_starting_other_seeds": True,
        "not_changing_height": True,
        "not_changing_method_from_full_dev": True,
        "confirm_waveforms_read": False,
        "confirm_metrics_read": False,
        "retained": {
            "PILOT.FAILED": {"path": str(PILOT_FAILED), "sha256": sha256_file(PILOT_FAILED), "writable": False},
            "PILOT.ADJUDICATION.FAILED": {"path": str(PILOT_ADJ), "sha256": sha256_file(PILOT_ADJ)},
            "protocol_amendment": {"path": str(AMENDMENT), "sha256": sha256_file(AMENDMENT)},
        },
        "precision": "fp32",
        "amp": False,
        "autocast": False,
        "grad_scaler": False,
        "height": HEIGHT,
        "official_height": OFFICIAL_HEIGHT,
        "extract_picks": {
            "thr": HEIGHT,
            "min_distance": EXTRACT_MIN_DISTANCE,
            "smooth": EXTRACT_SMOOTH,
            "source": "src/earthquake/stage10/dkpn_picks.py",
            "source_sha256": sha256_file(ROOT / "src/earthquake/stage10/dkpn_picks.py"),
            "signature": str(inspect.signature(extract_picks)),
            "src_sha256": hashlib.sha256(extract_src.encode()).hexdigest(),
        },
        "inference": {
            "kind": "official_dkpn_seisbench_annotate",
            "in_samples": IN_SAMPLES,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "stacking": STACKING,
            "blinding": list(BLINDING),
            "linear_detrend_before_cf": True,
            "window_std": "annotate_window_pre",
            "not_s_centered": True,
            "does_not_use_human_s_to_place_window": True,
        },
        "full_dev": {
            "source": "artifacts/results/stage6/phaseB_eval_manifest.csv",
            "n_traces": 87293,
            "n_events": 5341,
            "csv_sha256": sha256_file(man_csv),
            "json_sha256": sha256_file(man_json),
            "expected_csv_sha256": "2cc1e7a3376525ed46ff12d182f947602714a5b353e651e4d59542b56894c4e7",
            "trace_list_sha256": "1b44d8c83987bad802f6cecab47d5967730317003aafbb7c80f0e4cfbd45914e",
            "event_list_sha256": "f5428229a490bbdda0769005f8824e2fdf8307086338eb0ea1ff760ca4e48260",
            "confirm_excluded_at_phaseB": True,
        },
        "baseline": {
            "strongest_waveform_only": STRONGEST_WAVEFORM_ONLY,
            "selection_source": "existing Stage6/7/9 DEV metrics locked before this run",
            "dev_f1@0.5": WAVEFORM_ONLY_DEV_F1_LOCKED,
            "not_from_confirm": True,
            "stead_top1_npy_sha256": sha256_file(pred_dir / "STEAD_top1.npy"),
            "fixed_rescore_UNION_npy_sha256": sha256_file(pred_dir / "fixed_rescore_UNION.npy"),
            "ida_top1_npy_sha256": sha256_file(pred_dir / "IDA_top1.npy"),
        },
        "UNION": {
            "name": "UNION_STEAD5_IDA5",
            "schema_path": str(schema),
            "schema_sha256": sha256_file(schema),
            "dev_union_sha256": sha256_file(union_pq),
            "stead_top10_sha256": sha256_file(stead_pq),
            "ida_top10_sha256": sha256_file(ida_pq),
            "stead_k": 5,
            "ida_k": 5,
            "max_union": 10,
            "dedup_s": 0.05,
            "dkpn_k": DKPN_K5,
            "dkpn_k10": DKPN_K10,
        },
        "metric_impl": {
            "phaseC1_metrics_sha256": sha256_file(ROOT / "src/earthquake/stage6/phaseC1/metrics.py"),
            "phaseB_sha256": sha256_file(ROOT / "src/earthquake/stage6/phaseB.py"),
            "annotate_sha256": sha256_file(ROOT / "src/earthquake/stage10/full_dev_annotate.py"),
            "stop_gate_sha256": sha256_file(ROOT / "src/earthquake/stage10/full_dev_stop_gate.py"),
            "run_script_sha256": sha256_file(ROOT / "scripts/run_stage10_v4_full_dev_stop_gate.py"),
            "lock_script_sha256": sha256_file(ROOT / "scripts/lock_stage10_v4_fulldev_stop_gate.py"),
        },
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "event"},
        "noise_fpr": {"n": NOISE_N, "seed": 42, "height": HEIGHT},
        "stop_gate": {
            "A": "DKPN top1 vs strongest waveform-only ΔF1@0.5 >= +0.01",
            "B": "F1 not down AND detected AE P95 improve >= 0.15s AND miss not worse",
            "C": "UNION+DKPN5 oracle ΔF1@0.5 >= +0.01 AND event-bootstrap CI lo > 0",
            "margin_f1": DELTA_F1,
            "p95_improve_s": P95_IMPROVE_S,
            "not_lowered_because_pilot_looked_good": True,
        },
        "grouping": {
            "ps_interval_s": [list(x) for x in PS_INTERVAL_BINS],
            "distance_km": [list(x) for x in DISTANCE_BINS_KM],
            "snr_db": [list(x) for x in SNR_BINS_DB],
            "station": "all stations in frozen manifest; full CSV; no post-hoc subset",
            "channel_prefix": "all values in frozen manifest",
            "network": "all values in frozen manifest",
            "no_result_selected_groups": True,
        },
        "checkpoints": {
            "primary": {
                "role": "primary",
                "user_specified": "best_metric.pt = epoch1",
                "path": epoch1["path"],
                "file_sha256": epoch1["file_sha256"],
                "param_sha256": epoch1["param_sha256"],
                "epoch": 1,
                "why": "true max val F1 is epoch_1; file best_metric.pt was overwritten on every valid epoch so it stores epoch2 weights",
            },
            "best_metric_pt_file_is_not_epoch1": {
                "path": best_metric_file["path"],
                "epoch": best_metric_file["epoch"],
                "param_sha256": best_metric_file["param_sha256"],
                "matches": "epoch_2.pt / last.pt / best_loss.pt",
                "role": "secondary_reporting_only",
            },
            "secondary": {
                "epoch_0": epoch0,
                "epoch_2": epoch2,
                "best_loss": best_loss,
                "last": last,
                "best_metric_pt_file": best_metric_file,
            },
            "cannot_replace_primary_after_seeing_full_dev": True,
        },
        "confirm_unopened": {
            "predictions": _stat_no_open(CONFIRM_PRED),
            "metrics": _stat_no_open(CONFIRM_METRICS),
            "manifest": _stat_no_open(CONFIRM_MAN),
        },
        "gpu_policy": {"never_use_vllm_gpu2": True, "never_preempt": True},
    }
    if lock["full_dev"]["csv_sha256"] != lock["full_dev"]["expected_csv_sha256"]:
        raise SystemExit("full-dev manifest hash mismatch")

    out = ensure_dir(V4 / "full_dev_stop_gate")
    lock_path = out / "METHOD.LOCK.json"
    if lock_path.is_file():
        raise SystemExit("METHOD.LOCK.json already exists; refusing overwrite")
    save_json(lock, lock_path)
    os.chmod(lock_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    marker = V4 / MARK
    marker.write_text(json.dumps({"created_utc": lock["created_utc"], "lock": str(lock_path)}, indent=2) + "\n")
    print(json.dumps({"lock": str(lock_path), "primary_param": epoch1["param_sha256"], "confirm_metrics_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
