"""v4 full-dev stop gate: lock, A/B/C, grouping, confirm isolation. No training."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from earthquake.stage10.dkpn_picks import extract_picks
from earthquake.stage10.full_dev_annotate import IN_SAMPLES, OVERLAP, stack_avg, window_offsets
from earthquake.stage10.full_dev_stop_gate import (
    DELTA_F1,
    DISTANCE_BINS_KM,
    HEIGHT,
    PS_INTERVAL_BINS,
    SNR_BINS_DB,
    STRONGEST_WAVEFORM_ONLY,
    WAVEFORM_ONLY_DEV_F1_LOCKED,
    assign_bin,
    merge_union_dkpn5,
    stop_gates,
)
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT

ROOT = Path(__file__).resolve().parents[1]


def test_height_locked_0p2():
    assert HEIGHT == 0.2 == OFFICIAL_HEIGHT
    sig = inspect.signature(extract_picks)
    assert sig.parameters["thr"].default == 0.2
    assert sig.parameters["min_distance"].default == 50


def test_window_offsets_match_seisbench_12000():
    off = window_offsets(12000)
    assert off[0] == 0
    assert int(off[-1]) == 12000 - IN_SAMPLES
    stride = IN_SAMPLES - OVERLAP
    assert stride == 1501


def test_stack_avg_overlap():
    n = 20
    in_s = 10
    a = np.ones(in_s)
    b = np.full(in_s, 3.0)
    out = stack_avg(np.stack([a, b]), np.array([0, 10]), n, in_samples=in_s)
    assert out.shape == (n,)
    assert np.allclose(out[:10], 1.0)
    assert np.allclose(out[10:], 3.0)


def test_strongest_waveform_only_is_stead_from_locked_dev():
    assert STRONGEST_WAVEFORM_ONLY == "STEAD_top1"
    vals = WAVEFORM_ONLY_DEV_F1_LOCKED
    assert vals["STEAD_top1"] == max(vals.values())


def _m(**kw):
    base = {"f1@0.5": 0.5, "miss_rate": 0.1, "detected_ae_p95": 2.0}
    base.update(kw)
    return base


def test_gate_a_primary_candidate():
    g = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.86}),
        baseline=_m(**{"f1@0.5": 0.84}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.89},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": -0.01},
    )
    assert g["gates"]["A_vs_waveform_only_delta_f1_0.5_ge_0.01"] is True
    assert g["verdict"] == "keep_as_primary_picker_candidate"


def test_gate_b_requires_miss_and_p95():
    g = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.84, "detected_ae_p95": 1.8, "miss_rate": 0.1}),
        baseline=_m(**{"f1@0.5": 0.84, "detected_ae_p95": 2.0, "miss_rate": 0.1}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.89},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": -1},
    )
    assert g["gates"]["B_f1_not_down_and_p95_ge_0.15s_and_miss_not_worse"] is True
    g2 = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.84, "detected_ae_p95": 1.8, "miss_rate": 0.2}),
        baseline=_m(**{"f1@0.5": 0.84, "detected_ae_p95": 2.0, "miss_rate": 0.1}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.89},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": -1},
    )
    assert g2["gates"]["B_f1_not_down_and_p95_ge_0.15s_and_miss_not_worse"] is False


def test_gate_c_union_only_verdict():
    g = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.80}),
        baseline=_m(**{"f1@0.5": 0.84}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.91},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": 0.002},
    )
    assert g["gates"]["A_vs_waveform_only_delta_f1_0.5_ge_0.01"] is False
    assert g["gates"]["C_union_oracle_delta_f1_0.5_ge_0.01_and_ci_lo_gt_0"] is True
    assert g["verdict"] == "keep_as_union_candidate_only"


def test_gate_c_requires_ci_and_delta():
    g = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.80}),
        baseline=_m(**{"f1@0.5": 0.84}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.91},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": -0.001},
    )
    assert g["verdict"] == "rejected_candidate_source"


def test_margin_not_lowered():
    assert DELTA_F1 == 0.01
    g = stop_gates(
        dkpn=_m(**{"f1@0.5": 0.845}),
        baseline=_m(**{"f1@0.5": 0.84}),
        union_oracle={"f1@0.5": 0.89},
        union_plus_dkpn_oracle={"f1@0.5": 0.89},
        boot_dkpn_minus_baseline={},
        boot_union_plus_minus_union={"ci95_lo": -1},
    )
    assert g["gates"]["A_vs_waveform_only_delta_f1_0.5_ge_0.01"] is False
    assert g["did_not_lower_threshold_because_pilot_looked_good"] is True


def test_grouping_bins_include_long_ps_and_existing_distance():
    from earthquake.stage10.crop_v2 import LONG_PS_S as CROP_PS

    assert CROP_PS == 30.0
    assert any(b[0] == "ps_ge_30s" and b[1] == 30.0 for b in PS_INTERVAL_BINS)
    assert assign_bin(35.0, PS_INTERVAL_BINS) == "ps_ge_30s"
    assert assign_bin(25.0, DISTANCE_BINS_KM) == "dist_0_50"
    assert assign_bin(12.0, SNR_BINS_DB) == "snr_10_20"


def test_merge_union_keeps_union_adds_unmatched_dkpn():
    union = pd.DataFrame(
        {
            "trace_name": ["t", "t"],
            "event_id": ["e", "e"],
            "sampling_rate_hz": [100.0, 100.0],
            "candidate_sample": [1000.0, 2000.0],
        }
    )
    dkpn = pd.DataFrame(
        {
            "trace_name": ["t", "t"],
            "event_id": ["e", "e"],
            "sampling_rate_hz": [100.0, 100.0],
            "candidate_sample": [1002.0, 3500.0],
        }
    )
    m = merge_union_dkpn5(union, dkpn, dedup_s=0.05)
    samps = sorted(m["candidate_sample"].tolist())
    assert 1000.0 in samps and 2000.0 in samps and 3500.0 in samps
    assert 1002.0 not in samps


def test_scripts_do_not_train_or_mint_pilot_passed():
    run = (ROOT / "scripts/run_stage10_v4_full_dev_stop_gate.py").read_text()
    lock = (ROOT / "scripts/lock_stage10_v4_fulldev_stop_gate.py").read_text()
    assert "train_stage10_dkpn_v4" not in run
    assert "train_stage10_dkpn_v4" not in lock
    assert 'V4 / "PILOT.PASSED"' not in run
    assert "FULLDEV.STOP_GATE" in run
    assert "METHOD.LOCK" in lock
    assert "epoch_1.pt" in run
    assert "not_creating_v5" in lock


def test_confirm_metrics_not_loaded_in_runner():
    run = (ROOT / "scripts/run_stage10_v4_full_dev_stop_gate.py").read_text()
    assert "confirm_method_metrics.json" in run
    assert "json.loads" not in run or "confirm_method_metrics" not in run.split("confirm_method_metrics.json")[1][:200]
    assert "pd.read_parquet(CONFIRM_PRED)" not in run
    assert "load_json(CONFIRM_METRICS)" not in run
    assert "confirm_atime" in run
