"""v4 FP32 pilot locks. No confirm, no 30-epoch, no v3 overwrite."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from earthquake.stage10.checkpoint_policy import MIN_DELTA_F1, is_valid_best
from earthquake.stage10.fp32_guard import PrecisionMismatch, assert_fp32_tensor, assert_no_autocast
from earthquake.stage10.gpu_policy import launch_block_reason


def test_v4_config_locked():
    cfg = yaml.safe_load(Path("configs/stage10/dkpn_clean_v4_seed42_fp32.yaml").read_text())
    assert cfg["run_name"] == "dkpn_clean_v4_seed42_fp32"
    assert cfg["seed"] == 42
    assert cfg["init"] == "random"
    assert cfg["precision_mode"] == "fp32"
    assert cfg["amp"] is False
    assert cfg["autocast"] is False
    assert cfg["grad_scaler"] is False
    assert cfg["never_auto_continue_30"] is True
    assert cfg["hard_stop_after_pilot"] is True
    assert cfg["pilot_virtual_epochs"] == 3
    assert cfg["max_epoch"] == 3
    assert cfg["batch_size"] == 32
    assert cfg["lr"] == 0.001
    assert cfg["refuse_v3_epoch2"] is True
    assert cfg["refuse_f_smoke_weights"] is True
    assert cfg["confirm_read"] is False
    assert cfg["out_dir"].endswith("dkpn_clean_v4_seed42_fp32")
    assert "dkpn_clean_v3" not in cfg["out_dir"]


def test_v4_script_hard_stop_no_resume_no_amp():
    text = Path("scripts/train_stage10_dkpn_v4.py").read_text()
    assert 'choices=["pilot", "status"]' in text
    assert "max_ep = 3" in text
    assert "never_auto_continue_30" in text
    assert "continued_to_30" in text
    assert "v2h.batch_loss" not in text
    assert "GradScaler" not in text
    assert "model.half(" not in text
    assert "model.bfloat16(" not in text
    assert "TRAIN_BLOCKED_PRECISION_MISMATCH" in text
    assert "epoch_2" in text  # refuse path
    assert "PILOT.PASSED" in text
    assert "wait_for_explicit_approval" in text


def test_v4_gate_requires_valid_best_and_no_skip():
    from earthquake.stage10.v4_pilot import pilot_gate_v4

    h = [
        {"train_loss": 0.02, "val_s_f1_fixed0p2": 0.05, "init_s_f1_fixed0p2": 0.02, "init_oracle_acc@0.5s": 0.1, "oracle_acc@0.5s": 0.12, "all_n_collapse": False, "grad_norm": 0.2},
        {"train_loss": 0.015, "val_s_f1_fixed0p2": 0.08, "init_s_f1_fixed0p2": 0.02, "init_oracle_acc@0.5s": 0.1, "oracle_acc@0.5s": 0.2, "all_n_collapse": False, "grad_norm": 0.2},
        {"train_loss": 0.01, "val_s_f1_fixed0p2": 0.09, "init_s_f1_fixed0p2": 0.02, "init_oracle_acc@0.5s": 0.1, "oracle_acc@0.5s": 0.25, "all_n_collapse": False, "grad_norm": 0.2},
    ]
    kinds = [{"p_centered", "s_centered", "background"}] * 3
    div = {"stable": True}
    ok, reasons = pilot_gate_v4(h, 0.05, kinds, 0, False, 0, True, div, False)
    assert ok, reasons
    ok2, r2 = pilot_gate_v4(h, 0.05, kinds, 0, False, 1, True, div, False)
    assert not ok2 and "silent_skip" in r2
    ok3, r3 = pilot_gate_v4(h, 0.05, kinds, 0, False, 0, False, div, False)
    assert not ok3 and "best_metric_not_above_init_plus_min_delta" in r3


def test_fp32_guard_rejects_half():
    x = torch.zeros(2, 3, dtype=torch.float16)
    try:
        assert_fp32_tensor(x, "x")
        raise AssertionError("should have raised")
    except PrecisionMismatch:
        pass
    assert_fp32_tensor(torch.zeros(2, 3), "ok")
    assert_no_autocast()


def test_v4_launch_needs_4_idle():
    assert launch_block_reason(mode="pilot", n_idle=3, already_training=False, train_running_flag=False) == "need_min_gpus_4_have_3"


def test_valid_best_min_delta():
    assert is_valid_best(metric=0.0, init_metric=-1.0) is False
    assert MIN_DELTA_F1 == 0.01
    assert is_valid_best(metric=0.04, init_metric=0.02) is True
