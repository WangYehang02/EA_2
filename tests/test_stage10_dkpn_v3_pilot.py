"""v3 pilot locks: A-loss, 3-epoch hard stop, no v2 overwrite (no HDF5)."""

from __future__ import annotations

from pathlib import Path

import yaml

from earthquake.stage10.checkpoint_policy import is_valid_best
from earthquake.stage10.gpu_policy import launch_block_reason


def test_v3_config_locked():
    cfg = yaml.safe_load(Path("configs/stage10/dkpn_clean_v3_seed42_mixedcrop.yaml").read_text())
    assert cfg["run_name"] == "dkpn_clean_v3_seed42_mixedcrop"
    assert cfg["seed"] == 42
    assert cfg["init"] == "random"
    assert cfg["loss"] == "current_partial_label_nll"
    assert cfg["phase_balanced_loss"] is False
    assert cfg["never_auto_continue_30"] is True
    assert cfg["hard_stop_after_pilot"] is True
    assert cfg["pilot_virtual_epochs"] == 3
    assert cfg["official_height"] == 0.2
    assert cfg["never_substitute_calibrated_threshold"] is True
    assert cfg["confirm_read"] is False
    assert cfg["refuse_v2_checkpoint"] is True
    assert cfg["out_dir"].endswith("dkpn_clean_v3_seed42_mixedcrop")
    assert "dkpn_clean_v2" not in cfg["out_dir"]


def test_v3_script_hard_stop_and_a_loss_only():
    text = Path("scripts/train_stage10_dkpn_v3.py").read_text()
    assert 'choices=["pilot", "status", "resume"]' in text or "choices=['pilot', 'status', 'resume']" in text
    assert "max_ep = 3" in text
    assert "never_auto_continue_30" in text
    assert "v2h.batch_loss" in text
    assert "phase_balanced_partial_nll(" not in text
    assert 'val_cat["crop_kind"] = "s_centered"' in text
    assert "V2_DIR" in text
    assert "resume requires PILOT.PASSED" in text


def test_v3_best_metric_rules():
    assert is_valid_best(metric=0.0, init_metric=-1.0) is False
    assert is_valid_best(metric=0.12, init_metric=0.12) is False
    assert is_valid_best(metric=0.20, init_metric=0.12) is True


def test_v3_launch_still_needs_4_idle_and_refuses_if_running():
    assert launch_block_reason(mode="pilot", n_idle=3, already_training=False, train_running_flag=False) == "need_min_gpus_4_have_3"
    assert launch_block_reason(mode="pilot", n_idle=6, already_training=True, train_running_flag=False) == "training_already_running"
