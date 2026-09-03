"""v3 exact-resume audit: optimizer required; hashes; oscillation tags."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from earthquake.stage10.gpu_policy import launch_block_reason, training_process_running
from earthquake.stage10.v3_continue import consecutive_collapse_fail, tag_oscillation
from earthquake.stage10.v3_resume_audit import audit_resume_checkpoint


def test_weights_only_is_blocked(tmp_path: Path | None = None):
    d = Path(tempfile.mkdtemp())
    p = d / "w.pt"
    torch.save({"model": {"a": torch.zeros(1)}, "epoch": 2, "virtual_epoch": 2, "optimizer_steps": 10}, p)
    a = audit_resume_checkpoint(p)
    assert a["block"] is True
    assert a["has_optimizer"] is False


def test_optimizer_present_not_blocked(tmp_path: Path | None = None):
    d = Path(tempfile.mkdtemp())
    p = d / "ok.pt"
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    x = torch.ones(3, 2)
    opt.zero_grad()
    m(x).sum().backward()
    opt.step()
    torch.save(
        {
            "model": m.state_dict(),
            "opt": opt.state_dict(),
            "epoch": 2,
            "virtual_epoch": 2,
            "optimizer_steps": 1,
        },
        p,
    )
    a = audit_resume_checkpoint(p, expect_epoch=2)
    assert a["block"] is False
    assert a["has_optimizer"] is True
    assert a["has_scheduler"] is False
    assert a["amp_scaler_used_in_pilot"] is False


def test_oscillation_tags():
    prev = {"val_s_f1_fixed0p2": 0.11, "oracle_acc@0.5s": 0.14}
    assert tag_oscillation(prev, 0.03, 0.29) == "confidence_oscillation"
    assert tag_oscillation(prev, 0.03, 0.05) == "representation_regression"
    assert tag_oscillation(prev, 0.30, 0.29) is None


def test_consecutive_collapse_requires_two_epochs_and_oracle_drop():
    h = [
        {"frac_no_s_peak_height0p2": 0.1, "val_s_f1_fixed0p2": 0.3, "oracle_acc@0.5s": 0.3},
        {"frac_no_s_peak_height0p2": 0.99, "val_s_f1_fixed0p2": 0.01, "oracle_acc@0.5s": 0.20},
        {"frac_no_s_peak_height0p2": 0.99, "val_s_f1_fixed0p2": 0.005, "oracle_acc@0.5s": 0.10},
    ]
    assert consecutive_collapse_fail(h) is True
    # single epoch like pilot epoch 1 must NOT trip
    h1 = [
        {"frac_no_s_peak_height0p2": 0.39, "val_s_f1_fixed0p2": 0.11, "oracle_acc@0.5s": 0.14},
        {"frac_no_s_peak_height0p2": 0.98, "val_s_f1_fixed0p2": 0.027, "oracle_acc@0.5s": 0.29},
    ]
    assert consecutive_collapse_fail(h1) is False


def test_allocatable_allows_display_leftover_not_python_jobs():
    from earthquake.stage10.gpu_policy import GpuSnap, allocatable_gpu_indices

    snaps = [
        GpuSnap(0, 143, 0, 1, compute_users=["ptyxis"], name="4090"),
        GpuSnap(1, 2598, 0, 1, compute_users=["python"], name="4090"),
        GpuSnap(2, 21038, 0, 1, compute_users=["VLLM::EngineCore"], name="4090"),
        GpuSnap(7, 18, 0, 0, compute_users=[], name="4090"),
    ]
    assert allocatable_gpu_indices(snaps) == [0, 7]


def test_batch_32_not_divisible_by_6_gpus():
    from earthquake.stage10.v3_continue import batch_divisible_by_ngpu

    assert batch_divisible_by_ngpu(32, 4) is True
    assert batch_divisible_by_ngpu(32, 6) is False
    assert batch_divisible_by_ngpu(48, 6) is True


def test_resume_blocked_by_nonfinite_e3_flag():
    from earthquake.stage10.v3_continue import resume_blocked_nonfinite_e3

    d = Path(tempfile.mkdtemp())
    assert resume_blocked_nonfinite_e3(d) is None
    (d / "TRAIN.FAILED_NONFINITE_E3").write_text("x")
    assert "gate_incomplete" in resume_blocked_nonfinite_e3(d)


def test_resume_launch_needs_4_gpus():
    assert launch_block_reason(mode="resume", n_idle=3, already_training=False, train_running_flag=False) == "need_min_gpus_4_have_3"
    assert launch_block_reason(mode="resume", n_idle=4, already_training=False, train_running_flag=False) is None


def test_training_process_running_pattern_includes_resume_token():
    import inspect

    src = inspect.getsource(training_process_running)
    assert "--mode resume" in src
