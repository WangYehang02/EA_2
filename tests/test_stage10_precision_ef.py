"""Precision ablation helpers and freeze/overflow policy tests. No confirm, no 30-epoch."""

from __future__ import annotations

from pathlib import Path

from earthquake.stage10.train_monitor import divergence_report
from earthquake.stage10.v3_continue import GATE_KEYS, OVERFLOW_FLAG, resume_blocked_nonfinite_e3


def test_divergence_exponential_activation():
    series = [{"act_abs_max": 2.0 ** i, "param_abs_max": 0.1, "loss": 0.02, "finite": True} for i in range(10)]
    r = divergence_report(series, interval_steps=100)
    assert r["finite_throughout"] is True
    assert r["exponential_activation"] is True
    assert r["stable"] is False


def test_divergence_stable_flat_loss():
    series = [{"act_abs_max": 12.0 + (i % 3), "param_abs_max": 0.4, "loss": 0.02, "finite": True} for i in range(10)]
    r = divergence_report(series, interval_steps=100)
    assert r["stable"] is True
    assert r["exponential_activation"] is False
    assert r["exponential_loss"] is False


def test_overflow_flag_blocks_even_if_gate_all_true(tmp_path: Path):
    (tmp_path / "TRAIN.FAILED_NONFINITE_E3").write_text("flag\n")
    gdir = tmp_path / "forensic_nonfinite_e3"
    gdir.mkdir()
    (gdir / "RESUME.GATE.json").write_text(__import__("json").dumps({k: True for k in GATE_KEYS}))
    assert resume_blocked_nonfinite_e3(tmp_path) is None
    (tmp_path / OVERFLOW_FLAG).write_text("failed\n")
    assert resume_blocked_nonfinite_e3(tmp_path) == "V3.FAILED_FP16_ACTIVATION_OVERFLOW_do_not_resume_epoch2"


def test_ef_script_no_gradscaler_and_separate_dir():
    src = Path("scripts/smoke_stage10_v3_precision_ef.py").read_text()
    assert "GradScaler" not in src
    assert "torch.bfloat16" in src
    assert "smoke_ef" in src
    assert "SMOKE.FAILED.json" in src  # plan-local only
    assert "not_for_resume" in src
    assert "v4_not_started" in src
    old = Path("scripts/smoke_stage10_v3_nonfinite_fix.py").read_text()
    assert "refusing_overwrite_frozen_SMOKE.FAILED" in old


def test_state_finite_tiny_model():
    import torch
    from earthquake.stage10.state_finite import audit_model_opt

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(8, 4)
    opt.zero_grad(set_to_none=True)
    a = audit_model_opt(m, opt)
    assert a["gradients"]["gradients_all_none"] is True
    m(x).sum().backward()
    opt.step()
    b = audit_model_opt(m, opt)
    assert b["parameters"]["finite"]
    assert b["bn_buffers"]["finite"]
    assert b["optimizer"]["finite"]
