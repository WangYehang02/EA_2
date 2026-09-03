"""Eval must be FP32. epoch_2.pt + AMP FP16 eval reproduces Inf at BN.

No confirm, no 30-epoch train.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from earthquake.stage10.dkpn_clean import build_dkpn_random
from earthquake.stage10.eval_fp32 import eval_forward_amp_fp16_diagnostic, eval_forward_fp32, eval_loss_fp32, probs_fp32
from earthquake.stage10.finite_hooks import install_finite_hooks, remove_hooks
from earthquake.stage10.nonfinite import NonfiniteError, PHASES
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr

FORE = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop/forensic_nonfinite_e3")
CK = FORE / "epoch_2.copy.pt"
EVAL_BATCH = FORE / "eval_mode_false_positive_batch0.pt"


def test_nonfinite_error_requires_known_phase():
    with pytest.raises(ValueError, match="unknown phase"):
        NonfiniteError("not_a_phase", "x")
    for p in PHASES:
        e = NonfiniteError(p, "msg")
        assert e.phase == p
        assert "traceback" in e.to_dict()
        assert e.to_dict()["confirm_read"] is False


def _need_ckpt_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if not CK.is_file() or not EVAL_BATCH.is_file():
        pytest.skip("forensic epoch_2 copy or eval batch missing")


def _load_eval_batch_and_model(device):
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    m = build_dkpn_random().to(device)
    m.load_state_dict(ck["model"])
    blob = torch.load(EVAL_BATCH, map_location="cpu", weights_only=False)
    batch = blob["batch"] if isinstance(blob, dict) and "batch" in blob else blob
    return m, batch, ck


def test_epoch2_eval_fp16_reproduces_inf():
    _need_ckpt_cuda()
    device = torch.device("cuda:0")
    model, batch, _ = _load_eval_batch_and_model(device)
    model.eval()
    with pytest.raises(NonfiniteError) as ei:
        eval_forward_amp_fp16_diagnostic(model, batch["x"], device)
    err = ei.value
    assert err.phase == "eval_forward"
    assert err.context["model.training"] is False
    assert err.context.get("traceback")
    first = err.context.get("first_nonfinite_module") or {}
    if isinstance(first, dict) and first.get("module"):
        assert "down_branch" in str(first.get("module"))


def test_epoch2_same_batch_eval_fp32_finite():
    _need_ckpt_cuda()
    device = torch.device("cuda:0")
    model, batch, _ = _load_eval_batch_and_model(device)
    logits = eval_forward_fp32(model, batch["x"], device)
    assert torch.isfinite(logits).all()
    kw = {k: batch[k].to(device) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    loss = eval_loss_fp32(logits, kw, model=model, device=device)
    assert torch.isfinite(loss)


def test_fp32_eval_deterministic():
    _need_ckpt_cuda()
    device = torch.device("cuda:0")
    model, batch, _ = _load_eval_batch_and_model(device)
    a = eval_forward_fp32(model, batch["x"], device)
    b = eval_forward_fp32(model, batch["x"], device)
    assert torch.equal(a, b) or torch.allclose(a, b, rtol=0, atol=0)


def test_fixed_height_0p2_metrics_repeatable():
    rng = np.random.default_rng(0)
    s = rng.random((8, 64)).astype(np.float64)
    s = s / s.sum(axis=-1, keepdims=True)
    true = np.array([10.0] * 8)
    vis = np.ones(8, dtype=bool)
    m1 = pick_metrics_at_thr(s, true, vis, OFFICIAL_HEIGHT)
    m2 = pick_metrics_at_thr(s, true, vis, OFFICIAL_HEIGHT)
    assert m1 == m2
    assert m1["thr"] == 0.2


def test_fp32_eval_hooks_no_inf():
    _need_ckpt_cuda()
    device = torch.device("cuda:0")
    model, batch, _ = _load_eval_batch_and_model(device)
    first, handles = install_finite_hooks(model)
    try:
        logits = eval_forward_fp32(model, batch["x"], device)
        pr = probs_fp32(logits)
        assert torch.isfinite(logits).all() and torch.isfinite(pr).all()
        assert not first.get("module")
    finally:
        remove_hooks(handles)


def test_eval_fp32_source_disables_autocast():
    src = Path("src/earthquake/stage10/eval_fp32.py").read_text()
    assert 'torch.autocast("cuda", enabled=False)' in src
    assert "eval_forward_amp_fp16_diagnostic" in src
    assert "Official eval" in src or "official eval" in src.lower()
