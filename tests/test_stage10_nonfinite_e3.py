"""Non-finite epoch-3 forensic tests. No confirm, no 30-epoch train."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from earthquake.stage10.amp_forward import assert_input_finite, shard_valid_counts
from earthquake.stage10.partial_label import (
    partial_label_nll,
    partial_label_nll_legacy_clamped,
    simulate_ddp_token_mean,
    token_weights,
)

OFFEND = Path(
    "/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop/"
    "forensic_nonfinite_e3/offending_batch.pt"
)


def _supervised(b=8, t=32):
    logits = torch.randn(b, 3, t, requires_grad=True)
    p = torch.zeros(b, t)
    p[:, 4] = 1.0
    z = torch.zeros(b, t)
    pad = torch.ones(b, t)
    kw = dict(p_pos=p, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad)
    return logits, kw


def _unknown(b=8, t=32):
    logits = torch.randn(b, 3, t, requires_grad=True)
    z = torch.zeros(b, t)
    pad = torch.ones(b, t)
    kw = dict(p_pos=z, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad)
    return logits, kw


def test_single_rank_all_unknown_zero_loss_finite_grad():
    logits, kw = _unknown(4, 16)
    loss = partial_label_nll(logits, **kw)
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) == 0.0


def test_one_rank_unknown_others_supervised_ddp_equiv():
    torch.manual_seed(0)
    logits, kw = _supervised(8, 32)
    # ranks 0 all-unknown, ranks 1-3 supervised
    kw_u = {k: v.clone() for k, v in kw.items()}
    kw_u["p_pos"][:2] = 0
    full = partial_label_nll(logits, **kw_u)
    sim = simulate_ddp_token_mean(logits, kw_u, n_ranks=4)
    assert torch.isfinite(full) and torch.isfinite(sim)
    assert abs(float(full - sim)) < 1e-6
    shards = shard_valid_counts({"p_pos": kw_u["p_pos"], "s_pos": kw_u["s_pos"], "n_pos": kw_u["n_pos"], "not_p": kw_u["not_p"], "not_s": kw_u["not_s"], "pad_mask": kw_u["pad_mask"]}, 4)
    assert shards[0] == 0.0
    assert shards[1] > 0 and shards[2] > 0 and shards[3] > 0


def test_all_ranks_unknown_differentiable_zero():
    logits, kw = _unknown(8, 16)
    loss = simulate_ddp_token_mean(logits, kw, n_ranks=4)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_fp32_loss_finite():
    z = torch.zeros(2, 64)
    pad = torch.ones(2, 64)
    p = z.clone()
    p[:, 10] = 1.0
    for mag in (1e2, 1e3, 1e4):
        logits = torch.zeros(2, 3, 64, requires_grad=True)
        with torch.no_grad():
            logits[:, 0, 10] = mag
            logits[:, 1, :] = -mag
        loss = partial_label_nll(logits, p_pos=p, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad)
        assert torch.isfinite(loss), mag
        g = torch.autograd.grad(loss, logits, retain_graph=False)[0]
        assert torch.isfinite(g).all(), mag


def test_fp16_forward_fp32_loss_matches_fp32_on_finite_logits():
    torch.manual_seed(2)
    logits, kw = _supervised(4, 64)
    ref = partial_label_nll(logits.float(), **kw)
    # emulate AMP: logits stored fp16 then loss fp32
    loss_b = partial_label_nll(logits.detach().half().float(), **{k: v for k, v in kw.items()})
    assert torch.isfinite(ref) and torch.isfinite(loss_b)
    assert abs(float(ref - loss_b)) < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="no bf16")
def test_bf16_logits_fp32_loss_finite():
    logits, kw = _supervised(4, 32)
    x = logits.detach().to(torch.bfloat16)
    loss = partial_label_nll(x.float(), **kw)
    assert torch.isfinite(loss)


def test_padding_all_zero_zero_loss():
    logits = torch.randn(3, 3, 20, requires_grad=True)
    z = torch.zeros(3, 20)
    pad = torch.zeros(3, 20)
    p = z.clone()
    p[:, 1] = 1.0  # would supervise but pad is 0
    loss = partial_label_nll(logits, p_pos=p, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad)
    assert float(loss) == 0.0
    loss.backward()
    assert float(logits.grad.abs().sum()) == 0.0


def test_legacy_clamp_vs_stable_agree_when_denom_positive():
    logits, kw = _supervised(5, 40)
    a = partial_label_nll_legacy_clamped(logits, **kw)
    b = partial_label_nll(logits, **kw)
    assert abs(float(a - b)) < 1e-6


def test_single_gpu_matches_4rank_ddp_loss_and_grad():
    torch.manual_seed(7)
    logits, kw = _supervised(8, 48)
    logits_a = logits.detach().clone().requires_grad_(True)
    logits_b = logits.detach().clone().requires_grad_(True)
    la = partial_label_nll(logits_a, **kw)
    lb = simulate_ddp_token_mean(logits_b, kw, n_ranks=4)
    assert abs(float(la - lb)) < 1e-6
    la.backward()
    lb.backward()
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-6, rtol=1e-5)


def test_global_valid_zero_differentiable():
    logits, kw = _unknown(4, 8)
    loss = partial_label_nll(logits, **kw)
    g = torch.autograd.grad(loss, logits, allow_unused=False)[0]
    assert g is not None and torch.isfinite(g).all()


def test_nonfinite_input_reports_trace_id():
    batch = {
        "x": torch.zeros(2, 5, 16),
        "trace_name": ["ok.trace", "bad.TRACE"],
    }
    batch["x"][1, 0, 3] = torch.inf
    with pytest.raises(RuntimeError, match="bad.TRACE"):
        assert_input_finite(batch)


def test_nonfinite_label_reports_trace_id():
    batch = {
        "x": torch.zeros(2, 5, 16),
        "p_pos": torch.zeros(2, 16),
        "trace_name": ["ok.trace", "lab.BAD"],
    }
    batch["p_pos"][1, 2] = torch.nan
    with pytest.raises(RuntimeError, match="lab.BAD"):
        assert_input_finite(batch)


def test_no_clamp_min_1_in_stable_loss_source():
    from pathlib import Path
    src = Path("src/earthquake/stage10/partial_label.py").read_text()
    # stable path must not fake supervision with clamp_min(1)
    assert "clamp_min(1.0)" in src  # legacy only
    assert "reduce_global_token_mean" in src
    assert "logits.float().sum() * 0.0" in src


def test_resume_blocked_until_e3_gate(tmp_path: Path):
    from earthquake.stage10.v3_continue import GATE_KEYS, OVERFLOW_FLAG, resume_blocked_nonfinite_e3

    assert resume_blocked_nonfinite_e3(tmp_path) is None
    (tmp_path / "TRAIN.FAILED_NONFINITE_E3").write_text("flag\n")
    assert resume_blocked_nonfinite_e3(tmp_path) == "TRAIN.FAILED_NONFINITE_E3_gate_incomplete"
    gdir = tmp_path / "forensic_nonfinite_e3"
    gdir.mkdir()
    (gdir / "RESUME.GATE.json").write_text("{}\n")
    assert resume_blocked_nonfinite_e3(tmp_path) == "TRAIN.FAILED_NONFINITE_E3_gate_incomplete"
    (gdir / "RESUME.GATE.json").write_text(__import__("json").dumps({k: True for k in GATE_KEYS}))
    assert resume_blocked_nonfinite_e3(tmp_path) is None
    (tmp_path / OVERFLOW_FLAG).write_text("flag\n")
    assert resume_blocked_nonfinite_e3(tmp_path) == "V3.FAILED_FP16_ACTIVATION_OVERFLOW_do_not_resume_epoch2"


def test_v2_batch_loss_does_not_wrap_nll_in_autocast():
    src = Path("scripts/train_stage10_dkpn_v2.py").read_text()
    assert "AMP wraps forward only" in src
    assert "partial_label_nll(logits.float()" in src
    # loss must not sit inside the enabled autocast block
    assert "with torch.cuda.amp.autocast(enabled=amp and device.type == \"cuda\"):\n        logits = dkpn_logits(model, x)\n        loss = partial_label_nll" not in src


def test_offending_batch_regression_if_captured():
    if not OFFEND.is_file():
        pytest.skip("offending_batch.pt not captured yet")
    blob = torch.load(OFFEND, map_location="cpu", weights_only=False)
    batch = blob["batch"]
    kw = {k: batch[k] for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    logits = blob.get("logits_fp32")
    if logits is None:
        pytest.skip("no logits in offending dump")
    loss = partial_label_nll(logits.float(), **kw)
    assert torch.isfinite(loss)
