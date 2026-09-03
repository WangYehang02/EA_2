"""Scalar gate forward is deterministic in eval mode."""

from __future__ import annotations

import torch

from earthquake.gating.scalar_gate import ScalarGate


def test_scalar_gate_eval_deterministic():
    torch.manual_seed(123)
    m = ScalarGate(n_features=5, hidden_dim=16, dropout=0.5, init_gate=0.8)
    m.eval()
    x = torch.randn(4, 5)
    miss = torch.zeros_like(x)
    with torch.no_grad():
        a = m(x, miss)
        b = m(x, miss)
    assert torch.allclose(a, b)
