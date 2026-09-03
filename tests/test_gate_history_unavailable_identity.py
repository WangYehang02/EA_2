"""When history unavailable, inference must force gate=1 (== PhaseNet ranking)."""

from __future__ import annotations

import torch

from earthquake.gating.inference import predict_gate_pick, scores_at_gate_limits
from earthquake.gating.scalar_gate import ScalarGate


def test_history_unavailable_forces_gate_one():
    torch.manual_seed(0)
    model = ScalarGate(n_features=4, hidden_dim=8, dropout=0.0, init_gate=0.2)
    # train briefly toward gate=0 to ensure forcing matters
    x = torch.randn(2, 4)
    miss = torch.zeros_like(x)
    prob = torch.tensor([[0.9, 0.2, 0.1], [0.4, 0.35, 0.3]])
    prom = torch.tensor([[0.5, 0.2, 0.1], [0.3, 0.25, 0.2]])
    sample = torch.tensor([[1000.0, 1500.0, 2000.0], [1100.0, 1600.0, 2100.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    expected = torch.tensor([[1500.0], [1600.0]])
    sigma = torch.tensor([[50.0], [50.0]])
    hist = torch.tensor([[False], [True]])

    out = predict_gate_pick(model, x, miss, prob, prom, sample, mask, expected, sigma, hist)
    assert float(out["gate"][0]) == 1.0
    pn_idx, _ = scores_at_gate_limits(prob, prom, sample, mask, expected, sigma)
    assert int(out["pick_index"][0]) == int(pn_idx[0])
