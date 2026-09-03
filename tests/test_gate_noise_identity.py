"""Noise / no-catalog path must bypass gate and equal PhaseNet picks."""

from __future__ import annotations

import torch

from earthquake.gating.inference import predict_gate_pick, scores_at_gate_limits
from earthquake.gating.scalar_gate import ScalarGate


def test_noise_bypass_equals_phasenet_ranking():
    model = ScalarGate(n_features=3, hidden_dim=8, dropout=0.0, init_gate=0.5)
    x = torch.zeros(1, 3)
    miss = torch.ones(1, 3)  # all missing like noise
    prob = torch.tensor([[0.7, 0.6, 0.1]])
    prom = torch.tensor([[0.4, 0.35, 0.05]])
    sample = torch.tensor([[800.0, 900.0, 2000.0]])
    mask = torch.tensor([[True, True, True]])
    expected = torch.tensor([[float("nan")]])
    sigma = torch.tensor([[20.0]])
    hist = torch.tensor([[False]])
    out = predict_gate_pick(model, x, miss, prob, prom, sample, mask, expected, sigma, hist)
    pn_idx, _ = scores_at_gate_limits(prob, prom, sample, mask, torch.zeros(1, 1), sigma)
    assert float(out["gate"][0]) == 1.0
    assert int(out["pick_index"][0]) == int(pn_idx[0])
