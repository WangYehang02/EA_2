from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedGate(nn.Module):
    """MLP gate: 1=trust PhaseNet, 0=trust history."""

    def __init__(self, in_dim: int, hidden: int = 64, init_gate: float = 0.9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # gate_p, gate_s
        )
        # Initialize last bias so sigmoid(bias) ≈ init_gate
        bias = math.log(init_gate / (1.0 - init_gate))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, features: torch.Tensor, history_available: torch.Tensor | None = None) -> torch.Tensor:
        gates = torch.sigmoid(self.net(features))
        if history_available is not None:
            # force gate=1 when history unavailable
            mask = history_available.float().view(-1, 1)
            gates = mask * gates + (1.0 - mask) * torch.ones_like(gates)
        return gates
