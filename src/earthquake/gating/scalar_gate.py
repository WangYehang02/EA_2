"""Scalar gate network and candidate scoring utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarGate(nn.Module):
    """Maps trace features -> gate_s in (0,1); 1=PhaseNet, 0=history."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.1, init_gate: float = 0.8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),  # features + missing mask
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # bias so sigmoid(bias)≈init_gate
        bias = math.log(init_gate / max(1.0 - init_gate, 1e-6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        if missing_mask is None:
            missing_mask = torch.zeros_like(x)
        h = torch.cat([x, missing_mask], dim=-1)
        return torch.sigmoid(self.net(h))


def wave_scores(prob: torch.Tensor, prominence: torch.Tensor, prominence_weight: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(prob.clamp_min(eps)) + prominence_weight * torch.log(prominence.clamp_min(eps))


def history_scores(candidate_sample: torch.Tensor, expected: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sig = sigma.clamp_min(eps)
    z = (candidate_sample - expected) / sig
    return -0.5 * z * z


def gated_candidate_logprobs(
    gate: torch.Tensor,
    wave_score: torch.Tensor,
    hist_score: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    """gate: [B,1]; scores: [B,K]; cand_mask True=valid.

    Returns final logprob over candidates [B,K] (invalid -> -inf).
    """
    neg = torch.finfo(wave_score.dtype).min / 4
    wave = wave_score.masked_fill(~cand_mask, neg)
    hist = hist_score.masked_fill(~cand_mask, neg)
    wave_lp = F.log_softmax(wave, dim=-1)
    hist_lp = F.log_softmax(hist, dim=-1)
    g = gate
    final = g * wave_lp + (1.0 - g) * hist_lp
    final = final.masked_fill(~cand_mask, neg)
    return final


def pick_from_final(final_logprob: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(final_logprob.dtype).min / 4
    return torch.argmax(final_logprob.masked_fill(~cand_mask, neg), dim=-1)
