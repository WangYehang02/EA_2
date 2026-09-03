"""Inference helpers for scalar gate S re-ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from earthquake.gating.scalar_gate import (
    ScalarGate,
    gated_candidate_logprobs,
    history_scores,
    pick_from_final,
    wave_scores,
)


@torch.no_grad()
def predict_gate_pick(
    model: ScalarGate,
    x: torch.Tensor,
    missing: torch.Tensor,
    prob: torch.Tensor,
    prominence: torch.Tensor,
    sample: torch.Tensor,
    cand_mask: torch.Tensor,
    expected: torch.Tensor,
    sigma: torch.Tensor,
    history_available: torch.Tensor,
    prominence_weight: float = 0.1,
) -> dict[str, Any]:
    """Batch inference. Forces gate=1 when history unavailable."""
    model.eval()
    gate = model(x, missing)
    # hard force
    ha = history_available.view(-1, 1).float()
    gate = gate * ha + (1.0 - ha) * 1.0
    w = wave_scores(prob, prominence, prominence_weight=prominence_weight)
    # broadcast expected/sigma
    exp = expected.view(-1, 1).expand_as(sample)
    sig = sigma.view(-1, 1).expand_as(sample)
    h = history_scores(sample, exp, sig)
    final = gated_candidate_logprobs(gate, w, h, cand_mask)
    idx = pick_from_final(final, cand_mask)
    b = torch.arange(sample.size(0), device=sample.device)
    picked = sample[b, idx]
    # when no valid cand, nan
    valid = cand_mask.any(dim=-1)
    picked = torch.where(valid, picked, torch.full_like(picked, float("nan")))
    return {
        "gate": gate.view(-1),
        "pick_index": idx,
        "pick_sample": picked,
        "final_logprob": final,
    }


def scores_at_gate_limits(prob, prominence, sample, cand_mask, expected, sigma, prominence_weight=0.1):
    """Return PhaseNet-ranking pick (gate=1) and history-ranking pick (gate=0)."""
    w = wave_scores(prob, prominence, prominence_weight=prominence_weight)
    exp = expected.view(-1, 1).expand_as(sample)
    sig = sigma.view(-1, 1).expand_as(sample)
    h = history_scores(sample, exp, sig)
    ones = torch.ones(prob.size(0), 1, device=prob.device, dtype=prob.dtype)
    zeros = torch.zeros_like(ones)
    final1 = gated_candidate_logprobs(ones, w, h, cand_mask)
    final0 = gated_candidate_logprobs(zeros, w, h, cand_mask)
    return pick_from_final(final1, cand_mask), pick_from_final(final0, cand_mask)
