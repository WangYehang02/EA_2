"""gate=1 -> PhaseNet ranking; gate=0 -> history ranking."""

from __future__ import annotations

import torch

from earthquake.gating.inference import scores_at_gate_limits
from earthquake.gating.scalar_gate import gated_candidate_logprobs, history_scores, pick_from_final, wave_scores


def test_gate_score_limits_match_pure_rankings():
    prob = torch.tensor([[0.9, 0.4, 0.2]])
    prom = torch.tensor([[0.5, 0.4, 0.1]])
    sample = torch.tensor([[1000.0, 1300.0, 1800.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = torch.tensor([[1300.0]])
    sigma = torch.tensor([[40.0]])
    pw = 0.1

    pn_idx, hist_idx = scores_at_gate_limits(prob, prom, sample, mask, expected, sigma, prominence_weight=pw)

    w = wave_scores(prob, prom, prominence_weight=pw)
    h = history_scores(sample, expected.expand_as(sample), sigma.expand_as(sample))
    ones = torch.ones(1, 1)
    zeros = torch.zeros(1, 1)
    assert int(pick_from_final(gated_candidate_logprobs(ones, w, h, mask), mask)[0]) == int(pn_idx[0])
    assert int(pick_from_final(gated_candidate_logprobs(zeros, w, h, mask), mask)[0]) == int(hist_idx[0])
    # PhaseNet prefers cand0 (highest wave score); history prefers cand1 (closest to expected)
    assert int(pn_idx[0]) == 0
    assert int(hist_idx[0]) == 1
