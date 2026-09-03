"""Training losses for learned S-gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_from_errors(errors_s: torch.Tensor, cand_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """errors_s: [B,K] absolute error seconds; return normalized soft targets."""
    temp = max(float(temperature), 1e-4)
    logits = (-errors_s / temp).masked_fill(~cand_mask, torch.finfo(errors_s.dtype).min / 4)
    return F.softmax(logits, dim=-1)


def listwise_ce(final_logprob: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    # final_logprob already log-softmax-like; use NLL with soft targets
    neg = torch.finfo(final_logprob.dtype).min / 4
    lp = final_logprob.masked_fill(~cand_mask, neg)
    # renormalize logprob over valid cands
    lp = lp - torch.logsumexp(lp.masked_fill(~cand_mask, neg), dim=-1, keepdim=True)
    tgt = target * cand_mask.float()
    tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(tgt * lp).sum(dim=-1).mean()


def pairwise_margin_loss(
    scores: torch.Tensor,
    errors_s: torch.Tensor,
    cand_mask: torch.Tensor,
    margin: float = 0.5,
    good_thresh_s: float = 0.5,
) -> torch.Tensor:
    """If best cand error<=good_thresh, push best above others by margin."""
    neg = torch.finfo(scores.dtype).min / 4
    sc = scores.masked_fill(~cand_mask, neg)
    best_idx = torch.argmin(errors_s.masked_fill(~cand_mask, 1e9), dim=-1)
    b = torch.arange(scores.size(0), device=scores.device)
    best_err = errors_s[b, best_idx]
    active = best_err <= good_thresh_s
    if not active.any():
        return scores.new_zeros(())
    best_score = sc[b, best_idx]
    # margin over mean of other valid cands
    others = sc.clone()
    others[b, best_idx] = neg
    # max other
    max_other = others.max(dim=-1).values
    loss = F.relu(margin - best_score + max_other)
    return loss[active].mean()


def fallback_bce(gate: torch.Tensor, bad_history: torch.Tensor) -> torch.Tensor:
    """bad_history: [B] bool/float — push gate->1."""
    if bad_history.dtype != torch.bool:
        bad = bad_history > 0.5
    else:
        bad = bad_history
    if not bad.any():
        return gate.new_zeros(())
    target = torch.ones_like(gate[bad])
    return F.binary_cross_entropy(gate[bad].clamp(1e-6, 1 - 1e-6), target)
