"""Partial-label NLL for P/S/N (logits in, no double softmax)."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _assert_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 3 or logits.shape[1] != 3:
        raise ValueError(f"expected (B,3,T) logits, got {tuple(logits.shape)}")
    s = logits.sum(dim=1)
    if bool((logits.min() >= 0) and (logits.max() <= 1 + 1e-5) and torch.allclose(s, torch.ones_like(s), atol=2e-3)):
        raise RuntimeError("partial_label_nll received softmax probabilities; pass raw logits")


def complete_ce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scheme A: log_softmax CE. target (B,3,T) simplex. Reduction: mean_T, sum_C, mean_B."""
    _assert_logits(logits)
    logp = F.log_softmax(logits.float(), dim=1)
    return -(target * logp).mean(dim=-1).sum(dim=-1).mean()


def complete_ce_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scheme B: -y log(p+eps) on softmax probabilities (official DKPN shape)."""
    h = target * torch.log(probs.clamp_min(eps))
    return -h.mean(dim=-1).sum(dim=-1).mean()


def token_weights(
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    return (p_pos + s_pos + n_pos + not_p + not_s) * pad_mask


def partial_nll_numerator(
    logits: torch.Tensor,
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """FP32 token NLL (no reduction). Returns (nll*pad, weights, per-term sums).

    Never runs under autocast: log_softmax / logsumexp stay FP32.
    Does not softmax then log_softmax.
    """
    _assert_logits(logits)
    logits_fp32 = logits.float()
    logp = F.log_softmax(logits_fp32, dim=1)
    lp, ls, ln = logp[:, 0], logp[:, 1], logp[:, 2]
    log_s_or_n = torch.logsumexp(torch.stack([ls, ln], dim=-1), dim=-1)
    log_p_or_n = torch.logsumexp(torch.stack([lp, ln], dim=-1), dim=-1)
    terms = {
        "L_p": (p_pos * (-lp) * pad_mask).sum(),
        "L_s": (s_pos * (-ls) * pad_mask).sum(),
        "L_n": (n_pos * (-ln) * pad_mask).sum(),
        "L_not_p": (not_p * (-log_s_or_n) * pad_mask).sum(),
        "L_not_s": (not_s * (-log_p_or_n) * pad_mask).sum(),
    }
    nll = (
        p_pos * (-lp)
        + s_pos * (-ls)
        + n_pos * (-ln)
        + not_p * (-log_s_or_n)
        + not_s * (-log_p_or_n)
    ) * pad_mask
    w = token_weights(p_pos=p_pos, s_pos=s_pos, n_pos=n_pos, not_p=not_p, not_s=not_s, pad_mask=pad_mask)
    return nll, w, terms


def reduce_global_token_mean(
    numerator: torch.Tensor,
    valid_count: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Global token mean. valid_count==0 → differentiable zero, not divide-by-clamped-1.

    Always adds logits.sum()*0 so every rank's parameters stay in the graph (DDP).
    """
    anchor = logits.float().sum() * 0.0
    den = valid_count.float()
    num = numerator.float()
    if float(den.detach()) > 0.0:
        return num / den + anchor
    return anchor


def allreduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum across DDP ranks. No-op when not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        y = x.clone()
        torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        return y
    return x


def partial_label_nll(
    logits: torch.Tensor,
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    """Time-weighted partial-label NLL (global token mean).

    Call with autocast disabled. Weights (B,T) ≥ 0. All-unknown windows contribute
    zero to numerator and denominator (not N). If global valid count is 0, return
    a differentiable zero connected to logits.
    """
    from contextlib import nullcontext

    ctx = torch.autocast("cuda", enabled=False) if logits.is_cuda else nullcontext()
    with ctx:
        nll, w, _ = partial_nll_numerator(
            logits, p_pos=p_pos, s_pos=s_pos, n_pos=n_pos, not_p=not_p, not_s=not_s, pad_mask=pad_mask
        )
        local_num = nll.sum()
        local_den = w.sum()
        if distributed:
            local_num = allreduce_sum(local_num)
            local_den = allreduce_sum(local_den)
        return reduce_global_token_mean(local_num, local_den, logits)


def partial_label_nll_legacy_clamped(
    logits: torch.Tensor,
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Pre-forensic reduction: denom.clamp_min(1.0). Kept only to reproduce path A."""
    _assert_logits(logits)
    logp = F.log_softmax(logits.float(), dim=1)
    lp, ls, ln = logp[:, 0], logp[:, 1], logp[:, 2]
    log_s_or_n = torch.logsumexp(torch.stack([ls, ln], dim=-1), dim=-1)
    log_p_or_n = torch.logsumexp(torch.stack([lp, ln], dim=-1), dim=-1)
    nll = (
        p_pos * (-lp)
        + s_pos * (-ls)
        + n_pos * (-ln)
        + not_p * (-log_s_or_n)
        + not_s * (-log_p_or_n)
    )
    wsum = (p_pos + s_pos + n_pos + not_p + not_s) * pad_mask
    nll = nll * pad_mask
    denom = wsum.sum().clamp_min(1.0)
    return nll.sum() / denom


def simulate_ddp_token_mean(
    logits: torch.Tensor,
    weights: dict[str, torch.Tensor],
    n_ranks: int,
) -> torch.Tensor:
    """Math DDP: split batch across ranks, sum num/den, same as single-GPU global mean."""
    b = logits.shape[0]
    if b % n_ranks != 0:
        raise ValueError(f"batch {b} not divisible by n_ranks {n_ranks}")
    bs = b // n_ranks
    nums = []
    dens = []
    for r in range(n_ranks):
        sl = slice(r * bs, (r + 1) * bs)
        kw = {k: weights[k][sl] for k in ("p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask")}
        nll, w, _ = partial_nll_numerator(logits[sl], **kw)
        nums.append(nll.sum())
        dens.append(w.sum())
    return reduce_global_token_mean(sum(nums), sum(dens), logits)
