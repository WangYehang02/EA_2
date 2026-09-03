"""Phase-balanced partial-label loss (controlled ablation; NOT official token-mean CE).

Unknown S still uses -log[p(S)+p(N)]; unknown P still uses -log[p(P)+p(N)].
Event background stays zero-loss. Only certified noise may be full-window N.
Padding has zero weight and therefore zero gradient.
"""

from __future__ import annotations

import torch

from earthquake.stage10.partial_label import _assert_logits

# Pre-registered mix. Sum = 1. Not numerically equivalent to official per-timestep CE.
# complete_n isolates in-window simplex N on complete P+S event crops so 3001 N tokens
# cannot dominate P/S by count. Certified noise is a separate group.
W_P = 0.28
W_S = 0.28
W_PARTIAL = 0.22
W_CERTIFIED_NOISE = 0.12
W_COMPLETE_N = 0.10
ABLATION_NAME = "phase_balanced_partial_label_v1"
NOT_OFFICIAL_TOKEN_CE = True


def _as_noise_bt(is_noise: torch.Tensor, n_t: int) -> torch.Tensor:
    if is_noise.ndim == 1:
        return is_noise.to(dtype=torch.float32)[:, None].expand(-1, n_t)
    return is_noise.to(dtype=torch.float32)


def _group_mean(nll: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    mass = w.sum().clamp_min(1e-8)
    return (nll * w).sum() / mass


def partial_label_group_terms(
    logits: torch.Tensor,
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
    is_noise: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Per-group NLL tensors and weights. Shared by A diagnostics and B training."""
    _assert_logits(logits)
    logp = torch.nn.functional.log_softmax(logits.float(), dim=1)
    lp, ls, ln = logp[:, 0], logp[:, 1], logp[:, 2]
    log_s_or_n = torch.logsumexp(torch.stack([ls, ln], dim=-1), dim=-1)
    log_p_or_n = torch.logsumexp(torch.stack([lp, ln], dim=-1), dim=-1)
    pad = pad_mask
    noise = _as_noise_bt(is_noise, pad.shape[-1])
    nll_p = (-lp) * pad
    nll_s = (-ls) * pad
    nll_n = (-ln) * pad
    nll_np = (-log_s_or_n) * pad
    nll_ns = (-log_p_or_n) * pad
    wp = p_pos * pad
    ws = s_pos * pad
    wpart = (not_p + not_s) * pad
    nll_part = (nll_np * not_p + nll_ns * not_s) * pad
    wnoise = n_pos * pad * noise
    wcn = n_pos * pad * (1.0 - noise)
    return {
        "p": {"nll": nll_p, "w": wp, "mean": _group_mean(nll_p, wp)},
        "s": {"nll": nll_s, "w": ws, "mean": _group_mean(nll_s, ws)},
        "partial": {"nll": nll_part, "w": wpart, "mean": _group_mean(nll_part, wpart)},
        "certified_noise": {"nll": nll_n, "w": wnoise, "mean": _group_mean(nll_n, wnoise)},
        "complete_n": {"nll": nll_n, "w": wcn, "mean": _group_mean(nll_n, wcn)},
    }


def phase_balanced_partial_nll(
    logits: torch.Tensor,
    *,
    p_pos: torch.Tensor,
    s_pos: torch.Tensor,
    n_pos: torch.Tensor,
    not_p: torch.Tensor,
    not_s: torch.Tensor,
    pad_mask: torch.Tensor,
    is_noise: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    groups = partial_label_group_terms(
        logits,
        p_pos=p_pos,
        s_pos=s_pos,
        n_pos=n_pos,
        not_p=not_p,
        not_s=not_s,
        pad_mask=pad_mask,
        is_noise=is_noise,
    )
    coeff = {
        "p": W_P,
        "s": W_S,
        "partial": W_PARTIAL,
        "certified_noise": W_CERTIFIED_NOISE,
        "complete_n": W_COMPLETE_N,
    }
    used_w: list[float] = []
    used_t: list[torch.Tensor] = []
    for name, w in coeff.items():
        mass = groups[name]["w"].sum()
        if float(mass) > 0 and w > 0:
            used_w.append(w)
            used_t.append(groups[name]["mean"] * w)
    denom = sum(used_w) if used_w else 1.0
    if used_t:
        loss = sum(used_t) / denom
    else:
        loss = logits.sum() * 0.0
    stats = {
        "L_p": float(groups["p"]["mean"].detach()),
        "L_s": float(groups["s"]["mean"].detach()),
        "L_partial": float(groups["partial"]["mean"].detach()),
        "L_certified_noise": float(groups["certified_noise"]["mean"].detach()),
        "L_complete_n": float(groups["complete_n"]["mean"].detach()),
        "mass_p": float(groups["p"]["w"].sum().detach()),
        "mass_s": float(groups["s"]["w"].sum().detach()),
        "mass_partial": float(groups["partial"]["w"].sum().detach()),
        "mass_noise": float(groups["certified_noise"]["w"].sum().detach()),
        "mass_complete_n": float(groups["complete_n"]["w"].sum().detach()),
        "W_P": W_P,
        "W_S": W_S,
        "W_PARTIAL": W_PARTIAL,
        "W_CERTIFIED_NOISE": W_CERTIFIED_NOISE,
        "W_COMPLETE_N": W_COMPLETE_N,
        "not_official_token_ce": NOT_OFFICIAL_TOKEN_CE,
        "ablation": ABLATION_NAME,
    }
    means = {k: v["mean"] for k, v in groups.items()}
    return loss, stats, means
