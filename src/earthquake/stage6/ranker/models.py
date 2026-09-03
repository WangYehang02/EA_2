"""Lightweight Phase C rankers: R1 / R2 shared MLP, R3 DeepSets (<=250k params)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCandidateScorer(nn.Module):
    """R1/R2: shared MLP over candidates + none logit from pooled context."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score = nn.Linear(hidden, 1)
        self.none_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B,K,F], mask: [B,K] True=valid -> logits [B,K+1] (last=none)."""
        h = self.enc(x)
        scores = self.score(h).squeeze(-1)
        neg = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(~mask, neg)
        # masked mean/max
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        h_min = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        mx = h_min.max(dim=1).values
        none = self.none_head(torch.cat([mean, mx], dim=-1)).squeeze(-1)
        return torch.cat([scores, none.unsqueeze(-1)], dim=-1)


class DeepSetsRanker(nn.Module):
    """R3: DeepSets context ranker with none logit."""

    def __init__(self, n_features: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        h_fill = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        mx = h_fill.max(dim=1).values
        ctx = self.ctx(torch.cat([mean, mx], dim=-1))
        ctx_exp = ctx.unsqueeze(1).expand(-1, h.size(1), -1)
        scores = self.score(torch.cat([h, ctx_exp], dim=-1)).squeeze(-1)
        neg = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(~mask, neg)
        none = self.none_head(ctx).squeeze(-1)
        return torch.cat([scores, none.unsqueeze(-1)], dim=-1)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_ranker(variant: str, n_features: int) -> nn.Module:
    if variant in {"R1", "R2"}:
        hidden = 32 if variant == "R1" else 64
        return SharedCandidateScorer(n_features, hidden=hidden)
    if variant == "R3":
        return DeepSetsRanker(n_features, hidden=96)
    raise ValueError(variant)


def listwise_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    *,
    beta: float = 0.2,
    gamma: float = 1.0,
) -> torch.Tensor:
    """logits [B,K+1]; target_index in 0..K (K=none)."""
    logp = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(logp, target_index)
    # pairwise: correct vs hardest negative among candidates (exclude none slot for hardneg among K)
    b, kp1 = logits.shape
    k = kp1 - 1
    loss_pw = logits.new_zeros(())
    if beta > 0 and k > 0:
        for i in range(b):
            t = int(target_index[i].item())
            if t >= k:
                continue  # none target — skip pairwise
            pos = logits[i, t]
            neg_logits = logits[i, :k].clone()
            neg_logits[t] = torch.finfo(logits.dtype).min / 4
            hard = neg_logits.max()
            loss_pw = loss_pw + F.softplus(-(pos - hard))
        loss_pw = loss_pw / b
    # none emphasis (extra weight on none_of_k examples)
    is_none = (target_index == k).float()
    none_ce = -(is_none * logp[:, k]).sum() / max(b, 1)
    return ce + float(beta) * loss_pw + float(gamma) * none_ce
