"""Direct per-candidate ranker (auxiliary upper-bound model)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateRanker(nn.Module):
    def __init__(self, n_trace_features: int, n_cand_features: int = 6, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        in_dim = n_trace_features * 2 + n_cand_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        trace_x: torch.Tensor,
        trace_mask: torch.Tensor,
        cand_feat: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """cand_feat: [B,K,C] -> scores [B,K]."""
        b, k, _ = cand_feat.shape
        tx = torch.cat([trace_x, trace_mask], dim=-1).unsqueeze(1).expand(-1, k, -1)
        h = torch.cat([tx, cand_feat], dim=-1)
        scores = self.mlp(h).squeeze(-1)
        neg = torch.finfo(scores.dtype).min / 4
        return scores.masked_fill(~cand_mask, neg)
