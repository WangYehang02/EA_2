"""Waveform-level top1-vs-top2 pairwise pilot (experimental).

Does not modify Stage-6 method lock / confirm. No multi-station. No abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "true_s_sample",
        "s_arrival_sample",
        "p_arrival_sample",
        "ae",
        "error",
        "positive_index",
        "analysis_classes",
        "is_positive",
        "label_none_of_k",
        "closest_ae_s",
    }
)


@dataclass(frozen=True)
class PairwiseWindowConfig:
    pre_s: float = 1.5
    post_s: float = 2.5
    sampling_rate_hz: float = 100.0

    @property
    def n_samples(self) -> int:
        return int(round((self.pre_s + self.post_s) * self.sampling_rate_hz))

    @property
    def center_offset(self) -> int:
        return int(round(self.pre_s * self.sampling_rate_hz))


def crop_candidate_window(
    wave_zne: np.ndarray,
    candidate_sample: float,
    *,
    cfg: PairwiseWindowConfig,
    full_trace_mean: np.ndarray | None = None,
    full_trace_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop [c-1.5s, c+2.5s] with padding mask. Normalize with full-trace stats.

    Returns (x, mask) with shapes (3, T), (T,) where mask=1 for real samples.
    """
    assert wave_zne.ndim == 2 and wave_zne.shape[0] == 3
    t_len = wave_zne.shape[1]
    T = cfg.n_samples
    center = int(round(float(candidate_sample)))
    start = center - cfg.center_offset
    end = start + T
    out = np.zeros((3, T), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    src_a = max(0, start)
    src_b = min(t_len, end)
    dst_a = src_a - start
    dst_b = dst_a + (src_b - src_a)
    if src_b > src_a:
        out[:, dst_a:dst_b] = wave_zne[:, src_a:src_b]
        mask[dst_a:dst_b] = 1.0
    if full_trace_mean is None:
        full_trace_mean = np.mean(wave_zne, axis=1, keepdims=True)
    if full_trace_std is None:
        full_trace_std = np.std(wave_zne, axis=1, keepdims=True)
        full_trace_std = np.maximum(full_trace_std, 1e-6)
    mean = np.asarray(full_trace_mean, dtype=np.float32).reshape(3, 1)
    std = np.asarray(full_trace_std, dtype=np.float32).reshape(3, 1)
    out = (out - mean) / std
    # zero padded regions after norm so padding is not a learnable amplitude cue beyond mask
    out = out * mask[None, :]
    return out.astype(np.float32), mask.astype(np.float32)


class TinyWaveformEncoder(nn.Module):
    """Shared-weight 1D CNN encoder; target ≤300k params."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 3, T)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class PairwiseScorer(nn.Module):
    """s(c) = MLP([encoder(W(c)), scalar(c)]); P(c2>c1)=sigmoid(s2-s1)."""

    def __init__(
        self,
        n_scalar: int,
        *,
        use_waveform: bool = True,
        use_scalar: bool = True,
        enc_dim: int = 64,
        hidden: int = 64,
    ):
        super().__init__()
        if not use_waveform and not use_scalar:
            raise ValueError("need at least one modality")
        self.use_waveform = bool(use_waveform)
        self.use_scalar = bool(use_scalar)
        self.encoder = TinyWaveformEncoder(out_dim=enc_dim) if use_waveform else None
        in_dim = (enc_dim if use_waveform else 0) + (n_scalar if use_scalar else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.n_scalar = int(n_scalar)

    def score_one(
        self,
        wave: torch.Tensor | None,
        mask: torch.Tensor | None,
        scalar: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = []
        if self.use_waveform:
            assert wave is not None and self.encoder is not None
            parts.append(self.encoder(wave, mask))
        if self.use_scalar:
            assert scalar is not None
            parts.append(scalar)
        h = torch.cat(parts, dim=-1)
        return self.head(h).squeeze(-1)

    def forward(
        self,
        wave1: torch.Tensor | None,
        mask1: torch.Tensor | None,
        scalar1: torch.Tensor | None,
        wave2: torch.Tensor | None,
        mask2: torch.Tensor | None,
        scalar2: torch.Tensor | None,
    ) -> torch.Tensor:
        s1 = self.score_one(wave1, mask1, scalar1)
        s2 = self.score_one(wave2, mask2, scalar2)
        return torch.sigmoid(s2 - s1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def assert_swap_antisymmetry(model: PairwiseScorer, batch: dict[str, torch.Tensor], atol: float = 1e-5) -> float:
    model.eval()
    with torch.no_grad():
        p = model(
            batch.get("wave1"),
            batch.get("mask1"),
            batch.get("scalar1"),
            batch.get("wave2"),
            batch.get("mask2"),
            batch.get("scalar2"),
        )
        p_swap = model(
            batch.get("wave2"),
            batch.get("mask2"),
            batch.get("scalar2"),
            batch.get("wave1"),
            batch.get("mask1"),
            batch.get("scalar1"),
        )
        err = (p_swap - (1.0 - p)).abs().max().item()
    if err >= atol:
        raise AssertionError(f"swap antisymmetry failed: max|p_swap-(1-p)|={err}")
    return float(err)


def guard_no_forbidden_columns(columns: list[str]) -> None:
    bad = [c for c in columns if c in FORBIDDEN_FEATURE_NAMES]
    if bad:
        raise RuntimeError(f"forbidden feature columns present: {bad}")


def guard_no_confirm_or_fulldev_path(path: str | Any) -> None:
    s = str(path).lower()
    needles = [
        "confirm",
        "phaseb_eval_manifest",
        "full-dev",
        "full_dev",
        "dev_union.parquet",  # phaseB/dev candidates — not for pairwise train
        "internal_confirm",
    ]
    # allow mentions in comments/docs; this guards data loaders
    for n in needles:
        if n in s and "pairwise" not in s:
            # ranker_train paths are ok; block confirm explicitly
            if "confirm" in s or "phaseb_eval_manifest" in s or "dev_union" in s:
                raise RuntimeError(f"PAIRWISE forbidden path access: {path}")
