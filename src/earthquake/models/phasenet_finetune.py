"""PhaseNet finetuning utilities for INSTANCE chronological train split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def soft_phase_targets(
    n_samples: int,
    p_sample: float | None,
    s_sample: float | None,
    sampling_rate: float,
    sigma_s: float = 0.1,
    label_order: str = "PSN",
) -> np.ndarray:
    """Return (3, T) soft targets in `label_order` channel order.

    SeisBench STEAD PhaseNet uses labels ``PSN`` (not the class default ``NPS``).
    Targets must match ``model.labels`` or annotate()/classify will read the wrong channels.
    """
    t = np.arange(n_samples, dtype=np.float64)
    sigma = max(sigma_s * sampling_rate, 1.0)
    p = np.zeros(n_samples, dtype=np.float64)
    s = np.zeros(n_samples, dtype=np.float64)
    if p_sample is not None and np.isfinite(p_sample):
        p = np.exp(-0.5 * ((t - float(p_sample)) / sigma) ** 2)
    if s_sample is not None and np.isfinite(s_sample):
        s = np.exp(-0.5 * ((t - float(s_sample)) / sigma) ** 2)
    phase_mass = np.clip(p + s, 0, 1)
    noise = np.clip(1.0 - phase_mass, 0.0, 1.0)
    by_name = {"N": noise, "P": p, "S": s}
    order = str(label_order).upper()
    if len(order) != 3 or any(c not in by_name for c in order):
        raise ValueError(f"label_order must be a permutation of NPS, got {label_order!r}")
    stack = np.stack([by_name[c] for c in order], axis=0)
    denom = stack.sum(axis=0, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    stack = stack / denom
    return stack.astype(np.float32)


def random_crop_window(
    n_samples: int,
    win: int,
    p_sample: float | None,
    s_sample: float | None,
    mode: str,
    rng: np.random.Generator,
) -> int:
    """Return start index for a crop of length win according to mode."""
    max_start = max(n_samples - win, 0)
    if max_start == 0:
        return 0

    def _around(center: float, margin: int = 200) -> int:
        c = int(center)
        lo = max(0, c - win + margin)
        hi = min(max_start, max(c - margin, 0))
        if hi < lo:
            return int(rng.integers(0, max_start + 1))
        return int(rng.integers(lo, hi + 1))

    if mode == "both" and p_sample is not None and s_sample is not None and np.isfinite(p_sample) and np.isfinite(s_sample):
        # try to include both
        mid = 0.5 * (float(p_sample) + float(s_sample))
        return _around(mid, margin=win // 4)
    if mode == "p_only" and p_sample is not None and np.isfinite(p_sample):
        return _around(float(p_sample))
    if mode == "s_only" and s_sample is not None and np.isfinite(s_sample):
        return _around(float(s_sample))
    # noise / background: avoid phases if possible
    return int(rng.integers(0, max_start + 1))


@dataclass
class AugmentConfig:
    time_shift: int = 50
    amp_scale: tuple[float, float] = (0.5, 1.5)
    noise_std: float = 0.02
    channel_dropout_prob: float = 0.05
    gap_prob: float = 0.05
    polarity_flip_prob: float = 0.1


def augment_zne(x: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    y = x.astype(np.float32).copy()
    # amplitude
    scale = float(rng.uniform(*cfg.amp_scale))
    y *= scale
    # gaussian noise
    if cfg.noise_std > 0:
        y += rng.normal(0.0, cfg.noise_std * (np.std(y) + 1e-6), size=y.shape).astype(np.float32)
    # channel dropout
    if rng.random() < cfg.channel_dropout_prob:
        ch = int(rng.integers(0, 3))
        y[ch] = 0.0
    # gap
    if rng.random() < cfg.gap_prob:
        g0 = int(rng.integers(0, y.shape[-1]))
        g1 = min(y.shape[-1], g0 + int(rng.integers(10, 200)))
        y[:, g0:g1] = 0.0
    # polarity flip
    if rng.random() < cfg.polarity_flip_prob:
        y *= -1.0
    return y


class PhaseNetFinetuneDataset(Dataset):
    """Crops from INSTANCE waveforms for PhaseNet finetuning.

    Expects waveforms already as ENZ arrays retrieved lazily via a reader callback.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        read_fn,
        in_samples: int = 3001,
        sigma_s: float = 0.1,
        augment: bool = True,
        seed: int = 0,
        modes: tuple[str, ...] = ("both", "p_only", "s_only", "noise", "background"),
        label_order: str = "PSN",
    ):
        self.df = index_df.reset_index(drop=True)
        self.read_fn = read_fn
        self.in_samples = int(in_samples)
        self.sigma_s = float(sigma_s)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.modes = modes
        self.label_order = str(label_order).upper()
        self.aug_cfg = AugmentConfig()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        wave_enz = self.read_fn(str(row["trace_name"]))  # (3,T) ENZ
        # to ZNE
        wave = np.stack([wave_enz[2], wave_enz[1], wave_enz[0]], axis=0)
        n = wave.shape[-1]
        sr = float(row.get("sampling_rate_hz", 100.0))
        p = float(row["p_arrival_sample"]) if "p_arrival_sample" in row and pd.notna(row.get("p_arrival_sample")) else None
        s = float(row["s_arrival_sample"]) if "s_arrival_sample" in row and pd.notna(row.get("s_arrival_sample")) else None
        is_noise = bool(row.get("is_noise", False))
        mode = "noise" if is_noise else self.rng.choice(self.modes)
        start = random_crop_window(n, self.in_samples, p, s, mode, self.rng)
        crop = wave[:, start : start + self.in_samples]
        if crop.shape[-1] < self.in_samples:
            pad = np.zeros((3, self.in_samples), dtype=np.float32)
            pad[:, : crop.shape[-1]] = crop
            crop = pad
        # shift labels
        p_c = None if p is None else p - start
        s_c = None if s is None else s - start
        if is_noise or mode == "noise":
            p_c, s_c = None, None
        # optional small time shift augmentation on labels+data
        if self.augment:
            shift = int(self.rng.integers(-self.aug_cfg.time_shift, self.aug_cfg.time_shift + 1))
            crop = np.roll(crop, shift, axis=-1)
            if p_c is not None:
                p_c = p_c + shift
            if s_c is not None:
                s_c = s_c + shift
            crop = augment_zne(crop, self.rng, self.aug_cfg)
        target = soft_phase_targets(
            self.in_samples, p_c, s_c, sr, sigma_s=self.sigma_s, label_order=self.label_order
        )
        # normalize input
        x = crop.astype(np.float32)
        x = x - x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = x / np.maximum(std, 1e-6)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(target),
            "event_id": str(row.get("event_id", "")),
            "trace_name": str(row["trace_name"]),
            "mode": mode,
        }


def phasenet_soft_ce(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Soft cross-entropy. logits/target: (B, 3, T)."""
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(target * log_prob).sum(dim=1).mean()
    return loss
