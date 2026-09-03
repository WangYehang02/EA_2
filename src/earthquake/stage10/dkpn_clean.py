"""Clean DKPN training helpers (random init; Stage-6 picker_train only).

Official INSTANCE pretrained weights are excluded from the main table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from earthquake.models.phasenet_finetune import AugmentConfig, random_crop_window, soft_phase_targets
from earthquake.stage10.protocol import enz_to_zne

DKPN_REPO = Path("/home/yehang/EARTHQUAKE/baseline/DKPN")


def ensure_dkpn_on_path() -> Path:
    p = DKPN_REPO.resolve()
    if not (p / "dkpn" / "core.py").is_file():
        raise FileNotFoundError(f"DKPN core not found at {p}")
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p


def build_dkpn_random() -> torch.nn.Module:
    ensure_dkpn_on_path()
    from dkpn.core import DKPN

    model = DKPN(in_channels=5, classes=3, phases="PSN", sampling_rate=100)
    # Explicitly random (no from_pretrained)
    return model


def compute_cf_5ch(wave_zne: np.ndarray, default_args: dict | None = None) -> np.ndarray:
    """Compute DKPN 5-channel CF features for a ZNE matrix (3, T)."""
    ensure_dkpn_on_path()
    from dkpn.core import DKPN, PreProc

    args = default_args or DKPN().default_args
    pr = PreProc(**args)
    mm = np.asarray(wave_zne, dtype=np.float32)
    if mm.ndim != 2 or mm.shape[0] != 3:
        raise ValueError(f"expected (3,T) ZNE, got {mm.shape}")
    return pr.__matrix_cfs__(mm)


def normalize_cf_window(cf: np.ndarray, *, in_samples: int = 3001, fp_stabilization_s: float = 4.0, sr: float = 100.0) -> np.ndarray:
    """Crop fp_stabilization and apply STD norm matching annotate_window_pre."""
    fstab = int(sr * fp_stabilization_s)
    need = fstab + in_samples
    if cf.shape[-1] < need:
        pad = np.zeros((cf.shape[0], need), dtype=np.float32)
        pad[:, : cf.shape[-1]] = cf
        cf = pad
    w = cf[:, fstab : fstab + in_samples].astype(np.float32).copy()
    w[0:3, :] = w[0:3, :] / (np.std(w[0:3, :], axis=1, keepdims=True) + 1e-10)
    w[4, :] = w[4, :] / (np.std(w[4, :], axis=-1, keepdims=True) + 1e-10)
    return w


def dkpn_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """DKPN.forward defaults to softmax; training CE must use raw logits."""
    return model(x, logits=True)


def dkpn_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x, logits=False)


def official_soft_ce_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Official DKPN train.py loss: -mean(y * log(p+eps)) on softmax outputs (no extra softmax)."""
    h = target * torch.log(probs.clamp_min(eps))
    h = h.mean(-1).sum(-1)
    return -h.mean()


def masked_soft_ce(logits: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Soft CE with per-sample channel mask (B,3). Masked channels contribute 0.

    logits/target: (B,3,T); channel_mask: (B,3) with 1=supervise.
    Call with raw logits (DKPN.forward(..., logits=True)), never softmax probabilities.
    """
    if bool((logits.min() >= 0) and (logits.max() <= 1) and torch.allclose(logits.sum(dim=1), torch.ones_like(logits[:, 0]), atol=1e-3)):
        raise RuntimeError(
            "masked_soft_ce received simplex probabilities; pass logits=True. "
            "Applying log_softmax to softmax outputs is an implementation bug."
        )
    # Masked channels must not enter the softmax partition (P-only: S logits cannot steal P/N mass).
    m = channel_mask.to(dtype=logits.dtype)
    logits_m = logits.masked_fill(m[:, :, None] == 0, torch.finfo(logits.dtype).min)
    log_prob = F.log_softmax(logits_m, dim=1)
    per = -(target * log_prob)
    per = per.masked_fill(m[:, :, None] == 0, 0.0)
    per_ch = per.mean(dim=-1)  # (B,3)
    denom = m.sum(dim=-1).clamp_min(1.0)
    loss = (per_ch * m).sum(dim=-1) / denom
    return loss.mean()


class CleanDKPNCropDataset(Dataset):
    """INSTANCE ENZ → ZNE → CF(5) → 3001 crop; masked P/S/N soft labels.

    - P+S: supervise P,S,N
    - P-only: supervise P and N (from P); S mask=0 (NOT treated as noise peak)
    - noise: supervise all channels as noise-dominant soft labels
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        read_fn,
        *,
        in_samples: int = 3001,
        fp_stabilization_s: float = 4.0,
        sigma_s: float = 0.1,
        augment: bool = True,
        seed: int = 0,
        allow_p_only: bool = True,
        confirm_event_ids: set[str] | None = None,
        cf_cache: dict[str, np.ndarray] | None = None,
        precompute_cf: bool = False,
    ):
        self.meta = meta.reset_index(drop=True)
        self.read_fn = read_fn
        self.in_samples = int(in_samples)
        self.fp_s = float(fp_stabilization_s)
        self.sigma_s = float(sigma_s)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.allow_p_only = bool(allow_p_only)
        self.confirm_event_ids = set(confirm_event_ids or [])
        self.cf_cache = cf_cache if cf_cache is not None else {}
        self.aug_cfg = AugmentConfig()
        ensure_dkpn_on_path()
        from dkpn.core import DKPN

        self._default_args = DKPN().default_args

        # filter
        keep = []
        for i, row in self.meta.iterrows():
            eid = str(row.get("event_id", ""))
            if eid and eid in self.confirm_event_ids:
                raise RuntimeError(f"confirm event leaked into DKPN train: {eid}")
            is_noise = bool(row.get("is_noise", False))
            p = row.get("p_arrival_sample", np.nan)
            s = row.get("s_arrival_sample", np.nan)
            has_p = (not is_noise) and pd.notna(p) and np.isfinite(float(p))
            has_s = (not is_noise) and pd.notna(s) and np.isfinite(float(s))
            if is_noise or (has_p and has_s) or (self.allow_p_only and has_p):
                keep.append(i)
            # else drop unlabeled / S-only
        self.meta = self.meta.loc[keep].reset_index(drop=True)
        if precompute_cf:
            for i in range(len(self.meta)):
                self._ensure_cf(i)

    def __len__(self) -> int:
        return len(self.meta)

    def _read_enz(self, trace_name: str, *, is_noise: bool) -> np.ndarray:
        try:
            return self.read_fn(trace_name, is_noise=is_noise)
        except TypeError:
            return self.read_fn(trace_name)

    def _ensure_cf(self, idx: int) -> np.ndarray:
        row = self.meta.iloc[idx]
        key = str(row["trace_name"])
        if key in self.cf_cache:
            return self.cf_cache[key]
        is_noise = bool(row.get("is_noise", False))
        enz = self._read_enz(key, is_noise=is_noise)
        zne = enz_to_zne(np.asarray(enz, dtype=np.float32))
        cf = compute_cf_5ch(zne, self._default_args)
        self.cf_cache[key] = cf
        return cf

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        is_noise = bool(row.get("is_noise", False))
        sr = float(row.get("sampling_rate_hz", 100.0) or 100.0)
        p = float(row["p_arrival_sample"]) if (not is_noise and pd.notna(row.get("p_arrival_sample"))) else None
        s = float(row["s_arrival_sample"]) if (not is_noise and pd.notna(row.get("s_arrival_sample"))) else None
        has_p = p is not None and np.isfinite(p)
        has_s = s is not None and np.isfinite(s)

        enz = self._read_enz(str(row["trace_name"]), is_noise=is_noise)
        n = int(np.asarray(enz).shape[-1])
        fstab = int(sr * self.fp_s)
        # crop of length in_samples on the *label timeline* after dropping fstab from CF
        # Equivalent: crop raw of length in_samples+fstab starting at start, then CF+norm
        win_raw = self.in_samples + fstab
        mode = "noise" if is_noise else ("both" if has_s else "p_only")
        start = random_crop_window(n, win_raw, p, s, mode if not is_noise else "noise", self.rng)
        crop_enz = np.asarray(enz, dtype=np.float32)[:, start : start + win_raw]
        if crop_enz.shape[-1] < win_raw:
            pad = np.zeros((3, win_raw), dtype=np.float32)
            pad[:, : crop_enz.shape[-1]] = crop_enz
            crop_enz = pad
        crop_zne = enz_to_zne(crop_enz)

        # label coords relative to post-fstab window
        p_c = None if not has_p else (p - start - fstab)
        s_c = None if not has_s else (s - start - fstab)
        if is_noise:
            p_c, s_c = None, None
            has_p, has_s = False, False

        if self.augment:
            shift = int(self.rng.integers(-self.aug_cfg.time_shift, self.aug_cfg.time_shift + 1))
            crop_zne = np.roll(crop_zne, shift, axis=-1)
            if p_c is not None:
                p_c = p_c + shift
            if s_c is not None:
                s_c = s_c + shift
            # amplitude / polarity on waveform before CF
            scale = float(self.rng.uniform(*self.aug_cfg.amp_scale))
            crop_zne = crop_zne * scale
            if self.rng.random() < self.aug_cfg.polarity_flip_prob:
                crop_zne = -crop_zne
            if self.rng.random() < self.aug_cfg.gap_prob:
                g0 = int(self.rng.integers(0, crop_zne.shape[-1]))
                g1 = min(crop_zne.shape[-1], g0 + int(self.rng.integers(10, 200)))
                crop_zne[:, g0:g1] = 0.0

        cf = compute_cf_5ch(crop_zne, self._default_args)
        x = normalize_cf_window(cf, in_samples=self.in_samples, fp_stabilization_s=self.fp_s, sr=sr)

        target = soft_phase_targets(self.in_samples, p_c, s_c, sr, sigma_s=self.sigma_s, label_order="PSN")
        # channel mask: P, S, N
        if is_noise:
            mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif has_p and has_s:
            mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif has_p and not has_s:
            # P-only: do not supervise S; keep N from soft_phase_targets (uses only P)
            mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            # zero S target so any accidental leakage is harmless
            target[1] = 0.0
            # renormalize P/N only
            pn = target[0] + target[2]
            pn = np.maximum(pn, 1e-8)
            target[0] = target[0] / pn
            target[2] = target[2] / pn
        else:
            mask = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if not np.isfinite(x).all():
            raise RuntimeError(f"non-finite CF for {row['trace_name']}")

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(target.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "trace_name": str(row["trace_name"]),
            "event_id": str(row.get("event_id", "")),
            "has_s": bool(has_s),
            "is_noise": bool(is_noise),
        }
