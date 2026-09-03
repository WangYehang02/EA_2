"""SegPhase adapter for Stage-9 same-protocol eval (read-only third-party weights)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks

SEGPHASE_ROOT = Path("/home/yehang/EARTHQUAKE/baseline/SegPhase")
DEFAULT_CKPT = SEGPHASE_ROOT / "model" / "model_100Hz.pth"


@dataclass
class SegPhaseConfig:
    checkpoint: Path = DEFAULT_CKPT
    device: str = "cuda:0"
    sample_rate_hz: float = 100.0
    window_samples: int = 3000  # 30 s @ 100 Hz
    peak_height: float = 0.1  # official find_peaks height
    peak_distance_s: float = 1.0  # official distance=int(1/sf) with sf=0.01 → 100 samples? 
    # pred.py: distance=int(1/sf) where sf is sample interval in seconds (0.01) → distance=100
    channel_order: str = "UD_NS_EW"  # == Z,N,E from INSTANCE ENZ
    window_scheme: str = "A"  # A non-overlap 4x30s; B 50% overlap


def enz_to_ud_ns_ew(wave_enz: np.ndarray) -> np.ndarray:
    """INSTANCE (3, T) ENZ → SegPhase (3, T) UD,NS,EW = Z,N,E."""
    wave_enz = np.asarray(wave_enz, dtype=np.float32)
    assert wave_enz.ndim == 2 and wave_enz.shape[0] == 3
    e, n, z = wave_enz[0], wave_enz[1], wave_enz[2]
    return np.stack([z, n, e], axis=0)


def zscore_channels(wave: np.ndarray) -> np.ndarray:
    out = np.empty_like(wave, dtype=np.float32)
    for i in range(wave.shape[0]):
        x = wave[i]
        std = float(np.std(x))
        if std < 1e-12:
            out[i] = 0.0
        else:
            out[i] = (x - float(np.mean(x))) / std
    return out


def window_starts(scheme: str, n_samples: int = 12000, win: int = 3000) -> list[int]:
    if scheme == "A":
        starts = [0, 3000, 6000, 9000]
    elif scheme == "B":
        # 50% overlap: step=1500
        starts = list(range(0, n_samples - win + 1, 1500))
    else:
        raise ValueError(scheme)
    return [s for s in starts if s + win <= n_samples]


class SegPhasePicker:
    """Thin wrapper around official Model + find_peaks; does not modify repo files."""

    def __init__(self, cfg: SegPhaseConfig | None = None):
        self.cfg = cfg or SegPhaseConfig()
        if not self.cfg.checkpoint.exists():
            raise FileNotFoundError(self.cfg.checkpoint)
        # import model class from third-party tree without installing package
        model_dir = str(SEGPHASE_ROOT / "model")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from model_str import Model  # type: ignore

        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or "cpu" in self.cfg.device else "cpu")
        if "cuda" in str(self.cfg.device) and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model = Model(in_length=100 * 30, in_channels=3, class_num=3, strides=[3, 2, 2], kernel_size=3)
        state = torch.load(self.cfg.checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._ckpt_sha = None

    @property
    def checkpoint_sha256(self) -> str:
        if self._ckpt_sha is None:
            import hashlib

            h = hashlib.sha256()
            with open(self.cfg.checkpoint, "rb") as f:
                for c in iter(lambda: f.read(1 << 20), b""):
                    h.update(c)
            self._ckpt_sha = h.hexdigest()
        return self._ckpt_sha

    def _forward_windows(self, wave_zne: np.ndarray) -> np.ndarray:
        """Return S-proba of length n_samples by max-pooling overlapping window preds."""
        n = wave_zne.shape[-1]
        win = self.cfg.window_samples
        starts = window_starts(self.cfg.window_scheme, n, win)
        s_acc = np.full(n, np.nan, dtype=np.float32)
        p_acc = np.full(n, np.nan, dtype=np.float32)
        with torch.inference_mode():
            for s0 in starts:
                chunk = zscore_channels(wave_zne[:, s0 : s0 + win])
                x = torch.from_numpy(chunk[None, ...]).float().to(self.device)
                pred = self.model(x).detach().cpu().numpy()[0]  # (C, L) C: P,S,?
                # official: pred[0]=P, pred[1]=S
                p_seg = pred[0]
                s_seg = pred[1]
                # map 1:1 — model output length should equal 3000
                L = min(win, p_seg.shape[-1])
                for i in range(L):
                    gi = s0 + i
                    # take max across overlapping windows
                    pv, sv = float(p_seg[i]), float(s_seg[i])
                    if not np.isfinite(p_acc[gi]) or pv > p_acc[gi]:
                        p_acc[gi] = pv
                    if not np.isfinite(s_acc[gi]) or sv > s_acc[gi]:
                        s_acc[gi] = sv
        s_acc = np.nan_to_num(s_acc, nan=0.0)
        return s_acc

    def predict_s(
        self,
        wave_enz: np.ndarray,
        *,
        threshold: float | None = None,
        top_k: int = 1,
    ) -> dict[str, Any]:
        thr = self.cfg.peak_height if threshold is None else float(threshold)
        wave_zne = enz_to_ud_ns_ew(wave_enz)
        s_proba = self._forward_windows(wave_zne)
        sf = 1.0 / self.cfg.sample_rate_hz
        distance = int(1.0 / sf)  # official
        idxs, props = find_peaks(s_proba, distance=distance, height=thr)
        if len(idxs) == 0:
            return {
                "pred_s_sample": float("nan"),
                "s_peak_probability": float("nan"),
                "candidates": [],
                "s_proba": s_proba,
                "none_of_k": True,
            }
        heights = props["peak_heights"]
        order = np.argsort(-heights)
        cands = []
        for rank, j in enumerate(order[: max(top_k, 1)]):
            cands.append(
                {
                    "rank": rank,
                    "sample": int(idxs[j]),
                    "prob": float(heights[j]),
                }
            )
        top = cands[0]
        return {
            "pred_s_sample": float(top["sample"]),
            "s_peak_probability": float(top["prob"]),
            "candidates": cands,
            "s_proba": s_proba,
            "none_of_k": False,
            "n_peaks": int(len(idxs)),
        }
