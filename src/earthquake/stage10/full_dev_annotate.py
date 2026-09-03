"""Official DKPN full-trace annotate for Stage-6 full-dev stop gate.

Matches cbced5a SeisBench annotate: full-stream CF, in_samples=3001, overlap=1500,
stacking=avg, annotate_window_pre STD, then extract_picks(height=0.2, distance=50).

Does not center windows on human S. FP32 only. Never reads confirm.
"""

from __future__ import annotations

import numpy as np
import torch

from earthquake.stage10.crop_v2 import IN_SAMPLES
from earthquake.stage10.dkpn_clean import compute_cf_5ch, dkpn_logits
from earthquake.stage10.dkpn_picks import extract_picks
from earthquake.stage10.fp32_guard import assert_fp32_tensor, assert_no_autocast, assert_params_fp32
from earthquake.stage10.protocol import enz_to_zne
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT

OVERLAP = 1500
STRIDE = IN_SAMPLES - OVERLAP  # 1501
STACKING = "avg"
BLINDING = (0, 0)
EXTRACT_MIN_DISTANCE = 50
EXTRACT_SMOOTH = True
DKPN_K5 = 5
DKPN_K10 = 10


def window_offsets(n_samples: int, *, in_samples: int = IN_SAMPLES, overlap: int = OVERLAP) -> np.ndarray:
    """SeisBench WaveformModel._cut_fragments_array offsets."""
    stride = int(in_samples - overlap)
    if n_samples < in_samples:
        return np.asarray([0], dtype=np.int64)
    offsets = np.arange(0, n_samples - in_samples + 1, stride, dtype=np.int64)
    if int(offsets[-1]) + in_samples < n_samples:
        offsets = np.concatenate([offsets, np.asarray([n_samples - in_samples], dtype=np.int64)])
    return offsets


def linear_detrend_zne(zne: np.ndarray) -> np.ndarray:
    """ObsPy detrend('linear') equivalent used by DKPN PreProc.work before CF."""
    zne = np.asarray(zne, dtype=np.float64)
    t = np.arange(zne.shape[-1], dtype=np.float64)
    out = np.empty(zne.shape, dtype=np.float32)
    for i in range(zne.shape[0]):
        coef = np.polyfit(t, zne[i], 1)
        out[i] = (zne[i] - (coef[0] * t + coef[1])).astype(np.float32)
    return out


def annotate_window_pre_std(window: np.ndarray) -> np.ndarray:
    """In-place-copy of DKPN.annotate_window_pre STD normalization."""
    w = np.asarray(window, dtype=np.float32).copy()
    w[0:3, :] = w[0:3, :] / (np.std(w[0:3, :], axis=1, keepdims=True) + 1e-10)
    w[4, :] = w[4, :] / (np.std(w[4, :], axis=-1, keepdims=True) + 1e-10)
    return w


def windows_from_cf(cf: np.ndarray, *, in_samples: int = IN_SAMPLES) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (W, 5, 3001) windows and offsets for a (5, T) CF matrix."""
    cf = np.asarray(cf, dtype=np.float32)
    n = int(cf.shape[-1])
    if n < in_samples:
        pad = np.zeros((cf.shape[0], in_samples), dtype=np.float32)
        pad[:, :n] = cf
        cf = pad
        n_out = n
        offsets = np.asarray([0], dtype=np.int64)
    else:
        n_out = n
        offsets = window_offsets(n)
    xs = np.stack([annotate_window_pre_std(cf[:, int(o) : int(o) + in_samples]) for o in offsets], axis=0)
    return xs, offsets, n_out


def stack_avg(window_preds: np.ndarray, offsets: np.ndarray, n_samples: int, *, in_samples: int = IN_SAMPLES) -> np.ndarray:
    """Average overlapping window predictions onto the original timeline (SeisBench stacking=avg)."""
    window_preds = np.asarray(window_preds, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    pred_length = int(max(int(offsets.max()) + in_samples, n_samples))
    acc = np.zeros(pred_length, dtype=np.float64)
    cnt = np.zeros(pred_length, dtype=np.float64)
    for pred, off in zip(window_preds, offsets):
        sl = slice(int(off), int(off) + in_samples)
        acc[sl] += pred
        cnt[sl] += 1.0
    out = np.full(pred_length, np.nan, dtype=np.float64)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out[:n_samples]


def cf_from_enz(enz: np.ndarray) -> np.ndarray:
    zne = enz_to_zne(np.asarray(enz, dtype=np.float32))
    zne = linear_detrend_zne(zne)
    return compute_cf_5ch(zne)


@torch.no_grad()
def s_windows_fp32(model: torch.nn.Module, xs: np.ndarray, device: torch.device) -> np.ndarray:
    """xs: (W, 5, 3001) → S-channel softmax (W, 3001), FP32, no autocast."""
    assert_no_autocast()
    assert_params_fp32(model)
    x = torch.from_numpy(np.asarray(xs, dtype=np.float32)).to(device)
    assert_fp32_tensor(x, "annotate_input")
    logits = dkpn_logits(model, x)
    assert_fp32_tensor(logits, "annotate_logits")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite annotate logits")
    pr = torch.softmax(logits.float(), dim=1)
    return pr[:, 1, :].detach().cpu().numpy().astype(np.float64)


def extract_topk_peaks(
    s_proba: np.ndarray,
    *,
    k: int,
    thr: float = OFFICIAL_HEIGHT,
    min_distance: int = EXTRACT_MIN_DISTANCE,
    smooth: bool = EXTRACT_SMOOTH,
) -> tuple[np.ndarray, np.ndarray]:
    peaks, _, ampl, _ = extract_picks(s_proba, thr=float(thr), min_distance=int(min_distance), smooth=bool(smooth))
    if len(peaks) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(ampl)[::-1][: int(k)]
    return np.asarray(peaks, dtype=np.float64)[order], np.asarray(ampl, dtype=np.float64)[order]


def top1_from_peaks(peaks: np.ndarray, ampl: np.ndarray) -> float:
    if peaks.size == 0:
        return float("nan")
    return float(peaks[int(np.argmax(ampl))])


def picks_from_s_proba(s_proba: np.ndarray, *, k5: int = DKPN_K5, k10: int = DKPN_K10) -> dict:
    p5, a5 = extract_topk_peaks(s_proba, k=k5)
    p10, a10 = extract_topk_peaks(s_proba, k=k10)
    peaks_all, _, ampl_all, _ = extract_picks(s_proba, thr=OFFICIAL_HEIGHT, min_distance=EXTRACT_MIN_DISTANCE, smooth=EXTRACT_SMOOTH)
    return {
        "pred_s_sample": top1_from_peaks(peaks_all, ampl_all) if len(peaks_all) else float("nan"),
        "n_peaks": int(len(peaks_all)),
        "s_max": float(np.nanmax(s_proba)) if s_proba.size else float("nan"),
        "peaks_k5": p5.tolist(),
        "ampl_k5": a5.tolist(),
        "peaks_k10": p10.tolist(),
        "ampl_k10": a10.tolist(),
        "frac_no_s_peak": float(len(peaks_all) == 0),
    }
