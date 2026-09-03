"""Official DKPN peak extraction (commit cbced5a eval_utils.extract_picks)."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


def extract_picks(ts, thr: float = 0.2, min_distance: int = 50, smooth: bool = True):
    """Match DKPN/dkpn/eval_utils.py::extract_picks (cbced5a)."""
    ts = np.asarray(ts, dtype=np.float64)
    if smooth:
        smoothing_filter = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ts = np.convolve(ts, smoothing_filter, mode="same")
    if np.isnan(np.sum(ts)):
        raise ValueError("NaN in probability series")
    peaks, extra = find_peaks(ts, height=thr, distance=min_distance)
    ampl = extra["peak_heights"]
    widths = peak_widths(ts, peaks, rel_height=0.5, prominence_data=None, wlen=None)[0]
    return peaks, widths, ampl, ts


def forced_choice_peak(ts, thr: float = 0.2, min_distance: int = 50) -> int | None:
    peaks, _, ampl, _ = extract_picks(ts, thr=thr, min_distance=min_distance)
    if len(peaks) == 0:
        i = int(np.argmax(ts))
        return i if float(ts[i]) >= thr else None
    return int(peaks[int(np.argmax(ampl))])


def s_f1_at_tolerance(
    pred_s: np.ndarray,
    true_s: np.ndarray,
    vis_s: np.ndarray,
    *,
    tol_samples: int = 50,
) -> dict[str, float]:
    """Forced-choice S F1 at sample tolerance (0.5 s @ 100 Hz → 50 samples)."""
    tp = fp = fn = 0
    for p, t, v in zip(pred_s, true_s, vis_s):
        if not bool(v) or not np.isfinite(t) or t < 0:
            continue
        if p is None or (isinstance(p, float) and not np.isfinite(p)):
            fn += 1
            continue
        if abs(float(p) - float(t)) <= tol_samples:
            tp += 1
        else:
            fp += 1
            fn += 1  # forced-choice miss+false at this event
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
