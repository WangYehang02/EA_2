"""Extract multiple PhaseNet candidate peaks for re-ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass
class PeakCandidate:
    sample_index: int
    absolute_utc: str | None
    peak_probability: float
    prominence: float
    peak_width: float
    local_entropy: float
    rank: int
    fallback_peak: bool = False
    phase: str = "p"


def _local_entropy(prob: np.ndarray, center: int, half_win: int = 50) -> float:
    lo = max(0, center - half_win)
    hi = min(len(prob), center + half_win + 1)
    seg = np.asarray(prob[lo:hi], dtype=np.float64)
    s = seg.sum()
    if s <= 0:
        return 0.0
    p = seg / s
    return float(-(p * np.log(p + 1e-12)).sum())


def extract_candidates(
    prob: np.ndarray,
    *,
    phase: str = "p",
    k: int = 5,
    min_distance: int = 50,
    min_prominence: float = 0.05,
    min_probability: float = 0.1,
    sampling_rate: float = 100.0,
    waveform_starttime: Any = None,
) -> list[PeakCandidate]:
    """Return up to K peaks; if none pass threshold, keep global argmax as fallback."""
    prob = np.asarray(prob, dtype=np.float64)
    n = len(prob)
    if n == 0:
        return []

    peaks, props = find_peaks(
        prob,
        distance=max(int(min_distance), 1),
        prominence=max(float(min_prominence), 0.0),
        height=max(float(min_probability), 0.0),
    )
    cands: list[PeakCandidate] = []
    if peaks.size:
        # sort by height descending
        heights = prob[peaks]
        order = np.argsort(-heights)[:k]
        peaks = peaks[order]
        try:
            proms = peak_prominences(prob, peaks)[0]
        except Exception:
            proms = np.zeros(len(peaks))
        try:
            widths = peak_widths(prob, peaks, rel_height=0.5)[0]
        except Exception:
            widths = np.zeros(len(peaks))
        for rank, (pk, pr, wd) in enumerate(zip(peaks, proms, widths)):
            utc = None
            if waveform_starttime is not None:
                try:
                    import pandas as pd

                    t0 = pd.Timestamp(waveform_starttime)
                    utc = str(t0 + pd.to_timedelta(float(pk) / float(sampling_rate), unit="s"))
                except Exception:
                    utc = None
            cands.append(
                PeakCandidate(
                    sample_index=int(pk),
                    absolute_utc=utc,
                    peak_probability=float(prob[pk]),
                    prominence=float(pr),
                    peak_width=float(wd),
                    local_entropy=_local_entropy(prob, int(pk)),
                    rank=int(rank),
                    fallback_peak=False,
                    phase=phase,
                )
            )
        return cands

    # fallback: global max
    pk = int(np.argmax(prob))
    utc = None
    if waveform_starttime is not None:
        try:
            import pandas as pd

            t0 = pd.Timestamp(waveform_starttime)
            utc = str(t0 + pd.to_timedelta(float(pk) / float(sampling_rate), unit="s"))
        except Exception:
            utc = None
    return [
        PeakCandidate(
            sample_index=pk,
            absolute_utc=utc,
            peak_probability=float(prob[pk]),
            prominence=float(prob[pk]),
            peak_width=float("nan"),
            local_entropy=_local_entropy(prob, pk),
            rank=0,
            fallback_peak=True,
            phase=phase,
        )
    ]


def candidates_to_records(cands: list[PeakCandidate], trace_name: str, event_id: str) -> list[dict[str, Any]]:
    rows = []
    for c in cands:
        d = asdict(c)
        d["trace_name"] = trace_name
        d["event_id"] = event_id
        rows.append(d)
    return rows
