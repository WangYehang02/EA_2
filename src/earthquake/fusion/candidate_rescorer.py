"""History-consistency re-scoring of PhaseNet candidate peaks (catalog-assisted)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from earthquake.fusion.peak_candidates import PeakCandidate


def history_consistency(candidate_sample: float, expected_sample: float, sigma_samples: float) -> float:
    if not np.isfinite(candidate_sample) or not np.isfinite(expected_sample):
        return 0.0
    sig = float(sigma_samples) if np.isfinite(sigma_samples) and sigma_samples > 1e-6 else 1.0
    return float(np.exp(-0.5 * ((candidate_sample - expected_sample) / sig) ** 2))


def score_candidate(
    cand: PeakCandidate,
    *,
    expected_sample: float,
    sigma_samples: float,
    lambda_wave: float,
    lambda_history: float,
    lambda_prominence: float,
    max_prominence: float,
    history_available: bool,
    eps: float = 1e-8,
) -> float:
    """score = λw log(p+eps) + λh log(hist+eps) + λpr * norm_prominence."""
    lh = float(lambda_history) if history_available else 0.0
    hist = history_consistency(cand.sample_index, expected_sample, sigma_samples)
    prom = float(cand.prominence) if np.isfinite(cand.prominence) else 0.0
    norm_pr = prom / max(float(max_prominence), eps)
    return float(
        float(lambda_wave) * np.log(float(cand.peak_probability) + eps)
        + lh * np.log(hist + eps)
        + float(lambda_prominence) * norm_pr
    )


def rescore_phase_candidates(
    cands: list[PeakCandidate],
    *,
    expected_sample: float,
    sigma_samples: float,
    lambda_wave: float = 1.0,
    lambda_history: float = 0.5,
    lambda_prominence: float = 0.1,
    history_available: bool = True,
) -> tuple[PeakCandidate | None, list[dict]]:
    """Return best candidate and per-candidate score rows."""
    if not cands:
        return None, []
    max_pr = max((c.prominence for c in cands if np.isfinite(c.prominence)), default=1.0)
    rows = []
    best = None
    best_score = -float("inf")
    for c in cands:
        sc = score_candidate(
            c,
            expected_sample=expected_sample,
            sigma_samples=sigma_samples,
            lambda_wave=lambda_wave,
            lambda_history=lambda_history,
            lambda_prominence=lambda_prominence,
            max_prominence=max_pr if max_pr > 0 else 1.0,
            history_available=history_available,
        )
        rows.append(
            {
                "sample_index": c.sample_index,
                "peak_probability": c.peak_probability,
                "prominence": c.prominence,
                "rank": c.rank,
                "fallback_peak": c.fallback_peak,
                "score": sc,
                "history_score": history_consistency(c.sample_index, expected_sample, sigma_samples),
            }
        )
        if sc > best_score:
            best_score = sc
            best = c
    return best, rows


def pick_argmax_probability(cands: Iterable[PeakCandidate]) -> PeakCandidate | None:
    cands = list(cands)
    if not cands:
        return None
    return max(cands, key=lambda c: c.peak_probability)
