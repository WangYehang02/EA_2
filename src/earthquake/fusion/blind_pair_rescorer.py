"""Blind-S joint re-scoring of P/S candidate pairs using historical S-P residual prior."""

from __future__ import annotations

from typing import Any

import numpy as np

from earthquake.fusion.peak_candidates import PeakCandidate


def score_ps_pair(
    p_cand: PeakCandidate,
    s_cand: PeakCandidate,
    *,
    predicted_delta_sp_s: float,
    sigma_sp_s: float,
    sampling_rate: float,
    lambda_wave: float,
    lambda_history: float,
    lambda_prominence: float,
    history_available: bool,
    min_sp_s: float = 0.5,
    max_sp_s: float = 80.0,
    eps: float = 1e-8,
) -> float | None:
    """Return pair score or None if physically invalid.

    Blind mode uses only PhaseNet candidates + predicted Δ(S-P); no origin time.
    """
    if s_cand.sample_index <= p_cand.sample_index:
        return None
    sp_s = (s_cand.sample_index - p_cand.sample_index) / float(sampling_rate)
    if not (min_sp_s <= sp_s <= max_sp_s):
        return None
    if not np.isfinite(predicted_delta_sp_s):
        hist = 0.0
        lh = 0.0
    else:
        sig = float(sigma_sp_s) if np.isfinite(sigma_sp_s) and sigma_sp_s > 1e-6 else 0.5
        hist = float(np.exp(-0.5 * ((sp_s - float(predicted_delta_sp_s)) / sig) ** 2))
        lh = float(lambda_history) if history_available else 0.0
    max_pr = max(p_cand.prominence, s_cand.prominence, 1e-8)
    score = (
        float(lambda_wave) * (np.log(p_cand.peak_probability + eps) + np.log(s_cand.peak_probability + eps))
        + lh * np.log(hist + eps)
        + float(lambda_prominence) * ((p_cand.prominence + s_cand.prominence) / (2.0 * max_pr))
    )
    return float(score)


def rescore_blind_pairs(
    p_cands: list[PeakCandidate],
    s_cands: list[PeakCandidate],
    *,
    predicted_delta_sp_s: float,
    sigma_sp_s: float,
    sampling_rate: float,
    lambda_wave: float = 1.0,
    lambda_history: float = 0.5,
    lambda_prominence: float = 0.1,
    history_available: bool = True,
    min_sp_s: float = 0.5,
    max_sp_s: float = 80.0,
) -> dict[str, Any]:
    best = None
    best_score = -float("inf")
    n_valid = 0
    for pc in p_cands:
        for sc in s_cands:
            sc_val = score_ps_pair(
                pc,
                sc,
                predicted_delta_sp_s=predicted_delta_sp_s,
                sigma_sp_s=sigma_sp_s,
                sampling_rate=sampling_rate,
                lambda_wave=lambda_wave,
                lambda_history=lambda_history,
                lambda_prominence=lambda_prominence,
                history_available=history_available,
                min_sp_s=min_sp_s,
                max_sp_s=max_sp_s,
            )
            if sc_val is None:
                continue
            n_valid += 1
            if sc_val > best_score:
                best_score = sc_val
                best = (pc, sc, sc_val)
    if best is None:
        # fall back to independent argmax
        pc = max(p_cands, key=lambda c: c.peak_probability) if p_cands else None
        sc = max(s_cands, key=lambda c: c.peak_probability) if s_cands else None
        return {
            "p_cand": pc,
            "s_cand": sc,
            "score": float("nan"),
            "n_valid_pairs": 0,
            "fallback_independent": True,
        }
    pc, sc, scv = best
    return {
        "p_cand": pc,
        "s_cand": sc,
        "score": scv,
        "n_valid_pairs": n_valid,
        "fallback_independent": False,
    }
