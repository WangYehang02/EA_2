"""Candidate re-scoring tests."""

from __future__ import annotations

from earthquake.fusion.blind_pair_rescorer import rescore_blind_pairs
from earthquake.fusion.candidate_rescorer import pick_argmax_probability, rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate, extract_candidates
import numpy as np


def test_extract_candidates_fallback():
    prob = np.zeros(1000)
    prob[100] = 0.05  # below min_probability
    cands = extract_candidates(prob, min_probability=0.3, min_prominence=0.01, k=3)
    assert len(cands) == 1
    assert cands[0].fallback_peak is True
    assert cands[0].sample_index == 100


def test_history_rescoring_prefers_near_expected():
    cands = [
        PeakCandidate(50, None, 0.9, 0.9, 10.0, 1.0, 0, False, "s"),
        PeakCandidate(200, None, 0.8, 0.8, 10.0, 1.0, 1, False, "s"),
    ]
    best, rows = rescore_phase_candidates(
        cands,
        expected_sample=200.0,
        sigma_samples=20.0,
        lambda_wave=1.0,
        lambda_history=2.0,
        lambda_prominence=0.0,
        history_available=True,
    )
    assert best is not None
    assert best.sample_index == 200


def test_history_unavailable_identity_to_argmax():
    cands = [
        PeakCandidate(50, None, 0.95, 0.9, 10.0, 1.0, 0, False, "p"),
        PeakCandidate(200, None, 0.5, 0.4, 10.0, 1.0, 1, False, "p"),
    ]
    best, _ = rescore_phase_candidates(
        cands,
        expected_sample=200.0,
        sigma_samples=10.0,
        lambda_wave=1.0,
        lambda_history=5.0,
        lambda_prominence=0.0,
        history_available=False,
    )
    assert best is not None
    assert best.sample_index == pick_argmax_probability(cands).sample_index


def test_blind_pair_requires_s_after_p():
    p = [PeakCandidate(100, None, 0.8, 0.5, 5.0, 1.0, 0, False, "p")]
    s = [PeakCandidate(80, None, 0.9, 0.5, 5.0, 1.0, 0, False, "s"), PeakCandidate(250, None, 0.7, 0.5, 5.0, 1.0, 1, False, "s")]
    out = rescore_blind_pairs(
        p,
        s,
        predicted_delta_sp_s=1.5,
        sigma_sp_s=0.5,
        sampling_rate=100.0,
        lambda_wave=1.0,
        lambda_history=2.0,
        history_available=True,
    )
    assert out["s_cand"].sample_index == 250
