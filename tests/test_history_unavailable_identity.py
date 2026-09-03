"""History unavailable must leave PhaseNet argmax pick unchanged."""

from __future__ import annotations

import numpy as np

from earthquake.fusion.candidate_rescorer import pick_argmax_probability, rescore_phase_candidates
from earthquake.fusion.peak_candidates import extract_candidates


def test_history_unavailable_identity():
    rng = np.random.default_rng(0)
    for _ in range(5):
        prob = rng.random(2000) * 0.2
        peaks = rng.choice(np.arange(100, 1900), size=3, replace=False)
        for i, pk in enumerate(peaks):
            prob[pk] = 0.5 + 0.1 * i
        cands = extract_candidates(prob, k=5, min_probability=0.3, min_prominence=0.05, min_distance=30)
        best, _ = rescore_phase_candidates(
            cands,
            expected_sample=float(peaks[0]),
            sigma_samples=10.0,
            lambda_wave=1.0,
            lambda_history=10.0,
            lambda_prominence=0.5,
            history_available=False,
        )
        assert best.sample_index == pick_argmax_probability(cands).sample_index
