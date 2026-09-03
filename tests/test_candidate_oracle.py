"""Oracle candidate chooses nearest peak to label within top-K."""

from __future__ import annotations

import numpy as np

from earthquake.fusion.peak_candidates import PeakCandidate


def test_oracle_candidate_nearest():
    cands = [
        PeakCandidate(1000, None, 0.9, 0.5, 10, 1.0, 0, False, "s"),
        PeakCandidate(1500, None, 0.4, 0.2, 8, 1.0, 1, False, "s"),
        PeakCandidate(2000, None, 0.3, 0.1, 8, 1.0, 2, False, "s"),
    ]
    true = 1490.0
    sr = 100.0
    K = 3
    top = cands[:K]
    best = min(top, key=lambda c: abs(c.sample_index - true))
    assert best.sample_index == 1500
    assert abs(best.sample_index - true) / sr <= 0.5
