from __future__ import annotations

import numpy as np

from earthquake.metrics import match_picks, noise_false_positive_rate


def test_perfect_picks_f1():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([100.0, 200.0, 300.0])
    m = match_picks(pred, true, sampling_rate=100.0)
    assert m["f1@0.1s"] == 1.0
    assert m["mae"] == 0.0


def test_noise_fpr():
    assert noise_false_positive_rate(np.array([True, False, False, True])) == 0.5
