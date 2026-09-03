from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.history.prior import build_priors_for_row, gaussian_prior
from earthquake.history.temporal_store import PathStats


def test_gaussian_prior_normalized():
    p = gaussian_prior(1000, 200, 10)
    assert np.isclose(p.max(), 1.0, atol=1e-5)
    assert p.argmax() == 200


def test_unavailable_history_uniform_and_force():
    row = pd.Series(
        {
            "origin_time": pd.Timestamp("2020-01-01", tz="UTC"),
            "trace_start_time": pd.Timestamp("2020-01-01", tz="UTC"),
            "sampling_rate_hz": 100.0,
        }
    )
    stats = PathStats(history_available=False)
    out = build_priors_for_row(row, stats, n_samples=500)
    assert out["force_phasenet"] is True
    assert out["prior_p"].shape == (500,)
