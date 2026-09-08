#!/usr/bin/env python
"""Minimal tests for paper_strengthening_v1 invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.paper_strengthening import (
    C_CONFIGS,
    fit_linear_pairwise,
    pred_base_tau_c1c2,
    pred_fixed,
    pred_resid_control,
)


def _tiny_pairs(n=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ge2 = i % 3 != 0
        c1 = 1000.0 + i
        c2 = c1 + rng.normal(0, 5)
        rows.append(
            {
                "trace_name": f"t{i}",
                "event_id": f"e{i//2}",
                "n_candidates": 2 if ge2 else 1,
                "sampling_rate_hz": 100.0,
                "true_s_sample": c1,
                "c1_sample": c1,
                "c2_sample": c2 if ge2 else np.nan,
                "c1_fixed_score": 0.0,
                "c2_fixed_score": -0.1 if ge2 else np.nan,
                "c1_prob": 0.5,
                "c2_prob": 0.4 if ge2 else np.nan,
                "c1_resid_s": 0.1,
                "c2_resid_s": 0.8 if ge2 else np.nan,
                "c1_resid_sp": 0.0,
                "c2_resid_sp": 0.0 if ge2 else np.nan,
                "c1_stead_prob": 0.5,
                "c2_stead_prob": 0.4 if ge2 else 0.0,
                "c1_ida_prob": 0.4,
                "c2_ida_prob": 0.3 if ge2 else 0.0,
                "c1_delta_sp": 5.0,
                "c2_delta_sp": 5.0 if ge2 else np.nan,
                "c1_source": "stead",
                "c2_source": "ida" if ge2 else "",
                "y_choose_c2": 0 if ge2 and i % 2 == 0 else (1 if ge2 else np.nan),
            }
        )
    return pd.DataFrame(rows)


def test_c_grid_size_frozen():
    assert len(C_CONFIGS) == 20


def test_single_candidate_fallback_all_methods():
    p = _tiny_pairs()
    for pred in [
        pred_fixed(p),
        pred_resid_control(p),
        pred_base_tau_c1c2(p, sigma_s=0.5, lambda_history=2.0),
    ]:
        single = p["n_candidates"].to_numpy(int) < 2
        assert np.allclose(pred[single], p.loc[single, "c1_sample"].to_numpy(float))


def test_linear_no_intercept_and_label_isolation_features():
    p = _tiny_pairs(40)
    # force decisive
    p.loc[p["n_candidates"] >= 2, "y_choose_c2"] = 0
    p.loc[(p["n_candidates"] >= 2) & (p.index % 2 == 1), "y_choose_c2"] = 1
    m = fit_linear_pairwise(p, C=1.0)
    assert m.coef.shape == (10,)
    pred0, _ = m.predict(p, 0.5)
    p2 = p.copy()
    p2["true_s_sample"] = p2["true_s_sample"] + 9999
    pred1, _ = m.predict(p2, 0.5)
    assert np.allclose(pred0, pred1, equal_nan=True)
