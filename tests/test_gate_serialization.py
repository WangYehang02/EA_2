"""Scaler / schema serialization roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earthquake.gating.features import FEATURE_COLUMNS_CATALOG, fit_scaler, load_scaler, save_scaler


def test_scaler_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=50) for c in FEATURE_COLUMNS_CATALOG})
    sc = fit_scaler(df, FEATURE_COLUMNS_CATALOG)
    path = tmp_path / "feature_scaler.pkl"
    save_scaler(sc, path)
    sc2 = load_scaler(path)
    assert sc2.columns == sc.columns
    assert sc2.schema_hash == sc.schema_hash
    X1, M1 = sc.transform(df)
    X2, M2 = sc2.transform(df)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(M1, M2)
