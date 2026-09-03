"""Shrinkage unit tests."""

from __future__ import annotations

from earthquake.history.shrinkage import shrink_residual, shrinkage_weight


def test_shrinkage_weight_limits():
    assert shrinkage_weight(0, 10) == 0.0
    assert abs(shrinkage_weight(10, 10) - 0.5) < 1e-9
    assert abs(shrinkage_weight(1000, 10) - 1000 / 1010) < 1e-9


def test_shrink_residual_interpolates():
    # n=k -> halfway
    v = shrink_residual(path_residual=2.0, regional_or_global_residual=0.0, history_count=10, k=10)
    assert abs(v - 1.0) < 1e-9
    # no history -> global
    v2 = shrink_residual(path_residual=2.0, regional_or_global_residual=-1.0, history_count=0, k=5)
    assert abs(v2 - (-1.0)) < 1e-9
