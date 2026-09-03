"""Empirical-Bayes shrinkage of path residuals toward regional/global means."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def shrinkage_weight(history_count: float, k: float) -> float:
    """weight = n / (n + k); n<=0 -> 0."""
    n = float(history_count)
    kk = float(k)
    if not np.isfinite(n) or n <= 0:
        return 0.0
    if not np.isfinite(kk) or kk < 0:
        raise ValueError(f"shrinkage_k must be >=0, got {k}")
    return float(n / (n + kk))


def shrink_residual(
    path_residual: float,
    regional_or_global_residual: float,
    history_count: float,
    k: float,
) -> float:
    """shrunk = w * path + (1-w) * fallback."""
    w = shrinkage_weight(history_count, k)
    p = float(path_residual) if np.isfinite(path_residual) else float("nan")
    g = float(regional_or_global_residual) if np.isfinite(regional_or_global_residual) else 0.0
    if not np.isfinite(p):
        return g
    return float(w * p + (1.0 - w) * g)


def search_shrinkage_k(
    residuals_path: np.ndarray,
    residuals_fallback: np.ndarray,
    history_counts: np.ndarray,
    observed: np.ndarray,
    base: np.ndarray,
    ks: Iterable[float] = (1, 2, 5, 10, 20, 50),
) -> dict[str, float]:
    """Pick k minimizing MAE of base + shrunk residual vs observed (finite rows only)."""
    path = np.asarray(residuals_path, dtype=np.float64)
    fb = np.asarray(residuals_fallback, dtype=np.float64)
    n = np.asarray(history_counts, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(b)
    best_k = float("nan")
    best_mae = float("inf")
    rows = []
    for k in ks:
        pred = np.array(
            [b[i] + shrink_residual(path[i], fb[i] if np.isfinite(fb[i]) else 0.0, n[i], float(k)) for i in range(len(y))],
            dtype=np.float64,
        )
        mae = float(np.mean(np.abs(pred[mask] - y[mask]))) if mask.any() else float("nan")
        rows.append({"k": float(k), "mae": mae})
        if np.isfinite(mae) and mae < best_mae:
            best_mae = mae
            best_k = float(k)
    return {"best_k": best_k, "best_mae": best_mae, "grid": rows}
