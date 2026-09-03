from __future__ import annotations

import numpy as np


def linear_fusion(
    phasenet: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    force_phasenet: bool = False,
) -> np.ndarray:
    """final = alpha * phasenet + (1-alpha) * prior. alpha=1 => PhaseNet only."""
    if force_phasenet:
        return np.asarray(phasenet, dtype=np.float32)
    a = float(alpha)
    out = a * np.asarray(phasenet, dtype=np.float64) + (1.0 - a) * np.asarray(prior, dtype=np.float64)
    return out.astype(np.float32)


def log_space_fusion(
    phasenet: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    eps: float = 1e-8,
    force_phasenet: bool = False,
) -> np.ndarray:
    if force_phasenet:
        return np.asarray(phasenet, dtype=np.float32)
    a = float(alpha)
    log_final = a * np.log(np.asarray(phasenet, dtype=np.float64) + eps) + (1.0 - a) * np.log(
        np.asarray(prior, dtype=np.float64) + eps
    )
    final = np.exp(log_final - log_final.max())
    s = final.sum()
    if s > 0:
        final = final / s
    return final.astype(np.float32)
