from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.history.temporal_store import PathStats


def gaussian_prior(
    n_samples: int,
    center_sample: float,
    sigma_samples: float,
) -> np.ndarray:
    """Peak-normalized Gaussian prior (max=1), compatible with PhaseNet score scale."""
    t = np.arange(n_samples, dtype=np.float64)
    if not np.isfinite(center_sample) or not np.isfinite(sigma_samples) or sigma_samples <= 0:
        return np.ones(n_samples, dtype=np.float32)  # flat; fusion should force PhaseNet when unavailable
    prior = np.exp(-0.5 * ((t - center_sample) / sigma_samples) ** 2)
    m = prior.max()
    if m > 0:
        prior = prior / m
    return prior.astype(np.float32)


def sigma_from_mad(mad_seconds: float, sampling_rate: float, min_sigma_s: float = 0.05, max_sigma_s: float = 1.0) -> float:
    if not np.isfinite(mad_seconds):
        sigma_s = min_sigma_s
    else:
        sigma_s = float(np.clip(1.4826 * mad_seconds, min_sigma_s, max_sigma_s))
    return sigma_s * float(sampling_rate)


def catalog_assisted_centers(
    row: pd.Series,
    stats: PathStats,
) -> tuple[float, float]:
    """Return prior P/S sample centers using current origin time + historical tau."""
    if not stats.history_available:
        return float("nan"), float("nan")
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    prior_p_time = origin + pd.to_timedelta(stats.tau_p_median, unit="s")
    prior_s_time = origin + pd.to_timedelta(stats.tau_s_median, unit="s")
    prior_p_sample = (prior_p_time - start).total_seconds() * sr
    prior_s_sample = (prior_s_time - start).total_seconds() * sr
    return float(prior_p_sample), float(prior_s_sample)


def blind_s_center(
    phasenet_p_sample: float,
    stats: PathStats,
    sampling_rate: float,
) -> float:
    if not stats.history_available or not np.isfinite(phasenet_p_sample):
        return float("nan")
    return float(phasenet_p_sample + stats.delta_sp_median * sampling_rate)


def build_priors_for_row(
    row: pd.Series,
    stats: PathStats,
    n_samples: int,
    mode: str = "catalog_assisted",
    phasenet_p_sample: float | None = None,
    min_sigma_s: float = 0.05,
    max_sigma_s: float = 1.0,
) -> dict[str, np.ndarray | bool | float]:
    sr = float(row.get("sampling_rate_hz", 100.0))
    if not stats.history_available:
        uni = np.ones(n_samples, dtype=np.float32)
        return {
            "prior_p": uni,
            "prior_s": uni,
            "history_available": False,
            "prior_p_sample": float("nan"),
            "prior_s_sample": float("nan"),
            "force_phasenet": True,
        }

    if mode == "catalog_assisted":
        cp, cs = catalog_assisted_centers(row, stats)
        sp = sigma_from_mad(stats.tau_p_mad, sr, min_sigma_s, max_sigma_s)
        ss = sigma_from_mad(stats.tau_s_mad, sr, min_sigma_s, max_sigma_s)
        return {
            "prior_p": gaussian_prior(n_samples, cp, sp),
            "prior_s": gaussian_prior(n_samples, cs, ss),
            "history_available": True,
            "prior_p_sample": cp,
            "prior_s_sample": cs,
            "force_phasenet": False,
        }

    if mode == "blind_s":
        if phasenet_p_sample is None:
            raise ValueError("blind_s mode requires phasenet_p_sample")
        cs = blind_s_center(phasenet_p_sample, stats, sr)
        ss = sigma_from_mad(stats.delta_sp_mad, sr, min_sigma_s, max_sigma_s)
        uni = np.ones(n_samples, dtype=np.float32)
        return {
            "prior_p": uni,  # cannot constrain P without origin
            "prior_s": gaussian_prior(n_samples, cs, ss),
            "history_available": True,
            "prior_p_sample": float("nan"),
            "prior_s_sample": cs,
            "force_phasenet": False,
        }

    raise ValueError(f"Unknown prior mode: {mode}")
