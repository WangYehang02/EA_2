"""Path residual priors = observed tau - distance baseline, with shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from earthquake.history.shrinkage import shrink_residual, shrinkage_weight
from earthquake.history.temporal_store import PathStats, TemporalHistoryStore
from earthquake.history.travel_time_baseline import TravelTimeBaseline, _observed_taus
from earthquake.utils import mad


@dataclass
class ResidualPathStats:
    history_count: int = 0
    residual_p_median: float = float("nan")
    residual_p_mad: float = float("nan")
    residual_s_median: float = float("nan")
    residual_s_mad: float = float("nan")
    residual_sp_median: float = float("nan")
    residual_sp_mad: float = float("nan")
    history_available: bool = False
    matched_key: str | None = None
    fallback_level: int = -1
    # absolute tau stats kept for negative controls / legacy
    tau_p_median: float = float("nan")
    tau_s_median: float = float("nan")
    delta_sp_median: float = float("nan")
    tau_p_mad: float = float("nan")
    tau_s_mad: float = float("nan")
    delta_sp_mad: float = float("nan")


def residuals_from_obs_and_base(
    obs_tau_p: float,
    obs_tau_s: float,
    base_tau_p: float,
    base_tau_s: float,
) -> tuple[float, float, float]:
    rp = float(obs_tau_p - base_tau_p) if np.isfinite(obs_tau_p) and np.isfinite(base_tau_p) else float("nan")
    rs = float(obs_tau_s - base_tau_s) if np.isfinite(obs_tau_s) and np.isfinite(base_tau_s) else float("nan")
    rsp = float((obs_tau_s - obs_tau_p) - (base_tau_s - base_tau_p)) if all(
        np.isfinite(x) for x in (obs_tau_p, obs_tau_s, base_tau_p, base_tau_s)
    ) else float("nan")
    return rp, rs, rsp


def summarize_residuals(res_p: np.ndarray, res_s: np.ndarray, res_sp: np.ndarray) -> dict[str, float]:
    def _med_mad(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.median(x)), float(mad(x))

    pm, pd_ = _med_mad(res_p)
    sm, sd = _med_mad(res_s)
    spm, spd = _med_mad(res_sp)
    return {
        "residual_p_median": pm,
        "residual_p_mad": pd_,
        "residual_s_median": sm,
        "residual_s_mad": sd,
        "residual_sp_median": spm,
        "residual_sp_mad": spd,
        "history_count": int(np.isfinite(res_p).sum()),
    }


@dataclass
class ResidualPrediction:
    pred_tau_p: float
    pred_tau_s: float
    pred_delta_sp: float
    base_tau_p: float
    base_tau_s: float
    base_delta_sp: float
    shrunk_residual_p: float
    shrunk_residual_s: float
    shrunk_residual_sp: float
    weight: float
    used_path_residual: bool
    history_available: bool
    sigma_p_s: float
    sigma_s_s: float
    sigma_sp_s: float


def predict_with_residual_prior(
    base: dict[str, float],
    path: ResidualPathStats | None,
    global_residual: dict[str, float],
    shrinkage_k: float,
    min_history: int = 5,
    mad_disable_s: float = 1.0,
    min_sigma_s: float = 0.05,
    max_sigma_s: float = 1.0,
) -> ResidualPrediction:
    """Combine distance baseline + shrunk path residual."""
    bp = float(base.get("base_tau_p", np.nan))
    bs = float(base.get("base_tau_s", np.nan))
    bsp = float(base.get("base_delta_sp", np.nan))
    if not np.isfinite(bsp) and np.isfinite(bp) and np.isfinite(bs):
        bsp = bs - bp

    gp = float(global_residual.get("residual_p_median", 0.0) or 0.0)
    gs = float(global_residual.get("residual_s_median", 0.0) or 0.0)
    gsp = float(global_residual.get("residual_sp_median", 0.0) or 0.0)

    use_path = (
        path is not None
        and path.history_available
        and path.history_count >= min_history
        and (not np.isfinite(path.residual_p_mad) or path.residual_p_mad <= mad_disable_s)
    )
    if use_path and path is not None:
        w = shrinkage_weight(path.history_count, shrinkage_k)
        rp = shrink_residual(path.residual_p_median, gp, path.history_count, shrinkage_k)
        rs = shrink_residual(path.residual_s_median, gs, path.history_count, shrinkage_k)
        rsp = shrink_residual(path.residual_sp_median, gsp, path.history_count, shrinkage_k)
        hist_ok = True
        # sigma from residual MAD (fallback to absolute MAD / default)
        mad_p = path.residual_p_mad if np.isfinite(path.residual_p_mad) else path.tau_p_mad
        mad_s = path.residual_s_mad if np.isfinite(path.residual_s_mad) else path.tau_s_mad
        mad_sp = path.residual_sp_mad if np.isfinite(path.residual_sp_mad) else path.delta_sp_mad
    else:
        w = 0.0
        rp, rs, rsp = gp, gs, gsp
        hist_ok = False
        mad_p = mad_s = mad_sp = 0.2

    def _sig(m: float) -> float:
        if not np.isfinite(m):
            return float(min_sigma_s)
        return float(np.clip(1.4826 * m, min_sigma_s, max_sigma_s))

    return ResidualPrediction(
        pred_tau_p=bp + rp if np.isfinite(bp) else float("nan"),
        pred_tau_s=bs + rs if np.isfinite(bs) else float("nan"),
        pred_delta_sp=bsp + rsp if np.isfinite(bsp) else float("nan"),
        base_tau_p=bp,
        base_tau_s=bs,
        base_delta_sp=bsp,
        shrunk_residual_p=rp,
        shrunk_residual_s=rs,
        shrunk_residual_sp=rsp,
        weight=w,
        used_path_residual=bool(use_path),
        history_available=hist_ok,
        sigma_p_s=_sig(mad_p),
        sigma_s_s=_sig(mad_s),
        sigma_sp_s=_sig(mad_sp),
    )


def expected_samples_catalog(
    row: pd.Series,
    pred: ResidualPrediction,
) -> tuple[float, float]:
    """Convert predicted absolute travel times to waveform sample indices using origin time."""
    if not np.isfinite(pred.pred_tau_p):
        return float("nan"), float("nan")
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    p_abs = origin + pd.to_timedelta(pred.pred_tau_p, unit="s")
    s_abs = origin + pd.to_timedelta(pred.pred_tau_s, unit="s")
    return float((p_abs - start).total_seconds() * sr), float((s_abs - start).total_seconds() * sr)


def path_stats_to_residual_stats(stats: PathStats, base: dict[str, float]) -> ResidualPathStats:
    """Convert absolute path medians into residual medians vs a baseline prediction."""
    bp, bs = float(base.get("base_tau_p", np.nan)), float(base.get("base_tau_s", np.nan))
    bsp = float(base.get("base_delta_sp", np.nan))
    if not np.isfinite(bsp) and np.isfinite(bp) and np.isfinite(bs):
        bsp = bs - bp
    rp = float(stats.tau_p_median - bp) if stats.history_available and np.isfinite(stats.tau_p_median) and np.isfinite(bp) else float("nan")
    rs = float(stats.tau_s_median - bs) if stats.history_available and np.isfinite(stats.tau_s_median) and np.isfinite(bs) else float("nan")
    rsp = (
        float(stats.delta_sp_median - bsp)
        if stats.history_available and np.isfinite(stats.delta_sp_median) and np.isfinite(bsp)
        else float("nan")
    )
    return ResidualPathStats(
        history_count=int(stats.history_count),
        residual_p_median=rp,
        residual_p_mad=float(stats.tau_p_mad) if np.isfinite(stats.tau_p_mad) else float("nan"),
        residual_s_median=rs,
        residual_s_mad=float(stats.tau_s_mad) if np.isfinite(stats.tau_s_mad) else float("nan"),
        residual_sp_median=rsp,
        residual_sp_mad=float(stats.delta_sp_mad) if np.isfinite(stats.delta_sp_mad) else float("nan"),
        history_available=bool(stats.history_available),
        matched_key=stats.matched_key,
        fallback_level=int(stats.fallback_level),
        tau_p_median=float(stats.tau_p_median),
        tau_s_median=float(stats.tau_s_median),
        delta_sp_median=float(stats.delta_sp_median),
        tau_p_mad=float(stats.tau_p_mad),
        tau_s_mad=float(stats.tau_s_mad),
        delta_sp_mad=float(stats.delta_sp_mad),
    )


def attach_baseline_and_residuals(
    df: pd.DataFrame,
    baseline: TravelTimeBaseline,
) -> pd.DataFrame:
    """Add obs/base/residual columns for analysis (labels required for obs)."""
    obs = _observed_taus(df)
    base = baseline.predict_frame(obs)
    out = obs.copy()
    for c in base.columns:
        out[c] = base[c].to_numpy()
    out["residual_tau_p"] = out["obs_tau_p"] - out["base_tau_p"]
    out["residual_tau_s"] = out["obs_tau_s"] - out["base_tau_s"]
    out["residual_delta_sp"] = out["obs_delta_sp"] - out["base_delta_sp"]
    return out
