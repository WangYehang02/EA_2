"""Gate feature construction and robust scaling (train-fit only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS_CATALOG = [
    # waveform / PhaseNet
    "top1_s_probability",
    "top2_s_probability",
    "top1_top2_probability_gap",
    "top1_prominence",
    "top2_prominence",
    "top1_width",
    "top1_local_entropy",
    "global_s_entropy",
    "number_of_s_candidates",
    "fallback_peak",
    "p_peak_probability",
    "p_peak_entropy",
    "predicted_p_to_top1_s_gap",
    # history quality
    "log1p_history_count",
    "history_residual_s_median",
    "history_residual_s_mad",
    "history_delta_sp_median",
    "history_delta_sp_mad",
    "history_time_gap",
    "nearest_historical_source_distance_km",
    "shrinkage_weight",
    "history_available",
    "history_fallback_level",
    "history_sigma",
    "travel_time_baseline_uncertainty",
    # consistency
    "abs_top1_s_minus_expected",
    "normalized_top1_history_z",
    "abs_top2_s_minus_expected",
    "normalized_top2_history_z",
    "phase_top1_vs_history_expected_gap",
    "predicted_sp_vs_history_sp_gap",
    # catalog-assisted only
    "epicentral_distance",
    "source_depth",
    "station_elevation",
]

FEATURE_COLUMNS_BLIND = [c for c in FEATURE_COLUMNS_CATALOG if c not in {"epicentral_distance", "source_depth", "station_elevation"}]


FORBIDDEN_INPUT_SUBSTRINGS = [
    "true_",
    "label",
    "arrival_sample",
    "arrival_time",
    "manual",
    "oracle",
    "error_s",
    "snr_db",  # INSTANCE SNR often uses manual P window; eval-only
]


@dataclass
class FeatureScaler:
    columns: list[str]
    medians: dict[str, float]
    iqrs: dict[str, float]
    clip_lo: dict[str, float]
    clip_hi: dict[str, float]
    schema_hash: str

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, missing_mask) float32 arrays."""
        x = np.zeros((len(df), len(self.columns)), dtype=np.float32)
        m = np.zeros_like(x)
        for j, c in enumerate(self.columns):
            raw = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) if c in df.columns else np.full(len(df), np.nan)
            miss = ~np.isfinite(raw)
            m[:, j] = miss.astype(np.float32)
            fill = self.medians[c]
            raw = np.where(miss, fill, raw)
            lo, hi = self.clip_lo[c], self.clip_hi[c]
            raw = np.clip(raw, lo, hi)
            iqr = self.iqrs[c] if self.iqrs[c] > 1e-8 else 1.0
            x[:, j] = ((raw - self.medians[c]) / iqr).astype(np.float32)
        return x, m


def schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()[:16]


def assert_no_forbidden_columns(columns: list[str]) -> None:
    for c in columns:
        cl = c.lower()
        for bad in FORBIDDEN_INPUT_SUBSTRINGS:
            if bad in cl and c not in {"history_available", "fallback_peak"}:
                # allow fallback_peak (PhaseNet candidate flag), history_available
                if bad == "snr_db" or bad in cl:
                    if bad == "snr_db" and "snr" in cl:
                        raise ValueError(f"Forbidden feature column (label/SNR leakage risk): {c}")
                    if bad != "snr_db" and bad in cl:
                        raise ValueError(f"Forbidden feature column: {c}")


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> FeatureScaler:
    assert_no_forbidden_columns(columns)
    medians, iqrs, clo, chi = {}, {}, {}, {}
    for c in columns:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) if c in df.columns else np.array([])
        v = v[np.isfinite(v)]
        if v.size == 0:
            medians[c] = 0.0
            iqrs[c] = 1.0
            clo[c] = -1e6
            chi[c] = 1e6
            continue
        medians[c] = float(np.median(v))
        q25, q75 = np.percentile(v, [25, 75])
        iqrs[c] = float(max(q75 - q25, 1e-6))
        clo[c] = float(np.percentile(v, 0.5))
        chi[c] = float(np.percentile(v, 99.5))
    return FeatureScaler(
        columns=list(columns),
        medians=medians,
        iqrs=iqrs,
        clip_lo=clo,
        clip_hi=chi,
        schema_hash=schema_hash(columns),
    )


def save_scaler(scaler: FeatureScaler, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    schema = {
        "columns": scaler.columns,
        "schema_hash": scaler.schema_hash,
        "n_features": len(scaler.columns),
    }
    path.with_suffix(".json").write_text(json.dumps(schema, indent=2))
    # also feature_schema.json expected path companion
    (path.parent / "feature_schema.json").write_text(json.dumps(schema, indent=2))


def load_scaler(path: Path) -> FeatureScaler:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def build_trace_features(
    row: pd.Series,
    s_cands: list[dict[str, Any]],
    *,
    expected_s_sample: float,
    history_sigma_samples: float,
    sampling_rate: float,
    mode: str = "catalog",
) -> dict[str, float]:
    """Build per-trace gate features from candidates + residual history fields on row.

    Labels must not be used here.
    """
    sr = float(sampling_rate)
    cands = sorted(s_cands, key=lambda c: -float(c.get("peak_probability", 0.0)))
    def _get(i, key, default=np.nan):
        if i >= len(cands):
            return default
        v = cands[i].get(key, default)
        return float(v) if v is not None and np.isfinite(float(v)) else default

    top1_p = _get(0, "peak_probability", 0.0)
    top2_p = _get(1, "peak_probability", 0.0)
    top1_s = _get(0, "sample_index", np.nan)
    top2_s = _get(1, "sample_index", np.nan)
    p_sample = float(row.get("pred_p_sample", np.nan))
    hist_count = float(row.get("history_count", 0) or 0)
    hist_avail = 1.0 if bool(row.get("history_available", False)) else 0.0
    mad = float(row.get("residual_s_mad", row.get("tau_s_mad", np.nan)))
    sigma = float(history_sigma_samples) if np.isfinite(history_sigma_samples) and history_sigma_samples > 1e-6 else max(sr * 0.2, 1.0)
    z1 = (top1_s - expected_s_sample) / sigma if np.isfinite(top1_s) and np.isfinite(expected_s_sample) else np.nan
    z2 = (top2_s - expected_s_sample) / sigma if np.isfinite(top2_s) and np.isfinite(expected_s_sample) else np.nan
    # global entropy proxy from candidate probs
    probs = np.array([float(c.get("peak_probability", 0.0)) for c in cands], dtype=np.float64)
    if probs.size and probs.sum() > 0:
        p = probs / probs.sum()
        gent = float(-(p * np.log(p + 1e-12)).sum())
    else:
        gent = 0.0
    shrink_k = float(row.get("shrinkage_k", 50.0))
    shrink_w = hist_count / (hist_count + shrink_k) if hist_count > 0 else 0.0

    feat = {
        "top1_s_probability": top1_p,
        "top2_s_probability": top2_p,
        "top1_top2_probability_gap": top1_p - top2_p,
        "top1_prominence": _get(0, "prominence", 0.0),
        "top2_prominence": _get(1, "prominence", 0.0),
        "top1_width": _get(0, "peak_width", np.nan),
        "top1_local_entropy": _get(0, "local_entropy", np.nan),
        "global_s_entropy": gent,
        "number_of_s_candidates": float(len(cands)),
        "fallback_peak": 1.0 if (cands and bool(cands[0].get("fallback_peak", False))) else 0.0,
        "p_peak_probability": float(row.get("p_peak_probability", np.nan)),
        "p_peak_entropy": float(row.get("p_peak_entropy", np.nan)),
        "predicted_p_to_top1_s_gap": (top1_s - p_sample) / sr if np.isfinite(top1_s) and np.isfinite(p_sample) else np.nan,
        "log1p_history_count": float(np.log1p(max(hist_count, 0.0))),
        "history_residual_s_median": float(row.get("residual_s_median", np.nan)),
        "history_residual_s_mad": mad,
        "history_delta_sp_median": float(row.get("residual_sp_median", row.get("delta_sp_median", np.nan))),
        "history_delta_sp_mad": float(row.get("residual_sp_mad", row.get("delta_sp_mad", np.nan))),
        "history_time_gap": float(row.get("time_since_last_history", np.nan)),
        "nearest_historical_source_distance_km": float(row.get("nearest_historical_source_distance_km", np.nan)),
        "shrinkage_weight": float(shrink_w),
        "history_available": hist_avail,
        "history_fallback_level": float(row.get("fallback_level", -1) if pd.notna(row.get("fallback_level", np.nan)) else -1),
        "history_sigma": sigma / sr,
        "travel_time_baseline_uncertainty": float(row.get("travel_time_baseline_uncertainty", mad if np.isfinite(mad) else 0.2)),
        "abs_top1_s_minus_expected": abs(top1_s - expected_s_sample) / sr if np.isfinite(top1_s) and np.isfinite(expected_s_sample) else np.nan,
        "normalized_top1_history_z": float(z1) if np.isfinite(z1) else np.nan,
        "abs_top2_s_minus_expected": abs(top2_s - expected_s_sample) / sr if np.isfinite(top2_s) and np.isfinite(expected_s_sample) else np.nan,
        "normalized_top2_history_z": float(z2) if np.isfinite(z2) else np.nan,
        "phase_top1_vs_history_expected_gap": (top1_s - expected_s_sample) / sr if np.isfinite(top1_s) and np.isfinite(expected_s_sample) else np.nan,
        "predicted_sp_vs_history_sp_gap": (
            ((top1_s - p_sample) / sr) - float(row.get("pred_delta_sp", row.get("base_delta_sp", np.nan)))
            if np.isfinite(top1_s) and np.isfinite(p_sample)
            else np.nan
        ),
        "epicentral_distance": float(row.get("distance_km", np.nan)),
        "source_depth": float(row.get("source_depth_km", np.nan)),
        "station_elevation": float(row.get("station_elevation_m", np.nan)),
    }
    cols = FEATURE_COLUMNS_CATALOG if mode == "catalog" else FEATURE_COLUMNS_BLIND
    return {c: feat.get(c, np.nan) for c in cols}
