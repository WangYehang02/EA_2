"""Stage 10 unified protocol helpers: splits, channel maps, leakage guards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ROOT / "artifacts/results/stage6/splits_full"

# Forbidden feature names for Ours-Blind (catalog / labels)
BLIND_FORBIDDEN_COLUMNS = frozenset(
    {
        "source_latitude",
        "source_longitude",
        "source_depth_km",
        "origin_time",
        "distance_km",
        "hyp_distance_km",
        "path_travel_time_p_s",
        "path_travel_time_s_s",
        "snr_db",
        "p_arrival_sample",
        "s_arrival_sample",
        "p_arrival_time",
        "s_arrival_time",
        "history_median",
        "history_mad",
        "history_count",
        "shrinkage",
    }
)


def load_event_ids(split: str) -> set[str]:
    name = split if split.startswith("stage6_") else f"stage6_{split}"
    path = SPLITS / f"{name}_events.txt"
    return set(path.read_text().splitlines())


def load_trace_names(split: str) -> set[str]:
    name = split if split.startswith("stage6_") else f"stage6_{split}"
    path = SPLITS / f"{name}_traces.txt"
    return set(path.read_text().splitlines())


def assert_event_disjoint(a: Iterable[str], b: Iterable[str], *, label: str = "") -> None:
    inter = set(a) & set(b)
    if inter:
        raise RuntimeError(f"event leakage {label}: n={len(inter)} e.g. {list(inter)[:5]}")


def enz_to_zne(wave_enz: np.ndarray) -> np.ndarray:
    """INSTANCE ENZ (3,T) → ZNE (3,T)."""
    e, n, z = wave_enz[0], wave_enz[1], wave_enz[2]
    return np.stack([z, n, e], axis=0).astype(np.float32)


def enz_to_ud_ns_ew(wave_enz: np.ndarray) -> np.ndarray:
    """INSTANCE ENZ → SegPhase UD,NS,EW = Z,N,E."""
    return enz_to_zne(wave_enz)


def assert_unique_trace_names(df: pd.DataFrame, col: str = "trace_name") -> None:
    if df[col].astype(str).duplicated().any():
        raise RuntimeError(f"duplicate {col}")


def join_predictions_by_trace_name(manifest: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One-to-one keyed join; order follows manifest."""
    assert_unique_trace_names(manifest)
    assert_unique_trace_names(preds)
    m = manifest.copy()
    m["trace_name"] = m["trace_name"].astype(str)
    p = preds.copy()
    p["trace_name"] = p["trace_name"].astype(str)
    missing = set(m["trace_name"]) - set(p["trace_name"])
    if missing:
        raise RuntimeError(f"missing predictions for {len(missing)} traces")
    extra = set(p["trace_name"]) - set(m["trace_name"])
    if extra:
        raise RuntimeError(f"extra predictions not in manifest: {len(extra)}")
    out = m.merge(p, on="trace_name", how="left", suffixes=("", "_pred"))
    if len(out) != len(m):
        raise RuntimeError("join changed length")
    if out["trace_name"].tolist() != m["trace_name"].tolist():
        raise RuntimeError("join did not preserve manifest order")
    return out


def assert_blind_feature_frame(df: pd.DataFrame) -> None:
    bad = BLIND_FORBIDDEN_COLUMNS & set(df.columns)
    if bad:
        raise RuntimeError(f"Ours-Blind feature frame contains forbidden columns: {sorted(bad)}")


def soft_label_gaussian(n: int, center: float | None, sigma_samples: float, mask: bool) -> np.ndarray:
    """1D soft label; zeros if mask=False or center invalid."""
    y = np.zeros(n, dtype=np.float32)
    if not mask or center is None or not np.isfinite(center):
        return y
    c = float(center)
    if c < 0 or c >= n:
        return y
    x = np.arange(n, dtype=np.float32)
    sig = max(float(sigma_samples), 1.0)
    y = np.exp(-0.5 * ((x - c) / sig) ** 2).astype(np.float32)
    return y


def build_psn_targets(
    n: int,
    p_sample: float | None,
    s_sample: float | None,
    *,
    has_p: bool,
    has_s: bool,
    sigma_s: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target[3,n] P,S,N ; loss_mask[3] for P,S,N)."""
    p = soft_label_gaussian(n, p_sample, sigma_s, has_p)
    s = soft_label_gaussian(n, s_sample, sigma_s, has_s)
    # noise channel: 1 - max(P,S) on supervised channels only; for unsupervised S leave free
    mx = np.maximum(p, s)
    noise = np.clip(1.0 - mx, 0.0, 1.0).astype(np.float32)
    target = np.stack([p, s, noise], axis=0)
    mask = np.array([1.0 if has_p else 0.0, 1.0 if has_s else 0.0, 1.0 if (has_p or has_s) else 1.0], dtype=np.float32)
    # If event trace with P but no S: do NOT supervise S or treat as noise in S band — mask S=0
    # Noise channel still supervised from P peak only region via 1-P when has_p and not has_s:
    if has_p and not has_s:
        # recompute noise from P only so unlabeled S region is not forced to noise=1 everywhere incorrectly
        noise = np.clip(1.0 - p, 0.0, 1.0).astype(np.float32)
        target[2] = noise
        mask[1] = 0.0
    return target, mask
