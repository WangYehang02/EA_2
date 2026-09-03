"""Unified pick metrics for Phase C.1 (explicit coverage / none / miss / P95 denom)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from earthquake.metrics import audit_pick_errors, match_picks


def comprehensive_pick_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    *,
    none_mask: np.ndarray | None = None,
    windows_s: Iterable[float] = (0.1, 0.2, 0.5),
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    sr = np.asarray(sr, dtype=np.float64)
    n = len(true)
    has_true = np.isfinite(true)
    has_pred = np.isfinite(pred)
    if none_mask is None:
        none_mask = ~has_pred
    else:
        none_mask = np.asarray(none_mask, dtype=bool)

    m = match_picks(pred, true, sr, windows_s=windows_s)
    a = audit_pick_errors(pred, true, sr, window_s=0.5)

    both = has_true & has_pred
    err = np.full(n, np.nan)
    err[both] = np.abs((pred[both] - true[both]) / sr[both])
    detected = err[np.isfinite(err)]
    denom = int(detected.size)

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    out: dict[str, Any] = {
        "n_traces": int(n),
        "n_labeled": int(has_true.sum()),
        "n_predicted": int(has_pred.sum()),
        "n_finite_prediction": int(has_pred.sum()),
        "prediction_coverage": float(has_pred.sum() / max(int(has_true.sum()), 1)),
        "predicted_pick_count": int(has_pred.sum()),
        "none_of_k_count": int(none_mask.sum()),
        "none_of_k_rate": float(none_mask.sum() / max(n, 1)),
        "miss_count": int(m["n_miss"]),
        "miss_rate": float(m["miss_rate"]),
        "wrong_peak_count": int(a["n_wrong_peak_beyond_tol"]),
        "wrong_peak_rate": float(a["n_wrong_peak_beyond_tol"] / max(a["n_labeled"], 1)),
        "matched_count_0.5": int(m.get("tp@0.5s", 0)),
        "detected_ae_n": denom,
        "detected_ae_median": float(np.median(detected)) if denom else float("nan"),
        "detected_ae_mae": float(np.mean(detected)) if denom else float("nan"),
        "detected_ae_p90": _pct(detected, 90),
        "detected_ae_p95": _pct(detected, 95),
        "detected_ae_p99": _pct(detected, 99),
        "detected_ae_p95_denominator": denom,
        "p95_scope": "finite_pred_and_label_only_excludes_none_miss",
        "p95_naming_warning": m["p95_naming_warning"],
    }
    for w in windows_s:
        out[f"precision@{w}"] = float(m[f"precision@{w}s"])
        out[f"recall@{w}"] = float(m[f"recall@{w}s"])
        out[f"f1@{w}"] = float(m[f"f1@{w}s"])
        out[f"tp@{w}"] = int(m[f"tp@{w}s"])
        out[f"fp@{w}"] = int(m[f"fp@{w}s"])
        out[f"fn@{w}"] = int(m[f"fn@{w}s"])
    return out


def correct_at(pred: np.ndarray, true: np.ndarray, sr: np.ndarray, tol_s: float) -> np.ndarray:
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    sr = np.asarray(sr, float)
    out = np.zeros(len(true), dtype=bool)
    both = np.isfinite(pred) & np.isfinite(true)
    out[both] = np.abs((pred[both] - true[both]) / sr[both]) <= tol_s
    return out
