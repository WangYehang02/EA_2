from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def match_picks(
    pred_samples: np.ndarray,
    true_samples: np.ndarray,
    sampling_rate: float | np.ndarray,
    windows_s: Iterable[float] = (0.1, 0.5),
) -> dict[str, Any]:
    """One-prediction-per-trace matching.

    Definitions (single peak protocol):
    - TP@w: has label & has pred & |err| <= w
    - FP@w: has pred & not TP  (includes wrong-peak-beyond-tolerance and pred-without-label)
    - FN@w: has label & not TP (includes misses and wrong peaks; a wrong peak is FP and FN once each)
    A prediction outside tolerance is never counted as TP, and is not double-counted as two FPs.

    Timing metrics (naming):
    - ``detected_ae_*`` / historical aliases ``mae`` / ``e2e_*``:
      absolute error on labeled traces that also have a prediction
      (includes wrong peaks; **excludes NaN misses**).
    - Do **not** call this alone "complete end-to-end P95"; always report miss_rate separately.
    - Prefer ``audit_pick_errors`` for matched-only timing MAE / P95.
    """
    pred = np.asarray(pred_samples, dtype=np.float64)
    true = np.asarray(true_samples, dtype=np.float64)
    sr = np.asarray(sampling_rate, dtype=np.float64)
    if sr.ndim == 0:
        sr = np.full_like(pred, float(sr))

    has_true = np.isfinite(true)
    has_pred = np.isfinite(pred)
    err_s = np.full_like(pred, np.nan)
    both = has_true & has_pred
    err_s[both] = (pred[both] - true[both]) / sr[both]

    out: dict[str, Any] = {}
    for w in windows_s:
        matched = both & np.isfinite(err_s) & (np.abs(err_s) <= w)
        tp = int(matched.sum())
        fp = int((has_pred & ~matched).sum())
        fn = int((has_true & ~matched).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        out[f"precision@{w}s"] = precision
        out[f"recall@{w}s"] = recall
        out[f"f1@{w}s"] = f1
        out[f"tp@{w}s"] = tp
        out[f"fp@{w}s"] = fp
        out[f"fn@{w}s"] = fn

    labeled_pred = both & np.isfinite(err_s)
    abs_err = np.abs(err_s[labeled_pred])
    out["n_eval"] = int(has_true.sum())
    out["n_pred"] = int(has_pred.sum())
    out["n_miss"] = int((has_true & ~has_pred).sum())
    out["miss_rate"] = float(out["n_miss"] / max(out["n_eval"], 1))
    out["mae"] = float(np.mean(abs_err)) if abs_err.size else float("nan")
    out["median_ae"] = float(np.median(abs_err)) if abs_err.size else float("nan")
    out["p95_ae"] = float(np.percentile(abs_err, 95)) if abs_err.size else float("nan")
    # Preferred Stage-6 names (aliases keep old keys for frozen artifacts)
    out["detected_ae_mae"] = out["mae"]
    out["detected_ae_median"] = out["median_ae"]
    out["detected_ae_p95"] = out["p95_ae"]
    out["e2e_mae"] = out["mae"]
    out["e2e_median_ae"] = out["median_ae"]
    out["e2e_p95_ae"] = out["p95_ae"]
    out["p95_scope"] = "detected_non_missing_labeled_with_prediction"
    out["p95_naming_warning"] = (
        "detected_ae_p95 / legacy e2e_p95_ae excludes NaN misses; report miss_rate separately; "
        "do not call this alone complete end-to-end P95"
    )
    return out


def report_pick_timing_bundle(
    pred_samples: np.ndarray,
    true_samples: np.ndarray,
    sampling_rate: float | np.ndarray,
    *,
    windows_s: Iterable[float] = (0.1, 0.2, 0.5),
    match_window_s: float = 0.5,
) -> dict[str, Any]:
    """Stage-6 preferred metric bundle with explicit naming."""
    m = match_picks(pred_samples, true_samples, sampling_rate, windows_s=windows_s)
    a = audit_pick_errors(pred_samples, true_samples, sampling_rate, window_s=match_window_s)
    return {
        "f1@0.1": m.get("f1@0.1s"),
        "f1@0.2": m.get("f1@0.2s"),
        "f1@0.5": m.get("f1@0.5s"),
        "detected_ae_median": m["detected_ae_median"],
        "detected_ae_mae": m["detected_ae_mae"],
        "detected_ae_p95": m["detected_ae_p95"],
        "miss_rate": m["miss_rate"],
        "wrong_peak_rate": a["n_wrong_peak_beyond_tol"] / max(a["n_labeled"], 1),
        "no_pick_rate": a["n_missed_pick"] / max(a["n_labeled"], 1),
        "n_labeled": a["n_labeled"],
        "n_predicted": a["n_predicted"],
        "legacy_alias_e2e_p95_ae": m["e2e_p95_ae"],
        "p95_naming_warning": m["p95_naming_warning"],
    }


def audit_pick_errors(
    pred_samples: np.ndarray,
    true_samples: np.ndarray,
    sampling_rate: float | np.ndarray,
    *,
    window_s: float = 0.5,
    multi_pred_samples: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Detailed error audit separating detection vs timing.

    Categories (mutually exclusive for the primary single-peak protocol):
    - missed_pick: label present, prediction absent
    - false_pick: prediction present, label absent
    - matched_within_tol: |err| <= window
    - wrong_peak_beyond_tol: both present, |err| > window

    Timing:
    - e2e_* : absolute error on all labeled+predicted traces
    - matched_timing_* : absolute error only on matched_within_tol
    - wrong_peak_* : absolute error only on wrong_peak_beyond_tol
    - labeled_all_* : absolute error on all labeled traces (misses contribute NaN / excluded
      from mean unless imputed — here misses are excluded from means but counted)
    """
    pred = np.asarray(pred_samples, dtype=np.float64)
    true = np.asarray(true_samples, dtype=np.float64)
    sr = np.asarray(sampling_rate, dtype=np.float64)
    if sr.ndim == 0:
        sr = np.full_like(pred, float(sr))

    has_true = np.isfinite(true)
    has_pred = np.isfinite(pred)
    err_s = np.full(len(pred), np.nan, dtype=np.float64)
    both = has_true & has_pred
    err_s[both] = (pred[both] - true[both]) / sr[both]
    abs_err = np.abs(err_s)

    matched = both & (abs_err <= window_s)
    wrong = both & (abs_err > window_s)
    missed = has_true & (~has_pred)
    false = has_pred & (~has_true)

    def _stats(mask: np.ndarray) -> dict[str, float]:
        x = abs_err[mask & np.isfinite(abs_err)]
        if x.size == 0:
            return {"n": 0, "mae": float("nan"), "median_ae": float("nan"), "p95_ae": float("nan")}
        return {
            "n": int(x.size),
            "mae": float(np.mean(x)),
            "median_ae": float(np.median(x)),
            "p95_ae": float(np.percentile(x, 95)),
        }

    # Multi-peak protocol: success if ANY candidate is within tolerance of true
    multi = {"enabled": False}
    if multi_pred_samples is not None:
        multi["enabled"] = True
        hit = np.zeros(len(true), dtype=bool)
        best_abs = np.full(len(true), np.nan)
        for i in np.where(has_true)[0]:
            cands = multi_pred_samples[i] if i < len(multi_pred_samples) else None
            if cands is None:
                continue
            c = np.asarray(cands, dtype=np.float64)
            c = c[np.isfinite(c)]
            if c.size == 0:
                continue
            ae = np.abs((c - true[i]) / sr[i])
            best_abs[i] = float(np.min(ae))
            hit[i] = best_abs[i] <= window_s
        multi.update(
            {
                "n_labeled": int(has_true.sum()),
                "n_any_cand_within_tol": int(hit.sum()),
                "recall_any_cand": float(hit.sum() / max(int(has_true.sum()), 1)),
                "best_cand_mae": float(np.nanmean(best_abs[has_true])) if has_true.any() else float("nan"),
                "best_cand_median_ae": float(np.nanmedian(best_abs[has_true])) if has_true.any() else float("nan"),
            }
        )

    categories = np.full(len(pred), "other", dtype=object)
    categories[matched] = "matched_within_tol"
    categories[wrong] = "wrong_peak_beyond_tol"
    categories[missed] = "missed_pick"
    categories[false] = "false_pick"

    out = {
        "window_s": float(window_s),
        "n_labeled": int(has_true.sum()),
        "n_predicted": int(has_pred.sum()),
        "n_missed_pick": int(missed.sum()),
        "n_false_pick": int(false.sum()),
        "n_matched_within_tol": int(matched.sum()),
        "n_wrong_peak_beyond_tol": int(wrong.sum()),
        "e2e": _stats(both),
        "matched_timing": _stats(matched),
        "wrong_peak": _stats(wrong),
        "multi_peak_protocol": multi,
        "p95_e2e_scope": "all_labeled_with_prediction",
        "p95_matched_scope": f"matched_within_{window_s}s_only",
        "categories": categories,
        "abs_err_s": abs_err,
        "err_s": err_s,
    }
    # convenience flat keys
    out["e2e_mae"] = out["e2e"]["mae"]
    out["e2e_median_ae"] = out["e2e"]["median_ae"]
    out["e2e_p95_ae"] = out["e2e"]["p95_ae"]
    out["matched_timing_mae"] = out["matched_timing"]["mae"]
    out["matched_timing_median_ae"] = out["matched_timing"]["median_ae"]
    out["matched_timing_p95_ae"] = out["matched_timing"]["p95_ae"]
    return out


def noise_false_positive_rate(pred_detected: np.ndarray) -> float:
    pred = np.asarray(pred_detected, dtype=bool)
    if pred.size == 0:
        return float("nan")
    return float(pred.mean())


def metrics_from_pick_frame(df: pd.DataFrame, phase: str = "p", windows=(0.1, 0.5)) -> dict[str, Any]:
    pred_col = f"pred_{phase}_sample"
    true_col = f"true_{phase}_sample"
    return match_picks(
        df[pred_col].to_numpy(),
        df[true_col].to_numpy(),
        df["sampling_rate_hz"].to_numpy(),
        windows_s=windows,
    )


def event_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn,
    n_boot: int = 200,
    seed: int = 0,
    event_col: str = "event_id",
) -> dict[str, float]:
    """Bootstrap over events (not waveforms)."""
    rng = np.random.default_rng(seed)
    events = df[event_col].astype(str).unique()
    if len(events) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    vals = []
    for _ in range(n_boot):
        sample_events = rng.choice(events, size=len(events), replace=True)
        sub = df[df[event_col].astype(str).isin(sample_events)]
        vals.append(metric_fn(sub))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "low": float(np.nanpercentile(arr, 2.5)),
        "high": float(np.nanpercentile(arr, 97.5)),
    }


def groupby_history_metrics(df: pd.DataFrame, phase: str = "p") -> pd.DataFrame:
    bins = [
        ("unavailable", df["history_available"] == False),  # noqa: E712
        ("1-4", (df["history_count"] >= 1) & (df["history_count"] <= 4)),
        ("5-9", (df["history_count"] >= 5) & (df["history_count"] <= 9)),
        ("10-19", (df["history_count"] >= 10) & (df["history_count"] <= 19)),
        (">=20", df["history_count"] >= 20),
    ]
    rows = []
    for name, mask in bins:
        sub = df[mask]
        if len(sub) == 0:
            continue
        m = metrics_from_pick_frame(sub, phase=phase)
        m["group"] = name
        m["phase"] = phase
        m["n"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows)
