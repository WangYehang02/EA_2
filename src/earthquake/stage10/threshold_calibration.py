"""Pre-registered S-pick threshold selection. Calibration never peeks at evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from earthquake.stage10.dkpn_picks import extract_picks, s_f1_at_tolerance

THRESH_GRID = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5
)
OFFICIAL_HEIGHT = 0.2
PICKS_PER_TRACE_MAX = 3.0
PROTOCOL_NAME = "max_f1_0p5_on_calibration_picks_cap_3_tie_higher_thr"


def forced_choice_preds(s_probs: np.ndarray, thr: float) -> np.ndarray:
    preds = []
    for s in s_probs:
        pk, _, amp, _ = extract_picks(s, thr=float(thr))
        if len(pk) == 0:
            preds.append(np.nan)
        else:
            preds.append(float(pk[int(np.argmax(amp))]))
    return np.asarray(preds, dtype=float)


def window_prob_stats(s_probs: np.ndarray, p_probs: np.ndarray, true_s: np.ndarray, true_p: np.ndarray, vis_s: np.ndarray, vis_p: np.ndarray, local: int = 50) -> dict[str, float]:
    ts, tp, loc_s, loc_p = [], [], [], []
    oracle = []
    for s, p, t_s, t_p, vs, vp in zip(s_probs, p_probs, true_s, true_p, vis_s, vis_p):
        if bool(vs) and np.isfinite(t_s) and 0 <= t_s < len(s):
            i = int(round(float(t_s)))
            ts.append(float(s[i]))
            lo, hi = max(0, i - local), min(len(s), i + local + 1)
            loc_s.append(float(np.max(s[lo:hi])))
            oracle.append(abs(int(np.argmax(s)) - float(t_s)) <= local)
        if bool(vp) and np.isfinite(t_p) and 0 <= t_p < len(p):
            j = int(round(float(t_p)))
            tp.append(float(p[j]))
            lo, hi = max(0, j - local), min(len(p), j + local + 1)
            loc_p.append(float(np.max(p[lo:hi])))
    return {
        "true_s_probability": float(np.mean(ts) if ts else 0.0),
        "local_s_peak_probability": float(np.mean(loc_s) if loc_s else 0.0),
        "true_p_probability": float(np.mean(tp) if tp else 0.0),
        "local_p_peak_probability": float(np.mean(loc_p) if loc_p else 0.0),
        "oracle_argmax_accuracy@0.5s": float(np.mean(oracle) if oracle else 0.0),
        "n_vis_s_stats": int(len(ts)),
        "n_vis_p_stats": int(len(tp)),
    }


def pick_metrics_at_thr(s_probs: np.ndarray, true_s: np.ndarray, vis: np.ndarray, thr: float) -> dict[str, float]:
    preds = []
    n_peaks = []
    for s, t, v in zip(s_probs, true_s, vis):
        pk, _, amp, _ = extract_picks(s, thr=float(thr))
        n_peaks.append(len(pk))
        if len(pk) == 0:
            preds.append(np.nan)
        else:
            preds.append(float(pk[int(np.argmax(amp))]))
    pred = np.asarray(preds, dtype=float)
    m01 = s_f1_at_tolerance(pred, true_s, vis, tol_samples=10)
    m05 = s_f1_at_tolerance(pred, true_s, vis, tol_samples=50)
    return {
        "precision@0.1s": m01["precision"],
        "recall@0.1s": m01["recall"],
        "f1@0.1s": m01["f1"],
        "precision@0.5s": m05["precision"],
        "recall@0.5s": m05["recall"],
        "f1@0.5s": m05["f1"],
        "picks_per_trace": float(np.mean(n_peaks) if n_peaks else 0.0),
        "frac_no_s_peak": float(np.mean([n == 0 for n in n_peaks]) if n_peaks else 1.0),
        "n_vis": int(np.sum(vis)),
        "thr": float(thr),
    }


def select_threshold_on_calibration(s_probs: np.ndarray, true_s: np.ndarray, vis: np.ndarray) -> dict[str, Any]:
    """Use ONLY calibration arrays. Evaluation must call pick_metrics_at_thr with the returned thr."""
    rows = []
    for thr in THRESH_GRID:
        m = pick_metrics_at_thr(s_probs, true_s, vis, thr)
        m["eligible"] = m["picks_per_trace"] <= PICKS_PER_TRACE_MAX
        rows.append(m)
    elig = [r for r in rows if r["eligible"]]
    pool = elig if elig else rows
    best = max(pool, key=lambda r: (r["f1@0.5s"], r["thr"]))
    return {
        "protocol": PROTOCOL_NAME,
        "selected_thr": best["thr"],
        "selected_from_eligible_picks_cap": bool(elig),
        "calibration_metrics_at_selected": best,
        "grid": rows,
        "official_height": OFFICIAL_HEIGHT,
        "picks_per_trace_max": PICKS_PER_TRACE_MAX,
    }


def event_bootstrap_f1(
    pred: np.ndarray,
    true_s: np.ndarray,
    vis: np.ndarray,
    event_ids: np.ndarray,
    *,
    tol_samples: int = 50,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ev = np.asarray(event_ids)
    uniq = np.unique(ev)
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(ev, draw)
        m = s_f1_at_tolerance(pred[mask], true_s[mask], vis[mask], tol_samples=tol_samples)
        stats.append(m["f1"])
    a = np.asarray(stats, dtype=float)
    return {
        "f1_mean": float(a.mean()),
        "f1_lo": float(np.percentile(a, 2.5)),
        "f1_hi": float(np.percentile(a, 97.5)),
        "n_boot": n_boot,
        "n_events": int(len(uniq)),
    }
