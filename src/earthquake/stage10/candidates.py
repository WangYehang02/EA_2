"""Candidate union / dedup / oracle helpers for Stage 10D."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dedup_candidates(df: pd.DataFrame, sample_tol: int = 50) -> pd.DataFrame:
    """Merge near-duplicate peaks per trace; keep max prob and union of sources."""
    rows = []
    for tn, g in df.groupby("trace_name", sort=False):
        g = g.sort_values("sample").reset_index(drop=True)
        clusters: list[dict] = []
        for _, r in g.iterrows():
            s = float(r["sample"])
            placed = False
            for c in clusters:
                if abs(s - c["sample"]) <= sample_tol:
                    if float(r["prob"]) > c["prob"]:
                        c["sample"] = s
                        c["prob"] = float(r["prob"])
                        c["rank"] = int(r.get("rank", 0))
                    c["sources"].add(str(r["source"]))
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "trace_name": tn,
                        "sample": s,
                        "prob": float(r["prob"]),
                        "rank": int(r.get("rank", 0)),
                        "sources": {str(r["source"])},
                    }
                )
        for c in clusters:
            rows.append(
                {
                    "trace_name": c["trace_name"],
                    "sample": c["sample"],
                    "prob": c["prob"],
                    "rank": c["rank"],
                    "sources": ",".join(sorted(c["sources"])),
                    "n_sources": len(c["sources"]),
                }
            )
    return pd.DataFrame(rows)


def threshold_sweep_f1(pred_prob: np.ndarray, pred_sample: np.ndarray, true_sample: np.ndarray, sr: np.ndarray, thresholds: np.ndarray, window_s: float = 0.5) -> pd.DataFrame:
    """Apply probability threshold: below thr → no pick. Returns per-threshold F1/miss."""
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    rows = []
    for thr in thresholds:
        pred = pred_sample.copy()
        pred[pred_prob < thr] = np.nan
        m = comprehensive_pick_metrics(pred, true_sample, sr)
        rows.append({"threshold": float(thr), "f1@0.5": m["f1@0.5"], "f1@0.1": m["f1@0.1"], "miss_rate": m["miss_rate"], "detected_ae_p95": m["detected_ae_p95"]})
    return pd.DataFrame(rows)


def event_bootstrap_delta(
    event_ids: np.ndarray,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 20260815,
) -> dict:
    """Event-level bootstrap of mean(b)-mean(a). metric_* are per-trace 0/1 or scores aligned to event_ids."""
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids)
    # pre-group
    idx = {e: np.where(event_ids == e)[0] for e in events}
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draw = rng.choice(events, size=len(events), replace=True)
        ia = np.concatenate([idx[e] for e in draw])
        deltas[i] = float(metric_b[ia].mean() - metric_a[ia].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "mean_delta": float(deltas.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "ci95_lo_gt_0": bool(lo > 0),
    }


def oracle_hit_rate(gt_samples: np.ndarray, source_preds: dict[str, np.ndarray], tol_samples: float) -> float:
    """Fraction of traces where ≥1 source is within tol (NaN = miss for that source)."""
    n = len(gt_samples)
    hits = 0
    for i in range(n):
        g = gt_samples[i]
        if not np.isfinite(g):
            continue
        ok = False
        for pred in source_preds.values():
            p = pred[i]
            if np.isfinite(p) and abs(p - g) <= tol_samples:
                ok = True
                break
        hits += int(ok)
    return hits / max(n, 1)
