"""Paired event bootstrap and post-hoc non-inferiority (not the original v4 gate)."""

from __future__ import annotations

from typing import Any

import numpy as np

from earthquake.stage10.dkpn_picks import s_f1_at_tolerance

# From checkpoint_policy.MIN_DELTA_F1 and v3 full-dev stop_gate
# `A_vs_waveform_only_delta_f1_0.5_ge_0.01`. Not chosen from v4's -0.006.
F1_NONINFERIORITY_MARGIN = 0.01
# v3_continue.tag_oscillation uses 0.02 for oracle representation_regression.
ORACLE_DISASTER_POINT = 0.02
NOS_WORSEN_POINT = 0.02


def metrics_at_index(
    *,
    pred: np.ndarray,
    true_s: np.ndarray,
    vis: np.ndarray,
    n_peaks: np.ndarray,
    oracle: np.ndarray,
    loss: np.ndarray,
    idx: np.ndarray,
) -> dict[str, float]:
    m = s_f1_at_tolerance(pred[idx], true_s[idx], vis[idx], tol_samples=50)
    vis_i = vis[idx]
    ora = oracle[idx]
    ora_v = ora[vis_i & np.isfinite(ora)]
    return {
        "f1@0.5s": float(m["f1"]),
        "precision@0.5s": float(m["precision"]),
        "recall@0.5s": float(m["recall"]),
        "picks_per_trace": float(np.mean(n_peaks[idx])) if len(idx) else float("nan"),
        "frac_no_s_peak": float(np.mean(n_peaks[idx] == 0)) if len(idx) else float("nan"),
        "oracle@0.5s": float(np.mean(ora_v)) if ora_v.size else float("nan"),
        "val_loss": float(np.mean(loss[idx])) if len(idx) else float("nan"),
    }


def paired_event_bootstrap(
    event_ids: np.ndarray,
    pack_a: dict[str, np.ndarray],
    pack_b: dict[str, np.ndarray],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired: same event draw for A and B. delta = metric(B) - metric(A)."""
    ev = np.asarray(event_ids)
    uniq = np.unique(ev)
    idx = {e: np.where(ev == e)[0] for e in uniq}
    rng = np.random.default_rng(int(seed))
    keys = ("f1@0.5s", "oracle@0.5s", "precision@0.5s", "recall@0.5s", "picks_per_trace", "frac_no_s_peak", "val_loss")
    acc = {k: np.empty(n_boot, dtype=np.float64) for k in keys}
    n_ev = len(uniq)
    for i in range(n_boot):
        draw = rng.choice(uniq, size=n_ev, replace=True)
        ii = np.concatenate([idx[e] for e in draw])
        ma = metrics_at_index(idx=ii, **pack_a)
        mb = metrics_at_index(idx=ii, **pack_b)
        for k in keys:
            acc[k][i] = float(mb[k] - ma[k])
    out: dict[str, Any] = {"n_boot": int(n_boot), "n_events": int(n_ev), "seed": int(seed)}
    for k in keys:
        d = acc[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {
            "mean_delta": float(d.mean()),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        }
    return out


def adjudicate_e2_vs_e1(
    *,
    point: dict[str, float],
    boot: dict[str, Any],
    vs_init_e1: dict[str, Any],
    vs_init_e2: dict[str, Any],
    finite: bool,
    confirm_read: bool,
    margin: float = F1_NONINFERIORITY_MARGIN,
) -> tuple[bool, list[str], dict[str, bool]]:
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    d_f1 = float(point["f1@0.5s"])
    checks["point_f1_ge_neg_margin"] = d_f1 >= -margin
    checks["ci_lo_f1_gt_neg_margin"] = float(boot["f1@0.5s"]["ci95_lo"]) > -margin
    checks["oracle_not_disaster"] = float(point["oracle@0.5s"]) >= -ORACLE_DISASTER_POINT
    checks["nos_not_worse"] = float(point["frac_no_s_peak"]) <= NOS_WORSEN_POINT
    checks["e1_gt_init_f1"] = float(vs_init_e1["f1@0.5s"]["ci95_lo"]) > 0.0
    checks["e2_gt_init_f1"] = float(vs_init_e2["f1@0.5s"]["ci95_lo"]) > 0.0
    checks["finite_fp32"] = bool(finite)
    checks["confirm_unread"] = not bool(confirm_read)
    for k, ok in checks.items():
        if not ok:
            reasons.append(k)
    return (len(reasons) == 0), reasons, checks
