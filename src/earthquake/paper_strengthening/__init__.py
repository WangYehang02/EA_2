"""Paper strengthening v1: strong simple controls on shared c1/c2 (no new pickers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-8
WAVE_LW = 0.5

# Pre-registered C grid (order = tertiary tie-break)
SIGMA_S_GRID = (0.25, 0.5, 1.0, 2.0)
LAMBDA_HISTORY_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
C_CONFIGS: list[tuple[float, float]] = [
    (sig, lam) for sig in SIGMA_S_GRID for lam in LAMBDA_HISTORY_GRID
]
assert len(C_CONFIGS) == 20

D_C_VALUES = (0.01, 0.1, 1.0, 10.0)
D_TAU_VALUES = tuple(round(x, 1) for x in np.arange(0.1, 1.0, 0.1))

SCALAR_FEATURE_NAMES = [
    "prob",
    "stead_prob",
    "ida_prob",
    "fixed_score",
    "resid_s",
    "resid_sp",
    "delta_sp",
    "src_stead",
    "src_ida",
    "src_both",
]


def _finite(x: np.ndarray, fill: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, fill)


def base_tau_expected_score(
    prob: np.ndarray,
    resid_s: np.ndarray,
    *,
    sigma_s: float,
    lambda_history: float,
    wave_lw: float = WAVE_LW,
) -> np.ndarray:
    """Exact c1/c2 form of base_tau_as_expected with free (σ, λ_history).

    score = wave_lw * log(p+eps) + λ_history * log(exp(-0.5*(resid_s/σ)^2)+eps)

    Equivalent to fixed_candidate_scores with expected:=base_tau sample mapping,
    history_available:=True, history_sigma:=σ*sr, lh:=λ_history, lw:=wave_lw.
    Missing resid_s → treat as |resid|=99 (near-zero Gaussian term).
    """
    p = np.clip(_finite(prob, 0.0), 0.0, 1.0)
    r = _finite(resid_s, 99.0)
    wave = wave_lw * np.log(p + EPS)
    if abs(lambda_history) < 1e-15:
        return wave
    g = np.exp(-0.5 * (r / float(sigma_s)) ** 2)
    return wave + float(lambda_history) * np.log(g + EPS)


def pred_fixed(pairs: pd.DataFrame) -> np.ndarray:
    return pairs["c1_sample"].to_numpy(float).copy()


def pred_resid_control(pairs: pd.DataFrame, *, lam: float = 1.0, sigma: float = 0.5) -> np.ndarray:
    """Registered resid_s control on c1/c2 only (same as fulldev.resid_control_pred)."""
    out = pairs["c1_sample"].to_numpy(float).copy()
    n = pairs["n_candidates"].to_numpy(int)
    c2 = pairs["c2_sample"].to_numpy(float)
    fs1 = pairs["c1_fixed_score"].to_numpy(float)
    fs2 = pairs["c2_fixed_score"].to_numpy(float)
    r1 = _finite(pairs["c1_resid_s"].to_numpy(float), 99.0)
    r2 = _finite(pairs["c2_resid_s"].to_numpy(float), 99.0)
    s1 = fs1 + lam * np.exp(-0.5 * (r1 / sigma) ** 2)
    s2 = fs2 + lam * np.exp(-0.5 * (r2 / sigma) ** 2)
    switch = (n >= 2) & np.isfinite(c2) & (s2 > s1)
    out[switch] = c2[switch]
    return out


def pred_base_tau_c1c2(
    pairs: pd.DataFrame,
    *,
    sigma_s: float,
    lambda_history: float,
) -> np.ndarray:
    out = pairs["c1_sample"].to_numpy(float).copy()
    n = pairs["n_candidates"].to_numpy(int)
    c2 = pairs["c2_sample"].to_numpy(float)
    s1 = base_tau_expected_score(
        pairs["c1_prob"].to_numpy(float),
        pairs["c1_resid_s"].to_numpy(float),
        sigma_s=sigma_s,
        lambda_history=lambda_history,
    )
    s2 = base_tau_expected_score(
        pairs["c2_prob"].to_numpy(float),
        pairs["c2_resid_s"].to_numpy(float),
        sigma_s=sigma_s,
        lambda_history=lambda_history,
    )
    switch = (n >= 2) & np.isfinite(c2) & (s2 > s1)
    out[switch] = c2[switch]
    return out


def pack_scalar_matrix(pairs: pd.DataFrame, which: str) -> np.ndarray:
    from earthquake.pairwise.fulldev import normalize_source

    src = pairs[f"{which}_source"].map(normalize_source).to_numpy()
    prob = _finite(pairs[f"{which}_prob"].to_numpy(float), 0.0)
    st = _finite(pairs[f"{which}_stead_prob"].to_numpy(float), 0.0)
    ida = _finite(pairs[f"{which}_ida_prob"].to_numpy(float), 0.0)
    fs = _finite(pairs[f"{which}_fixed_score"].to_numpy(float), 0.0)
    rs = _finite(pairs[f"{which}_resid_s"].to_numpy(float), 0.0)
    rsp = _finite(pairs[f"{which}_resid_sp"].to_numpy(float), 0.0)
    dsp = _finite(pairs[f"{which}_delta_sp"].to_numpy(float), 0.0)
    return np.stack(
        [
            prob,
            st,
            ida,
            fs,
            rs,
            rsp,
            dsp,
            (src == "stead").astype(np.float64),
            (src == "ida").astype(np.float64),
            (src == "stead+ida").astype(np.float64),
        ],
        axis=1,
    )


def feature_diff_x2_minus_x1(pairs: pd.DataFrame) -> np.ndarray:
    return pack_scalar_matrix(pairs, "c2") - pack_scalar_matrix(pairs, "c1")


@dataclass
class LinearPairwiseModel:
    coef: np.ndarray  # shape (10,)
    mean: np.ndarray
    scale: np.ndarray
    C: float

    def p_switch(self, pairs: pd.DataFrame) -> np.ndarray:
        n = pairs["n_candidates"].to_numpy(int)
        out = np.full(len(pairs), np.nan, dtype=np.float64)
        ge2 = n >= 2
        if not ge2.any():
            return out
        d = feature_diff_x2_minus_x1(pairs.loc[ge2])
        d = (d - self.mean) / self.scale
        z = d @ self.coef
        out[ge2] = 1.0 / (1.0 + np.exp(-z))
        return out

    def predict(self, pairs: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p_switch(pairs)
        pred = pairs["c1_sample"].to_numpy(float).copy()
        ge2 = pairs["n_candidates"].to_numpy(int) >= 2
        switch = ge2 & np.isfinite(p) & (p > tau)
        pred[switch] = pairs["c2_sample"].to_numpy(float)[switch]
        return pred, switch


def fit_linear_pairwise(
    train_pairs: pd.DataFrame,
    *,
    C: float,
    random_state: int = 42,
) -> LinearPairwiseModel:
    """Train on decisive pairs only; no intercept; no sample weights; standardize on train."""
    from sklearn.linear_model import LogisticRegression

    dec = train_pairs[train_pairs["y_choose_c2"].isin([0, 1])].copy()
    if len(dec) < 10:
        raise RuntimeError(f"too few decisive pairs: {len(dec)}")
    X = feature_diff_x2_minus_x1(dec)
    y = dec["y_choose_c2"].to_numpy(int)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Xs = (X - mean) / scale
    clf = LogisticRegression(
        C=float(C),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    clf.fit(Xs, y)
    coef = clf.coef_.reshape(-1).astype(np.float64)
    return LinearPairwiseModel(coef=coef, mean=mean.astype(np.float64), scale=scale.astype(np.float64), C=float(C))


def metrics_from_pred(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict:
    from earthquake.metrics import match_picks

    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    return {
        "f1@0.5": float(m["f1@0.5s"]),
        "f1@0.1": float(m["f1@0.1s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_mae": float(m["detected_ae_mae"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "miss_rate": float(m["miss_rate"]),
        "frac_err_gt_1s": float(m.get("frac_err_gt_1s", np.nan)),
        "frac_err_gt_5s": float(m.get("frac_err_gt_5s", np.nan)),
        "frac_err_gt_10s": float(m.get("frac_err_gt_10s", np.nan)),
        "n": int(len(pred)),
    }


def switch_mechanism(pairs: pd.DataFrame, pred: np.ndarray) -> dict:
    """Mechanism metrics with explicit denominators."""
    n = pairs["n_candidates"].to_numpy(int)
    ge2 = n >= 2
    c1 = pairs["c1_sample"].to_numpy(float)
    c2 = pairs["c2_sample"].to_numpy(float)
    switched = ge2 & np.isfinite(pred) & np.isfinite(c2) & (np.abs(pred - c2) < 0.5) & ~(np.abs(pred - c1) < 0.5)
    # correctness vs 0.5s
    true = pairs["true_s_sample"].to_numpy(float)
    sr = pairs["sampling_rate_hz"].to_numpy(float)
    ae_c1 = np.abs(c1 - true) / sr
    ae_pred = np.abs(pred - true) / sr
    c1_ok = ae_c1 <= 0.5
    pred_ok = ae_pred <= 0.5
    # only among switched
    sw = switched
    fixes = int((sw & (~c1_ok) & pred_ok).sum())
    breaks = int((sw & c1_ok & (~pred_ok)).sum())
    unchanged_ok = int((sw & c1_ok & pred_ok).sum())
    unchanged_bad = int((sw & (~c1_ok) & (~pred_ok)).sum())
    n_switch = int(sw.sum())
    state_changing = fixes + breaks
    return {
        "n_traces": int(len(pairs)),
        "n_ge2": int(ge2.sum()),
        "n_switch": n_switch,
        "fixes": fixes,
        "breaks": breaks,
        "net_fixes": fixes - breaks,
        "unchanged_correct_after_switch": unchanged_ok,
        "unchanged_wrong_after_switch": unchanged_bad,
        "switch_precision_state_changing": (fixes / state_changing) if state_changing else float("nan"),
        "switch_precision_definition": "fixes/(fixes+breaks) among switches that change correctness state @0.5s",
        "Q1_count": int((ge2 & c1_ok & ~(np.abs(c2 - true) / sr <= 0.5)).sum()) if ge2.any() else 0,
        "Q2_count": int((ge2 & (~c1_ok) & (np.abs(c2 - true) / sr <= 0.5)).sum()) if ge2.any() else 0,
    }


def select_best_by_f1(rows: list[dict], *, order_keys: Iterable[str]) -> dict:
    """Primary F1@0.5, secondary F1@0.1, tertiary pre-registered order (smaller index better)."""

    def key(r):
        return (-float(r["f1@0.5"]), -float(r["f1@0.1"]), int(r.get("order_index", 10**9)))

    return sorted(rows, key=key)[0]


def paired_event_bootstrap_delta(
    event_ids: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    # precompute per-event index lists
    ev_to_idx = {}
    for i, e in enumerate(event_ids.astype(str)):
        ev_to_idx.setdefault(e, []).append(i)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(events, size=len(events), replace=True)
        idx = np.concatenate([ev_to_idx[e] for e in samp])
        ma = metrics_from_pred(pred_a[idx], true[idx], sr[idx])["f1@0.5"]
        mb = metrics_from_pred(pred_b[idx], true[idx], sr[idx])["f1@0.5"]
        deltas[b] = ma - mb
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_events": int(len(events)),
        "mean": float(deltas.mean()),
        "ci95": [float(lo), float(hi)],
    }
