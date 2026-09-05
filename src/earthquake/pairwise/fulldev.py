"""Frozen scalar-pairwise full-dev helpers (no waveform, no confirm)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from earthquake.pairwise.model import PairwiseScorer

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

RESID_CONTROL = {"lambda": 1.0, "sigma_s": 0.5}
FIXED_SCORE = {"lw": 0.5, "lh": 2.0, "lp": 0.0}
TAU = 0.50


def normalize_source(src: Any) -> str:
    s = str(src).strip().lower()
    if s in {"stead", "ida", "stead+ida"}:
        return s
    if "stead" in s and "ida" in s:
        return "stead+ida"
    if "ida" in s:
        return "ida"
    if "stead" in s:
        return "stead"
    return s


def scalar_vec_from_values(
    *,
    prob: float,
    stead_prob: float,
    ida_prob: float,
    fixed_score: float,
    resid_s: float,
    resid_sp: float,
    delta_sp: float,
    source: str,
) -> np.ndarray:
    src = normalize_source(source)
    return np.asarray(
        [
            float(prob),
            float(stead_prob) if np.isfinite(stead_prob) else 0.0,
            float(ida_prob) if np.isfinite(ida_prob) else 0.0,
            float(fixed_score),
            float(resid_s) if np.isfinite(resid_s) else 0.0,
            float(resid_sp) if np.isfinite(resid_sp) else 0.0,
            float(delta_sp) if np.isfinite(delta_sp) else 0.0,
            1.0 if src == "stead" else 0.0,
            1.0 if src == "ida" else 0.0,
            1.0 if src == "stead+ida" else 0.0,
        ],
        dtype=np.float32,
    )


def build_phaseb_pairs(enriched: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Build c1/c2 table from frozen fixed_score ranking (vectorized).

    c1 = argmax fixed_score; tie -> smaller candidate_index
    c2 = second by same rule
    Labels (true S) used only for metrics / Q1/Q2 — never for ranking.
    """
    need_cols = [
        "trace_name",
        "event_id",
        "candidate_index",
        "candidate_sample",
        "fixed_score",
        "cand_prob",
        "source",
        "resid_s",
        "resid_sp",
        "delta_sp",
        "stead_probability",
        "ida_probability",
        "sampling_rate_hz",
        "s_arrival_sample",
    ]
    missing = [c for c in need_cols if c not in enriched.columns]
    if missing:
        raise RuntimeError(f"enriched table missing columns: {missing}")

    e = enriched[need_cols].copy()
    e["trace_name"] = e["trace_name"].astype(str)
    e["event_id"] = e["event_id"].astype(str)
    e["source"] = e["source"].map(normalize_source)
    e = e.sort_values(
        ["trace_name", "fixed_score", "candidate_index"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    e["_rank"] = e.groupby("trace_name", sort=False).cumcount()
    c1 = e[e["_rank"] == 0].set_index("trace_name")
    c2 = e[e["_rank"] == 1].set_index("trace_name")
    n_cand = e.groupby("trace_name", sort=False).size().rename("n_candidates")

    man = manifest.copy()
    man["trace_name"] = man["trace_name"].astype(str)
    # keep manifest order (frozen phaseB)
    order = man["trace_name"].tolist()
    c1 = c1.reindex(order)
    c2 = c2.reindex(order)
    n_cand = n_cand.reindex(order)

    true = man.set_index("trace_name").reindex(order)["s_arrival_sample"].to_numpy(float)
    sr = c1["sampling_rate_hz"].to_numpy(float)
    # fallback sr from manifest if needed
    if not np.isfinite(sr).all() and "sampling_rate_hz" in man.columns:
        sr_m = man.set_index("trace_name").reindex(order)["sampling_rate_hz"].to_numpy(float)
        sr = np.where(np.isfinite(sr), sr, sr_m)

    ae1 = np.abs(c1["candidate_sample"].to_numpy(float) - true) / sr
    ae2 = np.abs(c2["candidate_sample"].to_numpy(float) - true) / sr
    c1_ok = ae1 <= 0.5
    c2_ok = np.isfinite(ae2) & (ae2 <= 0.5)
    n = n_cand.to_numpy(float)
    ge2 = np.isfinite(n) & (n >= 2)
    c2_ok = c2_ok & ge2

    def col(frame: pd.DataFrame, name: str, default=np.nan):
        if name not in frame.columns:
            return np.full(len(order), default)
        return frame[name].to_numpy()

    pairs = pd.DataFrame(
        {
            "trace_name": order,
            "event_id": c1["event_id"].astype(str).to_numpy(),
            "n_candidates": n_cand.to_numpy(),
            "sampling_rate_hz": sr,
            "true_s_sample": true,
            "c1_sample": col(c1, "candidate_sample"),
            "c1_index": col(c1, "candidate_index", -1),
            "c1_fixed_score": col(c1, "fixed_score"),
            "c1_prob": col(c1, "cand_prob", 0.0),
            "c1_source": c1["source"].fillna("").astype(str).to_numpy(),
            "c1_resid_s": col(c1, "resid_s"),
            "c1_resid_sp": col(c1, "resid_sp"),
            "c1_stead_prob": np.nan_to_num(col(c1, "stead_probability", 0.0), nan=0.0),
            "c1_ida_prob": np.nan_to_num(col(c1, "ida_probability", 0.0), nan=0.0),
            "c1_delta_sp": col(c1, "delta_sp"),
            "c2_sample": col(c2, "candidate_sample"),
            "c2_index": col(c2, "candidate_index", -1),
            "c2_fixed_score": col(c2, "fixed_score"),
            "c2_prob": col(c2, "cand_prob"),
            "c2_source": c2["source"].fillna("").astype(str).to_numpy() if "source" in c2.columns else "",
            "c2_resid_s": col(c2, "resid_s"),
            "c2_resid_sp": col(c2, "resid_sp"),
            "c2_stead_prob": np.nan_to_num(col(c2, "stead_probability", 0.0), nan=0.0),
            "c2_ida_prob": np.nan_to_num(col(c2, "ida_probability", 0.0), nan=0.0),
            "c2_delta_sp": col(c2, "delta_sp"),
        }
    )
    pairs["n_candidates"] = pairs["n_candidates"].fillna(0).astype(int)
    pairs["margin_fixed"] = pairs["c1_fixed_score"] - pairs["c2_fixed_score"]
    pairs.loc[pairs["n_candidates"] < 2, "margin_fixed"] = 0.0
    pairs["c1_ok"] = c1_ok & np.isfinite(pairs["c1_sample"].to_numpy(float))
    pairs["c2_ok"] = c2_ok
    pairs["ae_c1"] = ae1
    pairs["ae_c2"] = ae2
    # attach snr if present
    if "snr_db" in manifest.columns:
        pairs["snr_db"] = man.set_index("trace_name").reindex(order)["snr_db"].to_numpy(float)
    return pairs


def resid_control_pred(pairs: pd.DataFrame, *, lam: float = 1.0, sigma: float = 0.5) -> np.ndarray:
    out = np.empty(len(pairs), dtype=np.float64)
    c1 = pairs["c1_sample"].to_numpy(float)
    c2 = pairs["c2_sample"].to_numpy(float)
    n = pairs["n_candidates"].to_numpy(int)
    fs1 = pairs["c1_fixed_score"].to_numpy(float)
    fs2 = pairs["c2_fixed_score"].to_numpy(float)
    r1 = pairs["c1_resid_s"].to_numpy(float)
    r2 = pairs["c2_resid_s"].to_numpy(float)
    r1 = np.where(np.isfinite(r1), r1, 99.0)
    r2 = np.where(np.isfinite(r2), r2, 99.0)
    s1 = fs1 + lam * np.exp(-0.5 * (r1 / sigma) ** 2)
    s2 = fs2 + lam * np.exp(-0.5 * (r2 / sigma) ** 2)
    switch = (n >= 2) & np.isfinite(c2) & (s2 > s1)
    out[:] = c1
    out[switch] = c2[switch]
    return out


def load_scalar_model(ckpt_path, device: torch.device) -> PairwiseScorer:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_scalar = int(blob.get("n_scalar", len(SCALAR_FEATURE_NAMES)))
    model = PairwiseScorer(n_scalar, use_waveform=False, use_scalar=True)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_p_switch(model: PairwiseScorer, pairs: pd.DataFrame, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Return p(c2>c1) for each row; NaN when n_candidates < 2."""
    n = len(pairs)
    out = np.full(n, np.nan, dtype=np.float64)
    ge2 = pairs["n_candidates"].to_numpy(int) >= 2
    idx = np.where(ge2)[0]
    if idx.size == 0:
        return out

    def pack(rows: pd.DataFrame, which: str) -> np.ndarray:
        src = rows[f"{which}_source"].astype(str).tolist()
        mat = np.stack(
            [
                scalar_vec_from_values(
                    prob=float(r[f"{which}_prob"]),
                    stead_prob=float(r[f"{which}_stead_prob"]),
                    ida_prob=float(r[f"{which}_ida_prob"]),
                    fixed_score=float(r[f"{which}_fixed_score"]),
                    resid_s=float(r[f"{which}_resid_s"]),
                    resid_sp=float(r[f"{which}_resid_sp"]),
                    delta_sp=float(r[f"{which}_delta_sp"]),
                    source=s,
                )
                for r, s in zip(rows.itertuples(index=False), src)
            ],
            axis=0,
        )
        # itertuples with rename - safer use .iloc
        return mat

    # faster vectorized feature build
    def pack_fast(sub: pd.DataFrame, which: str) -> np.ndarray:
        src = sub[f"{which}_source"].map(normalize_source).to_numpy()
        prob = sub[f"{which}_prob"].to_numpy(float)
        st = sub[f"{which}_stead_prob"].to_numpy(float)
        ida = sub[f"{which}_ida_prob"].to_numpy(float)
        fs = sub[f"{which}_fixed_score"].to_numpy(float)
        rs = sub[f"{which}_resid_s"].to_numpy(float)
        rsp = sub[f"{which}_resid_sp"].to_numpy(float)
        dsp = sub[f"{which}_delta_sp"].to_numpy(float)
        st = np.where(np.isfinite(st), st, 0.0)
        ida = np.where(np.isfinite(ida), ida, 0.0)
        rs = np.where(np.isfinite(rs), rs, 0.0)
        rsp = np.where(np.isfinite(rsp), rsp, 0.0)
        dsp = np.where(np.isfinite(dsp), dsp, 0.0)
        mat = np.stack(
            [
                prob,
                st,
                ida,
                fs,
                rs,
                rsp,
                dsp,
                (src == "stead").astype(np.float32),
                (src == "ida").astype(np.float32),
                (src == "stead+ida").astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        return mat

    sub_all = pairs.iloc[idx]
    for start in range(0, len(sub_all), batch_size):
        sl = sub_all.iloc[start : start + batch_size]
        s1 = torch.from_numpy(pack_fast(sl, "c1")).to(device)
        s2 = torch.from_numpy(pack_fast(sl, "c2")).to(device)
        p = model(None, None, s1, None, None, s2).detach().cpu().numpy()
        out[idx[start : start + batch_size]] = p
    return out


def apply_switch(pairs: pd.DataFrame, p_switch: np.ndarray, tau: float = TAU) -> tuple[np.ndarray, np.ndarray]:
    pred = pairs["c1_sample"].to_numpy(float).copy()
    ge2 = pairs["n_candidates"].to_numpy(int) >= 2
    switch = ge2 & np.isfinite(p_switch) & (p_switch > tau)
    pred[switch] = pairs["c2_sample"].to_numpy(float)[switch]
    return pred, switch
