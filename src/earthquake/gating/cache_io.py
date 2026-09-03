"""Shared helpers for Stage-3 gate datasets / oracle / eval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from earthquake.fusion.candidate_rescorer import pick_argmax_probability, rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.history.residual_prior import (
    ResidualPathStats,
    expected_samples_catalog,
    predict_with_residual_prior,
)


def file_hash(path: Path, nbytes: int = 1_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(nbytes)
            if not chunk:
                break
            h.update(chunk)
            if nbytes and f.tell() > 8 * nbytes:
                break
    return h.hexdigest()[:16]


def build_cand_index(cands: pd.DataFrame) -> dict[tuple[str, str], list[PeakCandidate]]:
    idx: dict[tuple[str, str], list[PeakCandidate]] = {}
    for (trace, phase), sub in cands.groupby(["trace_name", "phase"], sort=False):
        lst = []
        for r in sub.itertuples(index=False):
            lst.append(
                PeakCandidate(
                    sample_index=int(r.sample_index),
                    absolute_utc=None,
                    peak_probability=float(r.peak_probability),
                    prominence=float(r.prominence) if pd.notna(r.prominence) else 0.0,
                    peak_width=float(r.peak_width) if pd.notna(r.peak_width) else float("nan"),
                    local_entropy=float(r.local_entropy) if pd.notna(r.local_entropy) else 0.0,
                    rank=int(r.rank),
                    fallback_peak=bool(r.fallback_peak),
                    phase=str(phase),
                )
            )
        lst = sorted(lst, key=lambda c: c.rank)
        idx[(str(trace), str(phase))] = lst
    return idx


def residual_stats_from_row(r: pd.Series) -> ResidualPathStats:
    return ResidualPathStats(
        history_count=int(r.get("history_count", 0) or 0),
        residual_p_median=float(r.get("residual_p_median", np.nan)),
        residual_p_mad=float(r.get("residual_p_mad", np.nan)),
        residual_s_median=float(r.get("residual_s_median", np.nan)),
        residual_s_mad=float(r.get("residual_s_mad", np.nan)),
        residual_sp_median=float(r.get("residual_sp_median", np.nan)),
        residual_sp_mad=float(r.get("residual_sp_mad", np.nan)),
        history_available=bool(r.get("history_available", False)),
        matched_key=r.get("matched_key"),
        fallback_level=int(r.get("fallback_level", -1)) if pd.notna(r.get("fallback_level", np.nan)) else -1,
        tau_p_median=float(r.get("tau_p_median", np.nan)),
        tau_s_median=float(r.get("tau_s_median", np.nan)),
        delta_sp_median=float(r.get("delta_sp_median", np.nan)),
        tau_p_mad=float(r.get("tau_p_mad", np.nan)),
        tau_s_mad=float(r.get("tau_s_mad", np.nan)),
        delta_sp_mad=float(r.get("delta_sp_mad", np.nan)),
    )


def attach_expected_s(
    row: pd.Series,
    *,
    global_res: dict[str, float],
    shrink_k: float,
    min_history: int,
    mad_disable_s: float,
    min_sigma_s: float,
    max_sigma_s: float,
) -> dict[str, float]:
    base = {
        "base_tau_p": float(row.get("base_tau_p", np.nan)),
        "base_tau_s": float(row.get("base_tau_s", np.nan)),
        "base_delta_sp": float(row.get("base_delta_sp", np.nan)),
    }
    path = residual_stats_from_row(row)
    pred = predict_with_residual_prior(
        base,
        path,
        global_res,
        shrinkage_k=shrink_k,
        min_history=min_history,
        mad_disable_s=mad_disable_s,
        min_sigma_s=min_sigma_s,
        max_sigma_s=max_sigma_s,
    )
    _, exp_s = expected_samples_catalog(row, pred)
    sr = float(row["sampling_rate_hz"])
    hist_ok = bool(pred.history_available or pred.used_path_residual)
    return {
        "expected_s_sample": float(exp_s) if np.isfinite(exp_s) else float("nan"),
        "history_sigma_samples": float(pred.sigma_s_s * sr),
        "history_sigma_s": float(pred.sigma_s_s),
        "pred_delta_sp": float(pred.pred_delta_sp),
        "used_path_residual": float(pred.used_path_residual),
        "gate_history_available": float(hist_ok),
        "shrinkage_weight_pred": float(pred.weight),
    }


def fixed_rescore_s(
    s_cands: list[PeakCandidate],
    *,
    expected_s: float,
    sigma_samples: float,
    history_available: bool,
    lw: float,
    lh: float,
    lp: float,
) -> float:
    best, _ = rescore_phase_candidates(
        s_cands,
        expected_sample=expected_s,
        sigma_samples=sigma_samples,
        lambda_wave=lw,
        lambda_history=lh,
        lambda_prominence=lp,
        history_available=history_available,
    )
    if best is None:
        return float("nan")
    return float(best.sample_index)


def phasenet_s_pick(s_cands: list[PeakCandidate], fallback: float) -> float:
    best = pick_argmax_probability(s_cands)
    if best is None:
        return float(fallback) if np.isfinite(fallback) else float("nan")
    return float(best.sample_index)


def sample_traces_by_event(
    df: pd.DataFrame,
    *,
    n_traces: int,
    max_per_event: int,
    seed: int,
    prefer_hard: bool = False,
) -> pd.DataFrame:
    """Event-disjoint sampling with per-event cap."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    if prefer_hard and "hard_score" in work.columns:
        work = work.sort_values("hard_score", ascending=False)
    else:
        work = work.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    selected = []
    per_event: dict[str, int] = {}
    for _, row in work.iterrows():
        eid = str(row.event_id)
        if per_event.get(eid, 0) >= max_per_event:
            continue
        selected.append(row)
        per_event[eid] = per_event.get(eid, 0) + 1
        if len(selected) >= n_traces:
            break
    out = pd.DataFrame(selected)
    if len(out) > n_traces:
        out = out.iloc[:n_traces]
    # shuffle final order
    if len(out):
        out = out.sample(frac=1.0, random_state=int(seed) + 1).reset_index(drop=True)
    return out


def save_gate_zarr(path: Path, records: list[dict[str, Any]], feature_columns: list[str], max_k: int) -> None:
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(records)
    f = len(feature_columns)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["n"] = n
    root.attrs["max_k"] = max_k
    root.attrs["feature_columns"] = feature_columns
    root.attrs["schema_note"] = "Stage3 gate dataset; labels stored for supervision/eval only"

    x = root.create_dataset("x", shape=(n, f), dtype="f4")
    miss = root.create_dataset("missing", shape=(n, f), dtype="f4")
    prob = root.create_dataset("cand_prob", shape=(n, max_k), dtype="f4")
    prom = root.create_dataset("cand_prominence", shape=(n, max_k), dtype="f4")
    width = root.create_dataset("cand_width", shape=(n, max_k), dtype="f4")
    samp = root.create_dataset("cand_sample", shape=(n, max_k), dtype="f4")
    mask = root.create_dataset("cand_mask", shape=(n, max_k), dtype="bool")
    expected = root.create_dataset("expected_s_sample", shape=(n,), dtype="f4")
    sigma = root.create_dataset("history_sigma_samples", shape=(n,), dtype="f4")
    hist_av = root.create_dataset("history_available", shape=(n,), dtype="bool")
    true_s = root.create_dataset("true_s_sample", shape=(n,), dtype="f4")
    true_p = root.create_dataset("true_p_sample", shape=(n,), dtype="f4")
    pred_p = root.create_dataset("pred_p_sample", shape=(n,), dtype="f4")
    pred_s = root.create_dataset("pred_s_phasenet", shape=(n,), dtype="f4")
    hist_count = root.create_dataset("history_count", shape=(n,), dtype="f4")
    hist_mad = root.create_dataset("history_mad", shape=(n,), dtype="f4")
    fallback = root.create_dataset("fallback_level", shape=(n,), dtype="i4")

    meta_rows = []
    for i, r in enumerate(records):
        x[i] = r["x"]
        miss[i] = r["missing"]
        k = min(len(r["s_cands"]), max_k)
        for j in range(max_k):
            if j < k:
                c = r["s_cands"][j]
                prob[i, j] = float(c["peak_probability"])
                prom[i, j] = float(c.get("prominence", c["peak_probability"]))
                w = float(c.get("peak_width", np.nan))
                width[i, j] = w if np.isfinite(w) else 0.0
                samp[i, j] = float(c["sample_index"])
                mask[i, j] = True
            else:
                mask[i, j] = False
        expected[i] = float(r["expected_s_sample"]) if np.isfinite(r["expected_s_sample"]) else np.nan
        sigma[i] = float(r["history_sigma_samples"])
        hist_av[i] = bool(r["history_available"])
        true_s[i] = float(r["true_s_sample"]) if r["true_s_sample"] is not None and np.isfinite(r["true_s_sample"]) else np.nan
        true_p[i] = float(r.get("true_p_sample", np.nan)) if np.isfinite(float(r.get("true_p_sample", np.nan))) else np.nan
        pred_p[i] = float(r.get("pred_p_sample", np.nan))
        pred_s[i] = float(r.get("pred_s_phasenet", np.nan))
        hist_count[i] = float(r.get("history_count", 0))
        hist_mad[i] = float(r.get("history_mad", np.nan))
        fallback[i] = int(r.get("fallback_level", -1))
        meta_rows.append(
            {
                "trace_name": r["trace_name"],
                "event_id": r["event_id"],
                "sampling_rate_hz": r["sampling_rate"],
                "abcd_class": r.get("abcd_class", ""),
                "split": r.get("split", ""),
            }
        )
    pd.DataFrame(meta_rows).to_parquet(path / "meta.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"n": n, "max_k": max_k, "feature_columns": feature_columns}, indent=2))


def load_gate_records(path: Path, *, min_history: float = 5.0, mad_threshold: float = 1.0) -> list[dict[str, Any]]:
    path = Path(path)
    root = zarr.open_group(str(path), mode="r")
    meta = pd.read_parquet(path / "meta.parquet")
    cols = list(root.attrs["feature_columns"])
    max_k = int(root.attrs["max_k"])
    n = int(root.attrs["n"])
    X = np.asarray(root["x"][:], dtype=np.float32)
    Miss = np.asarray(root["missing"][:], dtype=np.float32)
    Prob = np.asarray(root["cand_prob"][:], dtype=np.float32)
    Prom = np.asarray(root["cand_prominence"][:], dtype=np.float32)
    Width = np.asarray(root["cand_width"][:], dtype=np.float32)
    Samp = np.asarray(root["cand_sample"][:], dtype=np.float32)
    Mask = np.asarray(root["cand_mask"][:], dtype=bool)
    Exp = np.asarray(root["expected_s_sample"][:], dtype=np.float32)
    Sig = np.asarray(root["history_sigma_samples"][:], dtype=np.float32)
    HistA = np.asarray(root["history_available"][:], dtype=bool)
    TrueS = np.asarray(root["true_s_sample"][:], dtype=np.float32)
    TrueP = np.asarray(root["true_p_sample"][:], dtype=np.float32)
    PredP = np.asarray(root["pred_p_sample"][:], dtype=np.float32)
    PredS = np.asarray(root["pred_s_phasenet"][:], dtype=np.float32)
    HCount = np.asarray(root["history_count"][:], dtype=np.float32)
    HMad = np.asarray(root["history_mad"][:], dtype=np.float32)
    Fb = np.asarray(root["fallback_level"][:], dtype=np.int32)
    records = []
    for i in range(n):
        s_cands = []
        for j in range(max_k):
            if not bool(Mask[i, j]):
                continue
            s_cands.append(
                {
                    "peak_probability": float(Prob[i, j]),
                    "prominence": float(Prom[i, j]),
                    "peak_width": float(Width[i, j]),
                    "sample_index": float(Samp[i, j]),
                    "rank": j,
                    "fallback_peak": False,
                    "local_entropy": np.nan,
                }
            )
        records.append(
            {
                "x": X[i],
                "missing": Miss[i],
                "s_cands": s_cands,
                "expected_s_sample": float(Exp[i]),
                "history_sigma_samples": float(Sig[i]),
                "history_available": bool(HistA[i]),
                "true_s_sample": float(TrueS[i]),
                "true_p_sample": float(TrueP[i]),
                "pred_p_sample": float(PredP[i]),
                "pred_s_phasenet": float(PredS[i]),
                "history_count": float(HCount[i]),
                "history_mad": float(HMad[i]),
                "fallback_level": int(Fb[i]),
                "min_history": min_history,
                "mad_threshold": mad_threshold,
                "sampling_rate": float(meta.iloc[i]["sampling_rate_hz"]),
                "trace_name": str(meta.iloc[i]["trace_name"]),
                "event_id": str(meta.iloc[i]["event_id"]),
                "abcd_class": str(meta.iloc[i].get("abcd_class", "")),
                "feature_columns": cols,
            }
        )
    return records


def abcd_class(pn_err: float, hist_err: float, tol: float = 0.5) -> str:
    pn_ok = np.isfinite(pn_err) and pn_err <= tol
    h_ok = np.isfinite(hist_err) and hist_err <= tol
    if pn_ok and h_ok:
        return "A"
    if pn_ok and not h_ok:
        return "B"
    if (not pn_ok) and h_ok:
        return "C"
    return "D"
