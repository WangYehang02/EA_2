"""Build confirm UNION pairs with Stage-6 locked fixed_rescore scoring.

Uses attach_expected_s + rescore_phase_candidates (same as run_stage6_final_confirm),
so c1 matches frozen confirm_predictions.fixed_rescore_UNION.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from earthquake.config import artifacts_dir, load_json
from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.gating.cache_io import attach_expected_s
from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s
from earthquake.pairwise.fulldev import build_phaseb_pairs, normalize_source


def _score_one_trace(payload: dict) -> dict | None:
    """Worker: score all UNION candidates for one trace with Stage-6 formula."""
    tn = payload["trace_name"]
    g = payload["cand_rows"]
    row = payload["meta_row"]
    hist_row = payload.get("hist_row")
    global_res = payload["global_res"]
    lw, lh, lp = payload["lw"], payload["lh"], payload["lp"]

    merged = pd.Series(row)
    if hist_row is not None:
        for k, v in hist_row.items():
            merged[k] = v
    exp = attach_expected_s(
        merged,
        global_res=global_res,
        shrink_k=50.0,
        min_history=5,
        mad_disable_s=1.0,
        min_sigma_s=0.05,
        max_sigma_s=1.0,
    )
    cands = []
    meta_c = []
    for r in g:
        ps = float(r["stead_probability"]) if r.get("stead_probability") is not None and np.isfinite(r["stead_probability"]) else -1.0
        pi = float(r["ida_probability"]) if r.get("ida_probability") is not None and np.isfinite(r["ida_probability"]) else -1.0
        p = max(ps, pi, 1e-6)
        sample = int(r["candidate_sample"])
        cands.append(
            PeakCandidate(
                sample_index=sample,
                absolute_utc=None,
                peak_probability=p,
                prominence=p,
                peak_width=float("nan"),
                local_entropy=0.0,
                rank=int(r.get("candidate_index", 0)),
                fallback_peak=False,
                phase="S",
            )
        )
        meta_c.append(r)
    best, rows = rescore_phase_candidates(
        cands,
        expected_sample=float(exp["expected_s_sample"]),
        sigma_samples=float(exp["history_sigma_samples"]),
        lambda_wave=lw,
        lambda_history=lh,
        lambda_prominence=lp,
        history_available=bool(exp["gate_history_available"]),
    )
    # map scores by sample_index (int)
    score_by = {int(r["sample_index"]): float(r["score"]) for r in rows}
    out_rows = []
    for r in meta_c:
        samp = int(r["candidate_sample"])
        out_rows.append(
            {
                **r,
                "fixed_score": score_by.get(samp, float("nan")),
                "expected_s_sample": float(exp["expected_s_sample"]),
                "history_sigma_samples": float(exp["history_sigma_samples"]),
                "history_available": bool(exp["gate_history_available"]),
                "base_tau_s": float(merged.get("base_tau_s", np.nan)) if np.isfinite(float(merged.get("base_tau_s", np.nan))) else np.nan,
                "base_delta_sp": float(merged.get("base_delta_sp", np.nan)) if str(merged.get("base_delta_sp", "")) != "nan" else np.nan,
            }
        )
    return {"trace_name": tn, "rows": out_rows, "best_sample": float(best.sample_index) if best is not None else np.nan}


def build_confirm_enriched(
    *,
    n_workers: int = 8,
    max_traces: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (enriched_candidates, manifest, stats)."""
    fc = artifacts_dir() / "results" / "stage6" / "final_confirm"
    union = pd.read_parquet(fc / "confirm_union.parquet")
    hist = pd.read_parquet(fc / "confirm_history_features.parquet").drop_duplicates("trace_name")
    man = pd.read_csv(fc / "confirm_s_eval_manifest.csv")
    man["trace_name"] = man["trace_name"].astype(str)
    if max_traces is not None:
        man = man.head(int(max_traces)).copy()
        keep = set(man.trace_name)
        union = union[union.trace_name.astype(str).isin(keep)].copy()

    lock = load_json(artifacts_dir() / "results" / "stage6" / "method_lock_stage6.json")
    global_res = lock["fixed_rescore"]["global_residual"]
    lw = float(lock["fixed_rescore"]["lambdas_s"]["lw"])
    lh = float(lock["fixed_rescore"]["lambdas_s"]["lh"])
    lp = float(lock["fixed_rescore"]["lambdas_s"]["lp"])

    hist_ix = hist.set_index("trace_name")
    by = {str(tn): g for tn, g in union.groupby(union.trace_name.astype(str), sort=False)}

    payloads = []
    for _, row in man.iterrows():
        tn = str(row.trace_name)
        g = by.get(tn)
        if g is None or len(g) == 0:
            continue
        cand_rows = []
        for _, r in g.iterrows():
            cand_rows.append(
                {
                    "trace_name": tn,
                    "event_id": str(r.event_id),
                    "candidate_index": int(r.candidate_index),
                    "candidate_sample": float(r.candidate_sample),
                    "stead_probability": float(r.stead_probability) if pd.notna(r.stead_probability) else np.nan,
                    "ida_probability": float(r.ida_probability) if pd.notna(r.ida_probability) else np.nan,
                    "candidate_source_mask": str(r.candidate_source_mask) if pd.notna(r.get("candidate_source_mask")) else str(r.get("candidate_source", "unknown")),
                    "sampling_rate_hz": float(r.sampling_rate_hz),
                    "top1_p_stead": float(r.top1_p_stead) if pd.notna(r.get("top1_p_stead")) else np.nan,
                    "top1_p_ida": float(r.top1_p_ida) if pd.notna(r.get("top1_p_ida")) else np.nan,
                    "s_arrival_sample": float(row.s_arrival_sample),
                    "origin_time": str(row.origin_time),
                    "trace_start_time": str(row.trace_start_time),
                }
            )
        hist_row = None
        if tn in hist_ix.index:
            hist_row = hist_ix.loc[tn].to_dict()
        payloads.append(
            {
                "trace_name": tn,
                "cand_rows": cand_rows,
                "meta_row": row.to_dict(),
                "hist_row": hist_row,
                "global_res": global_res,
                "lw": lw,
                "lh": lh,
                "lp": lp,
            }
        )

    scored: list[dict] = []
    best_map: dict[str, float] = {}
    if n_workers <= 1:
        for p in payloads:
            r = _score_one_trace(p)
            if r:
                scored.extend(r["rows"])
                best_map[r["trace_name"]] = r["best_sample"]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_score_one_trace, p) for p in payloads]
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    scored.extend(r["rows"])
                    best_map[r["trace_name"]] = r["best_sample"]

    enr = pd.DataFrame(scored)
    # cand_prob / source / resid
    enr["cand_prob"] = np.fmax(
        np.nan_to_num(enr["stead_probability"].to_numpy(float), nan=-1.0),
        np.nan_to_num(enr["ida_probability"].to_numpy(float), nan=-1.0),
    )
    enr["cand_prob"] = np.maximum(enr["cand_prob"].to_numpy(float), 1e-6)
    enr["source"] = enr["candidate_source_mask"].map(normalize_source)
    enr["pred_p_sample"] = np.where(
        np.isfinite(enr["top1_p_stead"].to_numpy(float)),
        enr["top1_p_stead"].to_numpy(float),
        enr["top1_p_ida"].to_numpy(float),
    )
    enr["tau_s"] = absolute_travel_time_s(
        origin_time=enr["origin_time"],
        trace_start_time=enr["trace_start_time"],
        sample=enr["candidate_sample"].to_numpy(float),
        sampling_rate_hz=enr["sampling_rate_hz"].to_numpy(float),
    )
    enr["resid_s"] = enr["tau_s"] - enr["base_tau_s"].to_numpy(float)
    enr["delta_sp"] = (enr["candidate_sample"].to_numpy(float) - enr["pred_p_sample"].to_numpy(float)) / enr[
        "sampling_rate_hz"
    ].to_numpy(float)
    enr["resid_sp"] = enr["delta_sp"] - enr["base_delta_sp"].to_numpy(float)

    stats = {
        "n_manifest_traces": int(len(man)),
        "n_events": int(man.event_id.nunique()),
        "n_scored_traces": int(enr.trace_name.nunique()),
        "n_candidates": int(len(enr)),
        "lw": lw,
        "lh": lh,
        "lp": lp,
        "scoring": "stage6_rescore_phase_candidates+attach_expected_s",
    }
    return enr, man, stats


def build_confirm_pairs(
    enriched: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    frozen_fixed_by_trace: dict[str, float] | None = None,
    agree_tol: float = 1e-3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = build_phaseb_pairs(enriched, manifest)
    info: dict[str, Any] = {"n_pairs": len(pairs), "n_ge2": int((pairs.n_candidates >= 2).sum())}
    if frozen_fixed_by_trace is not None:
        frozen = np.asarray([frozen_fixed_by_trace.get(str(t), np.nan) for t in pairs.trace_name], float)
        c1 = pairs.c1_sample.to_numpy(float)
        ok = np.isfinite(frozen) & np.isfinite(c1)
        agree = float(np.mean(np.abs(c1[ok] - frozen[ok]) < agree_tol)) if ok.any() else 0.0
        info["c1_vs_frozen_fixed_agree"] = agree
        info["n_disagree"] = int(np.sum(ok & (np.abs(c1 - frozen) >= agree_tol)))
        if agree < 0.999:
            raise RuntimeError(f"c1 vs frozen fixed_rescore_UNION agree={agree} < 0.999")
    return pairs, info
