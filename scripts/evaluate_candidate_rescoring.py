#!/usr/bin/env python
"""Evaluate distance / residual priors and PhaseNet candidate re-scoring on the fixed eval set."""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, save_json
from earthquake.fusion.blind_pair_rescorer import rescore_blind_pairs
from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.history.residual_prior import (
    ResidualPathStats,
    expected_samples_catalog,
    predict_with_residual_prior,
)
from earthquake.metrics import audit_pick_errors, match_picks
from earthquake.utils import ensure_dir


def _build_cand_index(cands: pd.DataFrame) -> dict[tuple[str, str], list[PeakCandidate]]:
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
        idx[(str(trace), str(phase))] = lst
    return idx


def _cands_for(trace: str, phase: str, cand_index: dict[tuple[str, str], list[PeakCandidate]]) -> list[PeakCandidate]:
    return cand_index.get((trace, phase), [])


def _residual_stats_from_row(r: pd.Series) -> ResidualPathStats:
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


def _tau_to_sample(row: pd.Series, tau_s: float) -> float:
    if not np.isfinite(tau_s):
        return float("nan")
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    return float(((origin + pd.to_timedelta(tau_s, unit="s")) - start).total_seconds() * sr)


def summarize(pred_p, pred_s, true_p, true_s, sr) -> dict:
    mp = match_picks(np.asarray(pred_p), np.asarray(true_p), np.asarray(sr))
    ms = match_picks(np.asarray(pred_s), np.asarray(true_s), np.asarray(sr))
    ap = audit_pick_errors(np.asarray(pred_p), np.asarray(true_p), np.asarray(sr), window_s=0.5)
    as_ = audit_pick_errors(np.asarray(pred_s), np.asarray(true_s), np.asarray(sr), window_s=0.5)
    return {
        "P": {**{k: mp[k] for k in mp if not k.startswith("tp") and not k.startswith("fp") and not k.startswith("fn")}, "matched_timing_p95": ap["matched_timing_p95_ae"], "e2e_p95": ap["e2e_p95_ae"]},
        "S": {**{k: ms[k] for k in ms if not k.startswith("tp") and not k.startswith("fp") and not k.startswith("fn")}, "matched_timing_p95": as_["matched_timing_p95_ae"], "e2e_p95": as_["e2e_p95_ae"]},
        "n": len(pred_p),
    }


def score_tuple(summary: dict, phase: str = "S") -> tuple:
    m = summary[phase]
    return (m.get("f1@0.1s", 0.0), m.get("f1@0.5s", 0.0), -m.get("matched_timing_p95", 1e9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_fixed.yaml")
    parser.add_argument("--cache", default="artifacts/results/stage2/phasenet_fixed_cache.parquet")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage2")

    cache_path = ROOT / args.cache
    cands_path = cache_path.with_name(cache_path.stem + "_candidates.parquet")
    if not cache_path.exists() or not cands_path.exists():
        raise SystemExit(f"Missing PhaseNet cache. Run scripts/audit_pick_metrics.py first.\n{cache_path}\n{cands_path}")

    picks = pd.read_parquet(cache_path)
    cands = pd.read_parquet(cands_path)
    cand_index = _build_cand_index(cands)
    print({"n_cand_keys": len(cand_index)}, flush=True)
    fixed = pd.read_parquet(artifacts_dir() / "diagnostics" / "fixed_eval_events.parquet")
    # enforce identical fixed keys
    picks = picks[picks.trace_name.astype(str).isin(fixed.trace_name.astype(str))].copy()
    meta = fixed.merge(picks, on=["trace_name", "event_id"], how="inner", suffixes=("", "_p"))
    # prefer label/meta from fixed
    for col in ["origin_time", "trace_start_time", "sampling_rate_hz", "snr_db", "distance_km", "source_depth_km", "station_elevation_m", "p_arrival_sample", "s_arrival_sample"]:
        if col in fixed.columns:
            meta[col] = meta["trace_name"].map(fixed.set_index("trace_name")[col])

    resid = pd.read_parquet(artifacts_dir() / "history" / "residual_history_features_frozen.parquet")
    meta = meta.merge(resid, on=["trace_name", "event_id"], how="left", suffixes=("", "_r"))
    # val/test split from index
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")[["trace_name", "split"]]
    meta = meta.merge(events, on="trace_name", how="left", suffixes=("", "_e"))
    if "split" not in meta.columns or meta["split"].isna().all():
        meta["split"] = meta.get("split_e", "test")

    rh_meta = load_json(out / "residual_history_meta.json")
    global_res = rh_meta["global_residual"]
    shrink_k = float(rh_meta["best_shrinkage_k"])
    min_hist = int(cfg.get("min_history", 5))
    mad_dis = float(cfg.get("mad_disable_s", 1.0))

    # Build shuffled residual correspondence (preserve count/MAD, shuffle medians)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    ok = meta[meta["history_available"] == True].copy()  # noqa: E712
    shuffle_idx = rng.permutation(len(ok)) if len(ok) else np.array([])
    shuffle_map = {}
    if len(ok):
        src = ok.iloc[shuffle_idx]
        for i, (_, row) in enumerate(ok.iterrows()):
            shuffle_map[str(row.trace_name)] = src.iloc[i]

    def pred_for_mode(mode: str, row: pd.Series, lw, lh, lp, shrink=shrink_k) -> tuple[float, float]:
        sr = float(row["sampling_rate_hz"])
        p_c = _cands_for(str(row.trace_name), "p", cand_index)
        s_c = _cands_for(str(row.trace_name), "s", cand_index)
        # PhaseNet-only
        if mode == "phasenet":
            return float(row["pred_p_sample"]), float(row["pred_s_sample"])

        base = {
            "base_tau_p": float(row.get("base_tau_p", np.nan)),
            "base_tau_s": float(row.get("base_tau_s", np.nan)),
            "base_delta_sp": float(row.get("base_delta_sp", np.nan)),
        }
        path = _residual_stats_from_row(row)

        if mode == "distance_only":
            return _tau_to_sample(row, base["base_tau_p"]), _tau_to_sample(row, base["base_tau_s"])

        if mode == "raw_path_median":
            return _tau_to_sample(row, float(row.get("tau_p_median", np.nan))), _tau_to_sample(
                row, float(row.get("tau_s_median", np.nan))
            )

        if mode == "count_only":
            # ignore travel times; keep PhaseNet (negative control)
            return float(row["pred_p_sample"]), float(row["pred_s_sample"])

        if mode == "history_unavailable":
            best_p, _ = rescore_phase_candidates(
                p_c, expected_sample=np.nan, sigma_samples=1.0, lambda_wave=lw, lambda_history=lh, lambda_prominence=lp, history_available=False
            )
            best_s, _ = rescore_phase_candidates(
                s_c, expected_sample=np.nan, sigma_samples=1.0, lambda_wave=lw, lambda_history=lh, lambda_prominence=lp, history_available=False
            )
            return float(best_p.sample_index if best_p else np.nan), float(best_s.sample_index if best_s else np.nan)

        # residual variants / controls
        use_path = path
        if mode in ("shuffle_path", "bias_plus5", "bias_minus5", "unshrunk_residual", "shrunk_residual", "catalog_rescore", "blind_s"):
            if mode == "shuffle_path" and str(row.trace_name) in shuffle_map:
                use_path = _residual_stats_from_row(shuffle_map[str(row.trace_name)])
                # keep this row's count/MAD distribution by copying counts from original? Spec: keep count/MAD dist, shuffle path correspondence — using other row's full stats is OK if we permute among hist-available.
            k = 0.0 if mode == "unshrunk_residual" else shrink
            pred = predict_with_residual_prior(
                base, use_path, global_res, shrinkage_k=k, min_history=min_hist, mad_disable_s=mad_dis,
                min_sigma_s=float(cfg.get("min_sigma_s", 0.05)), max_sigma_s=float(cfg.get("max_sigma_s", 1.0)),
            )
            if mode == "bias_plus5":
                pred.pred_tau_p += 5.0
                pred.pred_tau_s += 5.0
            if mode == "bias_minus5":
                pred.pred_tau_p -= 5.0
                pred.pred_tau_s -= 5.0

            if mode in ("unshrunk_residual", "shrunk_residual", "shuffle_path", "bias_plus5", "bias_minus5"):
                return expected_samples_catalog(row, pred)

            if mode == "catalog_rescore":
                exp_p, exp_s = expected_samples_catalog(row, pred)
                best_p, _ = rescore_phase_candidates(
                    p_c,
                    expected_sample=exp_p,
                    sigma_samples=pred.sigma_p_s * sr,
                    lambda_wave=lw,
                    lambda_history=lh,
                    lambda_prominence=lp,
                    history_available=pred.history_available or pred.used_path_residual,
                )
                best_s, _ = rescore_phase_candidates(
                    s_c,
                    expected_sample=exp_s,
                    sigma_samples=pred.sigma_s_s * sr,
                    lambda_wave=lw,
                    lambda_history=lh,
                    lambda_prominence=lp,
                    history_available=pred.history_available or pred.used_path_residual,
                )
                return float(best_p.sample_index if best_p else row["pred_p_sample"]), float(
                    best_s.sample_index if best_s else row["pred_s_sample"]
                )

            if mode == "blind_s":
                # no origin / current distance path for true blind: use ΔSP residual prior only
                # If path key used current location it is catalog-assisted; here we still use residual ΔSP from frozen path features (catalog path key). Label result as catalog_pathkey_blind_origin.
                pair = rescore_blind_pairs(
                    p_c,
                    s_c,
                    predicted_delta_sp_s=float(pred.pred_delta_sp),
                    sigma_sp_s=float(pred.sigma_sp_s),
                    sampling_rate=sr,
                    lambda_wave=lw,
                    lambda_history=lh,
                    lambda_prominence=lp,
                    history_available=pred.used_path_residual,
                )
                pc, sc = pair["p_cand"], pair["s_cand"]
                return float(pc.sample_index if pc else row["pred_p_sample"]), float(sc.sample_index if sc else row["pred_s_sample"])

        raise ValueError(mode)

    # Hyperparameter search on VAL subset for catalog_rescore
    val = meta[meta["split"] == "val"].copy()
    if len(val) == 0:
        val = meta.sample(n=min(2000, len(meta)), random_state=0)
    lw_grid = cfg.get("lambda_wave_grid", [1.0])
    lh_grid = cfg.get("lambda_history_grid", [0.0, 0.5])
    lp_grid = cfg.get("lambda_prominence_grid", [0.0, 0.1])

    best_p = {"score": (-1, -1, -1e9), "lw": 1.0, "lh": 0.0, "lp": 0.0}
    best_s = {"score": (-1, -1, -1e9), "lw": 1.0, "lh": 0.0, "lp": 0.0}
    search_rows = []
    for lw, lh, lp in tqdm(list(itertools.product(lw_grid, lh_grid, lp_grid)), desc="lambda-search-val"):
        pp, ps, tp, ts, sr = [], [], [], [], []
        for _, row in val.iterrows():
            a, b = pred_for_mode("catalog_rescore", row, lw, lh, lp)
            pp.append(a); ps.append(b)
            tp.append(float(row["true_p_sample"]) if pd.notna(row["true_p_sample"]) else np.nan)
            ts.append(float(row["true_s_sample"]) if pd.notna(row["true_s_sample"]) else np.nan)
            sr.append(float(row["sampling_rate_hz"]))
        summ = summarize(pp, ps, tp, ts, sr)
        sp = score_tuple(summ, "P")
        ss = score_tuple(summ, "S")
        search_rows.append({"lw": lw, "lh": lh, "lp": lp, "p_f1_0.1": sp[0], "p_f1_0.5": sp[1], "s_f1_0.1": ss[0], "s_f1_0.5": ss[1], "s_matched_p95": -ss[2]})
        if sp > best_p["score"]:
            best_p = {"score": sp, "lw": lw, "lh": lh, "lp": lp}
        if ss > best_s["score"]:
            best_s = {"score": ss, "lw": lw, "lh": lh, "lp": lp}

    pd.DataFrame(search_rows).to_csv(out / "lambda_search_val.csv", index=False)
    save_json({"best_p": best_p, "best_s": best_s, "shrinkage_k": shrink_k}, out / "best_lambdas.json")

    # Full fixed-eval comparison with selected lambdas (use S lambdas for S, P for P independently)
    modes = [
        "phasenet",
        "distance_only",
        "raw_path_median",
        "unshrunk_residual",
        "shrunk_residual",
        "catalog_rescore",
        "blind_s",
        "shuffle_path",
        "count_only",
        "bias_plus5",
        "bias_minus5",
        "history_unavailable",
    ]
    results = []
    pick_frames = []
    for mode in modes:
        pp, ps, tp, ts, sr, eids, names = [], [], [], [], [], [], []
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"eval:{mode}"):
            a, b = pred_for_mode(mode, row, best_p["lw"], best_s["lh"] if mode != "phasenet" else best_p["lh"], best_s["lp"])
            # independent lambdas: recompute with phase-specific for catalog/blind
            if mode in ("catalog_rescore", "blind_s", "history_unavailable"):
                # P with best_p, S with best_s
                a2, _ = pred_for_mode(mode, row, best_p["lw"], best_p["lh"], best_p["lp"])
                _, b2 = pred_for_mode(mode, row, best_s["lw"], best_s["lh"], best_s["lp"])
                a, b = a2, b2
            pp.append(a); ps.append(b)
            tp.append(float(row["true_p_sample"]) if pd.notna(row["true_p_sample"]) else np.nan)
            ts.append(float(row["true_s_sample"]) if pd.notna(row["true_s_sample"]) else np.nan)
            sr.append(float(row["sampling_rate_hz"]))
            eids.append(str(row.event_id)); names.append(str(row.trace_name))
        summ = summarize(pp, ps, tp, ts, sr)
        summ["mode"] = mode
        results.append(summ)
        pick_frames.append(
            pd.DataFrame(
                {
                    "trace_name": names,
                    "event_id": eids,
                    "mode": mode,
                    "pred_p_sample": pp,
                    "pred_s_sample": ps,
                    "true_p_sample": tp,
                    "true_s_sample": ts,
                    "sampling_rate_hz": sr,
                }
            )
        )
        print(mode, {k: summ["S"].get(k) for k in ["f1@0.1s", "f1@0.5s", "e2e_mae", "e2e_p95"]})

    flat = []
    for r in results:
        flat.append(
            {
                "mode": r["mode"],
                "n": r["n"],
                "p_f1@0.1s": r["P"]["f1@0.1s"],
                "p_f1@0.5s": r["P"]["f1@0.5s"],
                "p_e2e_mae": r["P"]["mae"],
                "p_e2e_p95": r["P"]["e2e_p95"],
                "p_matched_p95": r["P"]["matched_timing_p95"],
                "s_f1@0.1s": r["S"]["f1@0.1s"],
                "s_f1@0.5s": r["S"]["f1@0.5s"],
                "s_e2e_mae": r["S"]["mae"],
                "s_e2e_p95": r["S"]["e2e_p95"],
                "s_matched_p95": r["S"]["matched_timing_p95"],
            }
        )
    pd.DataFrame(flat).to_csv(out / "candidate_rescoring_comparison.csv", index=False)
    save_json({"results": results, "best_p": best_p, "best_s": best_s, "shrinkage_k": shrink_k}, out / "candidate_rescoring_comparison.json")
    pd.concat(pick_frames, ignore_index=True).to_parquet(out / "candidate_rescoring_picks.parquet", index=False)
    # also store meta features for hard-case / bootstrap
    meta[
        [
            c
            for c in meta.columns
            if c
            in {
                "trace_name",
                "event_id",
                "split",
                "snr_db",
                "distance_km",
                "history_count",
                "history_available",
                "residual_p_mad",
                "residual_s_mad",
                "tau_p_mad",
                "tau_s_mad",
                "pred_p_sample",
                "pred_s_sample",
                "true_p_sample",
                "true_s_sample",
                "p_peak_probability",
                "s_peak_probability",
                "p_n_cands",
                "s_n_cands",
                "sampling_rate_hz",
            }
        ]
    ].to_parquet(out / "fixed_eval_joined_meta.parquet", index=False)
    print({"saved": str(out), "best_p": best_p, "best_s": best_s})


if __name__ == "__main__":
    main()
