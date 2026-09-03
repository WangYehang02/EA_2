#!/usr/bin/env python
"""Candidate coverage / oracle ceilings for Stage-3 learned gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, save_json
from earthquake.gating.cache_io import (
    attach_expected_s,
    build_cand_index,
    fixed_rescore_s,
    phasenet_s_pick,
)
from earthquake.metrics import match_picks
from earthquake.utils import ensure_dir


def _err_s(pred, true, sr):
    if not (np.isfinite(pred) and np.isfinite(true) and np.isfinite(sr) and sr > 0):
        return np.nan
    return abs(pred - true) / sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_debug.yaml")
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional parquet of traces to evaluate; default=test-only list joined with residual+stage3/stage2 cache",
    )
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage3")
    k_grid = list(cfg.get("candidate_k_grid", [1, 3, 5, 10]))
    tol_grid = [0.1, 0.5, 1.0]

    # load traces
    test_list = ROOT / cfg.get("test_list", "artifacts/diagnostics/fixed_eval_test_only_traces.txt")
    names = [ln.strip() for ln in test_list.read_text().splitlines() if ln.strip()]
    n_test = int(cfg.get("n_test_traces", len(names)))
    names = names[:n_test]

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    resid = pd.read_parquet(artifacts_dir() / "history" / "residual_history_features_frozen.parquet")
    meta = events[events.trace_name.astype(str).isin(names)].copy()
    meta = meta.merge(resid, on=["trace_name", "event_id"], how="left", suffixes=("", "_r"))
    order = {n: i for i, n in enumerate(names)}
    meta["_ord"] = meta.trace_name.astype(str).map(order)
    meta = meta.sort_values("_ord").drop(columns=["_ord"])

    # candidates: prefer stage3 cache then stage2
    cand_paths = [
        artifacts_dir() / "candidates" / "stage3_phasenet_cache_candidates.parquet",
        artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache_candidates.parquet",
    ]
    pick_paths = [
        artifacts_dir() / "candidates" / "stage3_phasenet_cache.parquet",
        artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache.parquet",
    ]
    cands = None
    for p in cand_paths:
        if p.exists():
            cands = pd.read_parquet(p)
            break
    picks = None
    for p in pick_paths:
        if p.exists():
            picks = pd.read_parquet(p)
            break
    if cands is None or picks is None:
        raise SystemExit("Missing candidate cache. Run build_gate_dataset.py or audit_pick_metrics.py first.")

    meta = meta.merge(
        picks[
            [
                c
                for c in [
                    "trace_name",
                    "pred_p_sample",
                    "pred_s_sample",
                    "true_p_sample",
                    "true_s_sample",
                    "sampling_rate_hz",
                ]
                if c in picks.columns
            ]
        ],
        on="trace_name",
        how="inner",
        suffixes=("", "_pk"),
    )
    if "s_arrival_sample" in meta.columns:
        meta["true_s_sample"] = meta["s_arrival_sample"]
    if "p_arrival_sample" in meta.columns:
        meta["true_p_sample"] = meta["p_arrival_sample"]
    if "sampling_rate_hz_pk" in meta.columns and "sampling_rate_hz" not in meta.columns:
        meta["sampling_rate_hz"] = meta["sampling_rate_hz_pk"]

    cand_index = build_cand_index(cands)
    rh_meta = load_json(artifacts_dir() / "results" / "stage2" / "residual_history_meta.json")
    global_res = rh_meta["global_residual"]
    best = load_json(artifacts_dir() / "results" / "stage2" / "best_lambdas.json")
    lw, lh, lp = best["best_s"]["lw"], best["best_s"]["lh"], best["best_s"]["lp"]

    recall_rows = []
    oracle_rows = []
    pn_s, fr_s, ora_sel, true_s, sr_arr = [], [], [], [], []

    max_k_avail = 0
    for _, row in meta.iterrows():
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        max_k_avail = max(max_k_avail, len(s_c))
        true = float(row.true_s_sample) if pd.notna(row.true_s_sample) else np.nan
        sr = float(row.sampling_rate_hz)
        exp = attach_expected_s(
            row,
            global_res=global_res,
            shrink_k=float(cfg.get("shrinkage_k", 50)),
            min_history=int(cfg.get("min_history", 5)),
            mad_disable_s=float(cfg.get("mad_threshold", 1.0)),
            min_sigma_s=float(cfg.get("min_sigma_s", 0.05)),
            max_sigma_s=float(cfg.get("max_sigma_s", 1.0)),
        )
        hist_ok = bool(exp["gate_history_available"]) and bool(row.get("history_available", False)) and np.isfinite(
            exp["expected_s_sample"]
        )
        pn = phasenet_s_pick(s_c, float(row.get("pred_s_sample", np.nan)))
        fr = fixed_rescore_s(
            s_c,
            expected_s=exp["expected_s_sample"],
            sigma_samples=exp["history_sigma_samples"],
            history_available=hist_ok,
            lw=lw,
            lh=lh,
            lp=lp,
        )
        pn_err = _err_s(pn, true, sr)
        fr_err = _err_s(fr, true, sr)
        if np.isfinite(pn_err) and np.isfinite(fr_err):
            sel = pn if pn_err <= fr_err else fr
        elif np.isfinite(pn_err):
            sel = pn
        else:
            sel = fr
        pn_s.append(pn)
        fr_s.append(fr)
        ora_sel.append(sel)
        true_s.append(true)
        sr_arr.append(sr)

        for K in k_grid:
            top = s_c[:K]
            if not top or not np.isfinite(true):
                for tol in tol_grid:
                    recall_rows.append({"K": K, "tol_s": tol, "hit": 0, "n": 1})
                oracle_rows.append({"K": K, "pred": np.nan, "true": true, "sr": sr})
                continue
            errs = [abs(c.sample_index - true) / sr for c in top]
            best_i = int(np.argmin(errs))
            oracle_rows.append({"K": K, "pred": float(top[best_i].sample_index), "true": true, "sr": sr})
            for tol in tol_grid:
                recall_rows.append({"K": K, "tol_s": tol, "hit": int(min(errs) <= tol), "n": 1})

    recall_df = pd.DataFrame(recall_rows)
    recall_agg = (
        recall_df.groupby(["K", "tol_s"], as_index=False)
        .agg(hits=("hit", "sum"), n=("n", "sum"))
        .assign(recall=lambda d: d["hits"] / d["n"].clip(lower=1))
    )
    recall_agg.to_csv(out / "candidate_recall_by_k.csv", index=False)

    ora_df = pd.DataFrame(oracle_rows)
    ceiling_rows = []
    for K in k_grid:
        sub = ora_df[ora_df["K"] == K]
        m = match_picks(sub["pred"].to_numpy(), sub["true"].to_numpy(), sub["sr"].to_numpy())
        ceiling_rows.append(
            {
                "K": K,
                "n": len(sub),
                "oracle_f1@0.1": m["f1@0.1s"],
                "oracle_f1@0.5": m["f1@0.5s"],
                "oracle_e2e_mae": m["mae"],
                "oracle_e2e_p95": m["p95_ae"],
            }
        )
    ceil = pd.DataFrame(ceiling_rows)
    ceil.to_csv(out / "oracle_candidate_ceiling.csv", index=False)

    m_pn = match_picks(np.asarray(pn_s), np.asarray(true_s), np.asarray(sr_arr))
    m_fr = match_picks(np.asarray(fr_s), np.asarray(true_s), np.asarray(sr_arr))
    m_sel = match_picks(np.asarray(ora_sel), np.asarray(true_s), np.asarray(sr_arr))

    # choose default K: if K=5 vs K=10 F1@0.5 diff < 0.005 use 5
    f5 = float(ceil.loc[ceil.K == 5, "oracle_f1@0.5"].iloc[0]) if (ceil.K == 5).any() else np.nan
    f10 = float(ceil.loc[ceil.K == 10, "oracle_f1@0.5"].iloc[0]) if (ceil.K == 10).any() else np.nan
    if np.isfinite(f5) and np.isfinite(f10) and abs(f10 - f5) < 0.005:
        recommended_k = 5
    else:
        recommended_k = int(ceil.sort_values("oracle_f1@0.5", ascending=False).iloc[0]["K"]) if len(ceil) else 5

    report = {
        "n_traces": int(len(meta)),
        "max_k_available_in_cache": int(max_k_avail),
        "note_k10": (
            "If max_k_available < 10, K=10 recall/oracle equals available peaks (often Stage2 K=5 cache)."
            if max_k_avail < 10
            else "Cache provides >=10 candidates for some traces."
        ),
        "phasenet": {k: m_pn[k] for k in ["f1@0.1s", "f1@0.5s", "mae", "p95_ae"]},
        "fixed_catalog_rescore": {k: m_fr[k] for k in ["f1@0.1s", "f1@0.5s", "mae", "p95_ae"]},
        "oracle_source_selector": {k: m_sel[k] for k in ["f1@0.1s", "f1@0.5s", "mae", "p95_ae"]},
        "recommended_candidate_k": recommended_k,
        "oracle_candidate_ceiling": ceiling_rows,
    }
    save_json(report, out / "oracle_selector_ceiling.json")
    print(report, flush=True)


if __name__ == "__main__":
    main()
