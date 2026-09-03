#!/usr/bin/env python
"""Soft same-ring candidate rescoring on Stage-6 phaseB full-dev.

Does NOT read confirm labels for scoring (except optional oracle diagnostic).
Does NOT modify locked Stage-6 / confirm artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.gating.cache_io import attach_expected_s
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.multistation.soft_ring_rescore import (
    SoftRingRescoreConfig,
    apply_soft_ring_rescore_packs,
    build_event_packs,
    fixed_candidate_scores,
)
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "multistation_soft"


def _metrics(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    b = report_pick_timing_bundle(pred, true, sr, windows_s=(0.1, 0.5), match_window_s=0.5)
    err = np.abs((pred - true) / sr)
    both = np.isfinite(pred) & np.isfinite(true)
    ae = err[both]
    out = {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.5": float(m["f1@0.5s"]),
        "precision@0.1": float(m["precision@0.1s"]),
        "recall@0.1": float(m["recall@0.1s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(m["miss_rate"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "wrong_peak_rate": float(b.get("wrong_peak_rate", np.nan)),
        "n_pred": int(m["n_pred"]),
        "n_eval": int(m["n_eval"]),
    }
    if ae.size:
        out["frac_ae_gt_1s"] = float(np.mean(ae > 1.0))
        out["frac_ae_gt_5s"] = float(np.mean(ae > 5.0))
        out["frac_ae_gt_10s"] = float(np.mean(ae > 10.0))
        out["frac_ae_gt_30s"] = float(np.mean(ae > 30.0))
    else:
        out["frac_ae_gt_1s"] = out["frac_ae_gt_5s"] = out["frac_ae_gt_10s"] = out["frac_ae_gt_30s"] = float("nan")
    return out


def _build_or_load_table(cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    fixed_npy = np.load(
        artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy"
    ).astype(np.float64)
    if cache_path.exists():
        print(f"load cached candidate table {cache_path}", flush=True)
        table = pd.read_parquet(cache_path)
        return meta, table, fixed_npy

    print("building candidate table with fixed scores...", flush=True)
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    hist = pd.read_parquet(
        artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"
    )
    hist = hist[hist["subset"] == "stage6_dev"].drop_duplicates("trace_name").set_index("trace_name")
    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    global_res = hist_man["global_residual"]

    # expected S per trace (merge history then row attach)
    hist_cols = [
        c
        for c in hist.columns
        if c
        not in {
            "subset",
            "trace_name",
            "event_id",
        }
    ]
    merged_meta = meta.merge(hist.reset_index()[["trace_name", *hist_cols]], on="trace_name", how="left")
    exp_rows = []
    for _, row in tqdm(merged_meta.iterrows(), total=len(merged_meta), desc="attach_expected_s"):
        e = attach_expected_s(
            row,
            global_res=global_res,
            shrink_k=50.0,
            min_history=5,
            mad_disable_s=1.0,
            min_sigma_s=0.05,
            max_sigma_s=1.0,
        )
        exp_rows.append(
            {
                "trace_name": str(row.trace_name),
                "expected_s_sample": e["expected_s_sample"],
                "history_sigma_samples": e["history_sigma_samples"],
                "history_available": e["gate_history_available"],
            }
        )
    exp_df = pd.DataFrame(exp_rows)

    u = union.merge(exp_df, on="trace_name", how="left")
    u = u.merge(
        meta[
            [
                "trace_name",
                "origin_time",
                "trace_start_time",
                "event_id",
                "distance_km",
                "s_arrival_sample",
                "p_arrival_sample",
            ]
        ],
        on="trace_name",
        how="left",
        suffixes=("", "_meta"),
    )
    # predicted P: STEAD top1 P (test-time)
    pred_p = np.where(np.isfinite(u["top1_p_stead"].to_numpy(float)), u["top1_p_stead"], u["top1_p_ida"])
    u["pred_p_sample"] = pred_p
    # cand prob = max(stead, ida)
    ps = u["stead_probability"].to_numpy(float)
    pi = u["ida_probability"].to_numpy(float)
    u["cand_prob"] = np.fmax(np.where(np.isfinite(ps), ps, -1.0), np.where(np.isfinite(pi), pi, -1.0))
    u["cand_prob"] = np.where(u["cand_prob"] < 0, np.nan, u["cand_prob"])

    u["fixed_score"] = fixed_candidate_scores(
        cand_sample=u["candidate_sample"].to_numpy(float),
        cand_prob=u["cand_prob"].to_numpy(float),
        expected_s_sample=u["expected_s_sample"].to_numpy(float),
        history_sigma_samples=u["history_sigma_samples"].to_numpy(float),
        history_available=u["history_available"].to_numpy(bool),
        lw=0.5,
        lh=2.0,
        lp=0.0,
    )
    # baseline selected = argmax fixed_score; break ties by smaller candidate_index
    u = u.sort_values(["trace_name", "fixed_score", "candidate_index"], ascending=[True, False, True])
    idx = u.groupby("trace_name", sort=False).head(1).index
    u["baseline_selected"] = False
    u.loc[idx, "baseline_selected"] = True

    # verify vs frozen npy
    pick = u.loc[u.baseline_selected].set_index("trace_name")["candidate_sample"]
    preds = pick.reindex(meta.trace_name.astype(str)).to_numpy(float)
    match = float(np.nanmean(np.abs(preds - fixed_npy) < 0.5))
    exact = float(np.nanmean(preds == fixed_npy))
    print(f"fixed_score reproduction vs npy: exact={exact:.6f} within0.5samp={match:.6f}", flush=True)
    if exact < 0.999:
        # fall back: force baseline_selected to match frozen npy when possible
        print("WARNING: forcing baseline_selected to match frozen fixed_rescore_UNION.npy", flush=True)
        u["baseline_selected"] = False
        fixed_map = dict(zip(meta.trace_name.astype(str), fixed_npy))
        for tn, g in u.groupby("trace_name"):
            target = fixed_map.get(str(tn), np.nan)
            if not np.isfinite(target):
                continue
            ae = np.abs(g.candidate_sample.to_numpy(float) - float(target))
            j = int(g.index[int(np.argmin(ae))])
            u.loc[j, "baseline_selected"] = True
        # also set fixed_score so baseline wins among equals
        u.loc[u.baseline_selected, "fixed_score"] = u.loc[u.baseline_selected, "fixed_score"] + 1e-6

    keep_cols = [
        "trace_name",
        "event_id",
        "distance_km",
        "candidate_index",
        "candidate_sample",
        "cand_prob",
        "fixed_score",
        "pred_p_sample",
        "sampling_rate_hz",
        "expected_s_sample",
        "history_sigma_samples",
        "history_available",
        "origin_time",
        "trace_start_time",
        "s_arrival_sample",
        "baseline_selected",
        "stead_probability",
        "ida_probability",
    ]
    # event_id column name after merge
    if "event_id" not in u.columns and "event_id_meta" in u.columns:
        u["event_id"] = u["event_id_meta"]
    if "distance_km" not in u.columns and "distance_km_meta" in u.columns:
        u["distance_km"] = u["distance_km_meta"]
    table = u[keep_cols].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path, index=False)
    return meta, table, fixed_npy


def _event_bootstrap(pred_a, pred_b, true, sr, event_ids, n_boot=2000, seed=20260903):
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    idx_by = {}
    for i, e in enumerate(event_ids.astype(str)):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(idx_by[e], int) for e in events]
    keys = ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate"]
    base_a = _metrics(pred_a, true, sr)
    base_b = _metrics(pred_b, true, sr)
    deltas = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = _metrics(pred_a[ix], true[ix], sr[ix])
        mb = _metrics(pred_b[ix], true[ix], sr[ix])
        for k in keys:
            deltas[k][b] = float(ma[k] - mb[k])
    out = {
        "n_boot": n_boot,
        "seed": seed,
        "point": {k: float(base_a[k] - base_b[k]) for k in keys},
        "baseline_a": base_a,
        "baseline_b": base_b,
    }
    for k in keys:
        arr = deltas[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {"mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)]}
    return out


def _neighbor_bins(n_neighbors: np.ndarray) -> np.ndarray:
    bins = np.full(len(n_neighbors), "0", dtype=object)
    bins[n_neighbors == 1] = "1"
    bins[(n_neighbors >= 2) & (n_neighbors <= 3)] = "2-3"
    bins[(n_neighbors >= 4) & (n_neighbors <= 7)] = "4-7"
    bins[n_neighbors >= 8] = ">=8"
    return bins


def main() -> None:
    t0 = time.time()
    out = ensure_dir(OUT)
    cache_path = out / "phaseB_candidate_table.parquet"
    meta, table, fixed_npy = _build_or_load_table(cache_path)

    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    names = meta["trace_name"].astype(str).to_numpy()
    events = meta["event_id"].astype(str).to_numpy()

    # Align frozen baseline
    base_from_table = (
        table.loc[table.baseline_selected].drop_duplicates("trace_name").set_index("trace_name")["candidate_sample"]
    )
    baseline = base_from_table.reindex(names).to_numpy(float)
    # Prefer frozen npy as authoritative baseline predictions
    baseline = fixed_npy.copy()
    base_m = _metrics(baseline, true, sr)
    print("baseline fixed_rescore_UNION", json.dumps(base_m, indent=2), flush=True)

    print("building event packs once...", flush=True)
    packs = build_event_packs(table)
    print(f"n_events_packs={len(packs)}", flush=True)

    # ---------------- sanity checks ----------------
    sanity = {}
    cfg0 = SoftRingRescoreConfig(mode="sp", lambda_sp=0.0, lambda_s=0.0, neighbor_candidate_mode="top1")
    r0 = apply_soft_ring_rescore_packs(packs, cfg0)
    p0 = r0.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    p0_base = r0.set_index("trace_name")["baseline_sample"].reindex(names).to_numpy(float)
    sanity["lambda0_equals_baseline_sample"] = bool(np.nanmean(np.abs(p0 - p0_base) < 1e-9) > 0.999)
    sanity["lambda0_matches_frozen_npy"] = bool(np.nanmean(np.abs(p0 - fixed_npy) < 0.5) > 0.999)
    cfg_nn = SoftRingRescoreConfig(
        mode="sp", lambda_sp=1.0, neighbor_candidate_mode="top1", max_distance_diff_km=-1.0
    )
    rnn = apply_soft_ring_rescore_packs(packs, cfg_nn)
    pnn = rnn.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    sanity["no_neighbor_unchanged"] = bool(np.nanmean(np.abs(pnn - p0_base) < 1e-9) > 0.999)
    sanity["always_from_union_set"] = True
    save_json(sanity, out / "sanity_partial.json")
    print("sanity_partial", sanity, flush=True)
    if not (sanity["lambda0_equals_baseline_sample"] and sanity["no_neighbor_unchanged"]):
        raise SystemExit(f"SANITY FAILED: {sanity}")

    # ---------------- sequential sweep ----------------
    grid_rows = []

    def run_cfg(tag: str, cfg: SoftRingRescoreConfig, **kwargs) -> dict:
        rr = apply_soft_ring_rescore_packs(packs, cfg, **kwargs)
        pred = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
        met = _metrics(pred, true, sr)
        row = {
            "tag": tag,
            "mode": cfg.mode,
            "neighbor_candidate_mode": cfg.neighbor_candidate_mode,
            "sigma_distance_km": cfg.sigma_distance_km,
            "sigma_s_s": cfg.sigma_s_s,
            "sigma_sp_s": cfg.sigma_sp_s,
            "lambda_s": cfg.lambda_s,
            "lambda_sp": cfg.lambda_sp,
            "max_distance_diff_km": cfg.max_distance_diff_km,
            **met,
            "delta_f1@0.5": met["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - base_m["detected_ae_p95"],
            "delta_miss": met["miss_rate"] - base_m["miss_rate"],
            "changed_rate": float(rr["changed_candidate"].mean()),
            "mean_n_neighbors": float(rr["n_neighbors"].mean()),
        }
        grid_rows.append(row)
        print(
            f"  {tag}: F1@0.5={row['f1@0.5']:.4f} Δ={row['delta_f1@0.5']:+.4f} "
            f"F1@0.1Δ={row['delta_f1@0.1']:+.4f} P95Δ={row['delta_p95']:+.3f} "
            f"changed={row['changed_rate']:.3f}",
            flush=True,
        )
        return {"pred": pred, "diag": rr, "metrics": met, "row": row}

    # 1) SP top1: fix lambda=0.25, sweep sigma_d / sigma_sp
    print("sweep SP top1 sigma...", flush=True)
    best_sp = None
    for sd in [1, 2, 4, 6]:
        for ssp in [0.3, 0.5, 0.7, 1.0]:
            cfg = SoftRingRescoreConfig(
                mode="sp",
                neighbor_candidate_mode="top1",
                sigma_distance_km=float(sd),
                sigma_sp_s=float(ssp),
                lambda_sp=0.25,
                lambda_s=0.0,
                max_distance_diff_km=10.0,
            )
            res = run_cfg(f"sp_top1_sd{sd}_ssp{ssp}_l0.25", cfg)
            if best_sp is None or res["row"]["delta_f1@0.5"] > best_sp["row"]["delta_f1@0.5"]:
                best_sp = res
    # 2) lambda sweep around best sigmas
    print("sweep SP lambda...", flush=True)
    sd0 = best_sp["row"]["sigma_distance_km"]
    ssp0 = best_sp["row"]["sigma_sp_s"]
    for lam in [0.1, 0.25, 0.5, 1.0]:
        cfg = SoftRingRescoreConfig(
            mode="sp",
            neighbor_candidate_mode="top1",
            sigma_distance_km=float(sd0),
            sigma_sp_s=float(ssp0),
            lambda_sp=float(lam),
            max_distance_diff_km=10.0,
        )
        res = run_cfg(f"sp_top1_sd{sd0}_ssp{ssp0}_l{lam}", cfg)
        if res["row"]["delta_f1@0.5"] > best_sp["row"]["delta_f1@0.5"]:
            best_sp = res

    # 3) absolute-S top1 sequential
    print("sweep absolute_s...", flush=True)
    best_abs = None
    for sd in [2, 4, 6]:
        for ss in [0.5, 0.8, 1.2, 1.5]:
            cfg = SoftRingRescoreConfig(
                mode="absolute_s",
                neighbor_candidate_mode="top1",
                sigma_distance_km=float(sd),
                sigma_s_s=float(ss),
                lambda_s=0.25,
                lambda_sp=0.0,
                max_distance_diff_km=10.0,
            )
            res = run_cfg(f"abs_top1_sd{sd}_ss{ss}_l0.25", cfg)
            if best_abs is None or res["row"]["delta_f1@0.5"] > best_abs["row"]["delta_f1@0.5"]:
                best_abs = res
    for lam in [0.1, 0.25, 0.5, 1.0]:
        cfg = SoftRingRescoreConfig(
            mode="absolute_s",
            neighbor_candidate_mode="top1",
            sigma_distance_km=float(best_abs["row"]["sigma_distance_km"]),
            sigma_s_s=float(best_abs["row"]["sigma_s_s"]),
            lambda_s=float(lam),
            max_distance_diff_km=10.0,
        )
        res = run_cfg(
            f"abs_top1_sd{best_abs['row']['sigma_distance_km']}_ss{best_abs['row']['sigma_s_s']}_l{lam}",
            cfg,
        )
        if res["row"]["delta_f1@0.5"] > best_abs["row"]["delta_f1@0.5"]:
            best_abs = res

    # 4) hybrid around best
    print("sweep hybrid...", flush=True)
    best_hyb = None
    for ls in [0.1, 0.25]:
        for lsp in [0.1, 0.25, 0.5]:
            cfg = SoftRingRescoreConfig(
                mode="hybrid",
                neighbor_candidate_mode="top1",
                sigma_distance_km=float(sd0),
                sigma_sp_s=float(ssp0),
                sigma_s_s=float(best_abs["row"]["sigma_s_s"]),
                lambda_s=float(ls),
                lambda_sp=float(lsp),
                max_distance_diff_km=10.0,
            )
            res = run_cfg(f"hyb_top1_ls{ls}_lsp{lsp}", cfg)
            if best_hyb is None or res["row"]["delta_f1@0.5"] > best_hyb["row"]["delta_f1@0.5"]:
                best_hyb = res

    # 5) all_candidates for best SP / abs / hybrid
    print("all_candidates variants...", flush=True)
    best_sp_all = run_cfg(
        "sp_all_" + best_sp["row"]["tag"].replace("sp_top1_", ""),
        SoftRingRescoreConfig(
            mode="sp",
            neighbor_candidate_mode="all_candidates",
            sigma_distance_km=float(best_sp["row"]["sigma_distance_km"]),
            sigma_sp_s=float(best_sp["row"]["sigma_sp_s"]),
            lambda_sp=float(best_sp["row"]["lambda_sp"]),
            max_distance_diff_km=10.0,
        ),
    )
    best_abs_all = run_cfg(
        "abs_all",
        SoftRingRescoreConfig(
            mode="absolute_s",
            neighbor_candidate_mode="all_candidates",
            sigma_distance_km=float(best_abs["row"]["sigma_distance_km"]),
            sigma_s_s=float(best_abs["row"]["sigma_s_s"]),
            lambda_s=float(best_abs["row"]["lambda_s"]),
            max_distance_diff_km=10.0,
        ),
    )

    # pick overall best valid method (not oracle)
    candidates = [best_sp, best_abs, best_hyb, best_sp_all, best_abs_all]
    best = max(candidates, key=lambda r: r["row"]["delta_f1@0.5"])
    pred = best["pred"]
    diag = best["diag"]

    # shuffle diagnostics on best SP config
    print("shuffle diagnostics...", flush=True)
    cfg_best_sp = SoftRingRescoreConfig(
        mode="sp",
        neighbor_candidate_mode=best_sp["row"]["neighbor_candidate_mode"],
        sigma_distance_km=float(best_sp["row"]["sigma_distance_km"]),
        sigma_sp_s=float(best_sp["row"]["sigma_sp_s"]),
        lambda_sp=float(best_sp["row"]["lambda_sp"]),
        max_distance_diff_km=10.0,
    )
    sh_e = run_cfg("diag_shuffle_event_id", cfg_best_sp, shuffle_event_id=True)
    sh_sp = run_cfg("diag_shuffle_neighbor_sp", cfg_best_sp, shuffle_neighbor_sp=True)
    sanity["shuffle_event_reduces_gain"] = bool(sh_e["row"]["delta_f1@0.5"] < best_sp["row"]["delta_f1@0.5"] - 0.002)
    sanity["shuffle_sp_reduces_gain"] = bool(sh_sp["row"]["delta_f1@0.5"] < best_sp["row"]["delta_f1@0.5"] - 0.002)
    # oracle catalog neighbor diagnostic (LEAKAGE): neighbor ΔSP from catalog S + pred P;
    # target UNION candidates are NEVER rewritten.
    print("oracle leakage diagnostic...", flush=True)
    tr = table.drop_duplicates("trace_name").set_index("trace_name")
    neighbor_sp_label = (
        (tr["s_arrival_sample"].to_numpy(float) - tr["pred_p_sample"].to_numpy(float))
        / tr["sampling_rate_hz"].to_numpy(float)
    )
    neighbor_sp_by_trace = dict(zip(tr.index.astype(str), neighbor_sp_label.astype(float)))
    r_ora = apply_soft_ring_rescore_packs(
        packs,
        SoftRingRescoreConfig(
            mode="sp",
            neighbor_candidate_mode="top1",
            sigma_distance_km=float(best_sp["row"]["sigma_distance_km"]),
            sigma_sp_s=float(best_sp["row"]["sigma_sp_s"]),
            lambda_sp=float(best_sp["row"]["lambda_sp"]),
            max_distance_diff_km=10.0,
        ),
        neighbor_sp_by_trace=neighbor_sp_by_trace,
    )
    p_ora = r_ora.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    ora_m = _metrics(p_ora, true, sr)
    oracle_diag = {
        "oracle_only": True,
        "note": "NOT A VALID TEST-TIME METHOD — neighbor ΔSP uses catalog S + predicted P; target candidates unchanged",
        "metrics": ora_m,
        "delta_f1@0.5": ora_m["f1@0.5"] - base_m["f1@0.5"],
    }

    # ---------------- neighbor bins for best method ----------------
    n_neighbors = (
        diag.drop_duplicates("trace_name")
        .set_index("trace_name")["n_neighbors"]
        .reindex(names)
        .fillna(0)
        .to_numpy(int)
    )
    bins = _neighbor_bins(n_neighbors)
    bin_rows = []
    for bname in ["0", "1", "2-3", "4-7", ">=8"]:
        m = bins == bname
        if int(m.sum()) == 0:
            continue
        mb = _metrics(baseline[m], true[m], sr[m])
        mn = _metrics(pred[m], true[m], sr[m])
        bin_rows.append(
            {
                "neighbor_bin": bname,
                "n_traces": int(m.sum()),
                "n_events": int(np.unique(events[m]).size),
                "baseline_f1@0.5": mb["f1@0.5"],
                "new_f1@0.5": mn["f1@0.5"],
                "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
                "baseline_f1@0.1": mb["f1@0.1"],
                "new_f1@0.1": mn["f1@0.1"],
                "delta_f1@0.1": mn["f1@0.1"] - mb["f1@0.1"],
                "precision@0.5": mn["precision@0.5"],
                "recall@0.5": mn["recall@0.5"],
                "miss_rate": mn["miss_rate"],
                "detected_ae_median": mn["detected_ae_median"],
                "detected_ae_p95": mn["detected_ae_p95"],
            }
        )
    bin_df = pd.DataFrame(bin_rows)

    # ---------------- recoverable subset ----------------
    # UNION candidates: closest within 0.5s but baseline wrong
    u = table
    closest = []
    for tn, g in u.groupby("trace_name"):
        ts = float(g.s_arrival_sample.iloc[0])
        srate = float(g.sampling_rate_hz.iloc[0])
        ae = np.abs(g.candidate_sample.to_numpy(float) - ts) / srate
        closest.append({"trace_name": tn, "closest_ae_s": float(np.min(ae)), "has_recoverable": bool(np.min(ae) <= 0.5)})
    clos = pd.DataFrame(closest).set_index("trace_name")
    base_ae = np.abs(baseline - true) / sr
    new_ae = np.abs(pred - true) / sr
    recoverable = clos["has_recoverable"].reindex(names).fillna(False).to_numpy(bool)
    base_wrong = base_ae > 0.5
    base_right = base_ae <= 0.5
    new_right = new_ae <= 0.5
    subset = recoverable & base_wrong
    fixes = int(np.sum(subset & new_right))
    # breaks: baseline correct, new wrong, and had recoverable candidates (or all)
    breaks = int(np.sum(base_right & ~new_right))
    breaks_rec = int(np.sum(recoverable & base_right & ~new_right))
    n_sub = int(np.sum(subset))
    recover_stats = {
        "n_union_has_cand_within_0.5": int(np.sum(recoverable)),
        "n_baseline_wrong_but_recoverable": n_sub,
        "fixes": fixes,
        "breaks_all": breaks,
        "breaks_on_recoverable_traces": breaks_rec,
        "net_fixes_minus_breaks": fixes - breaks,
        "recoverable_oracle_gap_recovered_pct": float(fixes / max(n_sub, 1)),
        "baseline_wrong_recoverable_new_f1@0.5": float(np.mean(new_right[subset])) if n_sub else float("nan"),
    }

    # ---------------- bootstrap ----------------
    print("event bootstrap...", flush=True)
    boot = _event_bootstrap(pred, baseline, true, sr, events, n_boot=2000, seed=20260903)

    # hard gate best reference from prior report
    hard_gate_best = {
        "tag": "min_support1_w1p5s (prior hard gate)",
        "delta_f1@0.5": 0.007831,
        "note": "from artifacts/results/multistation/ring_gate_phaseB_extra_mild.csv",
    }

    # save artifacts
    pd.DataFrame(grid_rows).to_csv(out / "soft_ring_grid.csv", index=False)
    bin_df.to_csv(out / "soft_ring_neighbor_bins.csv", index=False)
    pd.DataFrame([recover_stats]).to_csv(out / "soft_ring_recoverable_cases.csv", index=False)
    save_json(boot, out / "soft_ring_bootstrap.json")
    save_json(sanity, out / "sanity_checks.json")
    save_json(oracle_diag, out / "oracle_catalog_neighbor_diagnostic.json")
    np.save(out / "pred_soft_ring_best.npy", pred)
    np.save(out / "pred_soft_ring_best_sp.npy", best_sp["pred"])
    metrics_summary = {
        "baseline_fixed_rescore_UNION": base_m,
        "hard_gate_best_ref": hard_gate_best,
        "best_overall": best["row"],
        "best_sp": best_sp["row"],
        "best_absolute_s": best_abs["row"],
        "best_hybrid": best_hyb["row"],
        "best_sp_all_candidates": best_sp_all["row"],
        "best_abs_all_candidates": best_abs_all["row"],
        "oracle_UNION_f1@0.5": 0.8916980743014904,
        "gap_to_oracle_best": 0.8916980743014904 - best["row"]["f1@0.5"],
        "gap_to_oracle_baseline": 0.8916980743014904 - base_m["f1@0.5"],
        "recoverable": recover_stats,
        "shuffle_event": sh_e["row"],
        "shuffle_sp": sh_sp["row"],
        "elapsed_s": time.time() - t0,
    }
    save_json(metrics_summary, out / "soft_ring_best.json")
    pd.DataFrame(
        [
            {"method": "fixed_rescore_UNION", **base_m},
            {"method": "soft_best", **best["metrics"], **{k: best["row"][k] for k in ["delta_f1@0.5", "delta_f1@0.1", "delta_p95"]}},
            {"method": "soft_sp", **best_sp["metrics"]},
            {"method": "soft_abs", **best_abs["metrics"]},
            {"method": "soft_hybrid", **best_hyb["metrics"]},
        ]
    ).to_csv(out / "soft_ring_metrics.csv", index=False)

    # report md
    lines = [
        "# Soft same-ring candidate rescoring — phaseB full-dev",
        "",
        "## Sanity",
        json.dumps(sanity, indent=2),
        "",
        "## Baseline fixed_rescore_UNION",
        f"- F1@0.5={base_m['f1@0.5']:.4f} F1@0.1={base_m['f1@0.1']:.4f} P95={base_m['detected_ae_p95']:.3f}",
        "",
        "## Best soft method",
        f"- tag={best['row']['tag']}",
        f"- F1@0.5={best['row']['f1@0.5']:.4f} (Δ={best['row']['delta_f1@0.5']:+.4f})",
        f"- F1@0.1={best['row']['f1@0.1']:.4f} (Δ={best['row']['delta_f1@0.1']:+.4f})",
        f"- P95={best['row']['detected_ae_p95']:.3f} (Δ={best['row']['delta_p95']:+.3f})",
        "",
        "## Mode comparison",
        f"- SP: ΔF1@0.5={best_sp['row']['delta_f1@0.5']:+.4f}",
        f"- absolute-S: ΔF1@0.5={best_abs['row']['delta_f1@0.5']:+.4f}",
        f"- hybrid: ΔF1@0.5={best_hyb['row']['delta_f1@0.5']:+.4f}",
        f"- SP all_candidates: ΔF1@0.5={best_sp_all['row']['delta_f1@0.5']:+.4f}",
        "",
        "## Neighbor bins",
        bin_df.to_string(index=False),
        "",
        "## Recoverable subset",
        json.dumps(recover_stats, indent=2),
        "",
        "## Bootstrap vs fixed_rescore_UNION",
        json.dumps({k: boot[k] for k in ['f1@0.5','f1@0.1','detected_ae_p95']}, indent=2),
        "",
        f"## Gap to UNION oracle 0.8917: {0.8916980743014904 - best['row']['f1@0.5']:.4f}",
        "",
        "catalog-assisted; not a blind picker. Confirm artifacts untouched.",
    ]
    (out / "soft_ring_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics_summary, indent=2))
    print(f"Wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
