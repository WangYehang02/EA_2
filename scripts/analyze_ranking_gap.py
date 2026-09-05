#!/usr/bin/env python
"""Ranking-gap forensic audit on phaseB recoverable UNION candidates.

Ground-truth is used ONLY to define good/bad candidates for forensics.
Does not modify confirm / Stage-6 method lock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.multistation.moveout_rescore import (
    build_moveout_packs,
    per_candidate_moveout_delta,
)
from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "multistation_moveout"
ORACLE_F1 = 0.8916980743014904


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _auc_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    ok = np.isfinite(scores)
    if ok.sum() < 10 or len(np.unique(labels[ok])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels[ok], scores[ok]))
    except ValueError:
        return float("nan")


def load_enriched_table() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    table = pd.read_parquet(artifacts_dir() / "results" / "multistation_soft" / "phaseB_candidate_table.parquet")
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    hist = pd.read_parquet(
        artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"
    )
    hist = hist[hist["subset"] == "stage6_dev"].drop_duplicates("trace_name")
    fixed = np.load(
        artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy"
    ).astype(float)

    # merge source / ranks from union
    u_cols = [
        "trace_name",
        "candidate_index",
        "candidate_sample",
        "candidate_source_mask",
        "stead_probability",
        "ida_probability",
        "stead_rank",
        "ida_rank",
        "prob_margin_top12",
        "top1_p_prob_stead",
        "top1_p_prob_ida",
    ]
    u = union[u_cols].copy()
    t = table.merge(
        u,
        on=["trace_name", "candidate_index", "candidate_sample"],
        how="left",
        suffixes=("", "_u"),
    )
    # prefer union probs if present
    if "stead_probability_u" in t.columns:
        t["stead_probability"] = t["stead_probability"].fillna(t["stead_probability_u"])
        t["ida_probability"] = t["ida_probability"].fillna(t["ida_probability_u"])
    t = t.merge(
        hist[
            [
                "trace_name",
                "base_tau_s",
                "base_tau_p",
                "base_delta_sp",
                "history_count",
                "residual_s_mad",
                "residual_sp_mad",
            ]
        ],
        on="trace_name",
        how="left",
    )
    t = t.merge(
        meta[
            [
                "trace_name",
                "network",
                "station",
                "snr_db",
                "hyp_distance_km",
                "source_depth_km",
            ]
        ],
        on="trace_name",
        how="left",
    )

    # derived features
    sr = t["sampling_rate_hz"].to_numpy(float)
    t["tau_s"] = absolute_travel_time_s(
        origin_time=t["origin_time"],
        trace_start_time=t["trace_start_time"],
        sample=t["candidate_sample"].to_numpy(float),
        sampling_rate_hz=sr,
    )
    t["delta_sp"] = (t["candidate_sample"].to_numpy(float) - t["pred_p_sample"].to_numpy(float)) / sr
    t["resid_s"] = t["tau_s"] - t["base_tau_s"].to_numpy(float)
    t["resid_sp"] = t["delta_sp"] - t["base_delta_sp"].to_numpy(float)
    # history prior term (same as fixed_rescore)
    exp = t["expected_s_sample"].to_numpy(float)
    sig = t["history_sigma_samples"].to_numpy(float)
    hav = t["history_available"].to_numpy(float) > 0.5
    samp = t["candidate_sample"].to_numpy(float)
    hist_score = np.zeros(len(t))
    ok = hav & np.isfinite(samp) & np.isfinite(exp) & np.isfinite(sig) & (sig > 1e-6)
    hist_score[ok] = np.exp(-0.5 * ((samp[ok] - exp[ok]) / sig[ok]) ** 2)
    t["history_prior"] = hist_score
    t["abs_hist_dev_s"] = np.abs(samp - exp) / sr
    t["source"] = t["candidate_source_mask"].fillna("unknown").astype(str)
    t["ae_to_label_s"] = np.abs(samp - t["s_arrival_sample"].to_numpy(float)) / sr

    # per-trace ranks by fixed_score (1 = best)
    t["score_rank"] = (
        t.groupby("trace_name")["fixed_score"].rank(ascending=False, method="first").astype(int)
    )
    # margins vs top1
    top1 = (
        t.sort_values("fixed_score", ascending=False)
        .groupby("trace_name", sort=False)
        .first()[["fixed_score", "cand_prob"]]
        .rename(columns={"fixed_score": "top1_fixed_score", "cand_prob": "top1_cand_prob"})
    )
    t = t.merge(top1, left_on="trace_name", right_index=True, how="left")
    t["score_margin_to_top1"] = t["top1_fixed_score"] - t["fixed_score"]
    t["prob_margin_to_top1"] = t["top1_cand_prob"] - t["cand_prob"]

    return meta, t, fixed


def build_pairs(table: pd.DataFrame, fixed: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    names = meta["trace_name"].astype(str).to_numpy()
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    fixed_map = dict(zip(names, fixed))

    pairs = []
    for tn, g in table.groupby("trace_name"):
        g = g.sort_values("candidate_index")
        ts = float(g["s_arrival_sample"].iloc[0])
        srate = float(g["sampling_rate_hz"].iloc[0])
        ae = np.abs(g["candidate_sample"].to_numpy(float) - ts) / srate
        has_good = bool(np.min(ae) <= 0.5)
        # baseline selected
        if g["baseline_selected"].any():
            bad_row = g.loc[g["baseline_selected"]].iloc[0]
        else:
            bad_row = g.loc[g["fixed_score"].idxmax()]
        bad_ae = abs(float(bad_row["candidate_sample"]) - ts) / srate
        if not (has_good and bad_ae > 0.5):
            continue
        # good = closest within 0.5; tie-break higher fixed_score
        cand = g.copy()
        cand["_ae"] = ae
        good_pool = cand[cand["_ae"] <= 0.5].sort_values(["_ae", "fixed_score"], ascending=[True, False])
        good_row = good_pool.iloc[0]
        pairs.append(
            {
                "trace_name": str(tn),
                "event_id": str(g["event_id"].iloc[0]),
                "station": str(g["station"].iloc[0]) if "station" in g.columns else "",
                "network": str(g["network"].iloc[0]) if "network" in g.columns else "",
                "true_s": ts,
                "sampling_rate_hz": srate,
                "c_bad_sample": float(bad_row["candidate_sample"]),
                "c_good_sample": float(good_row["candidate_sample"]),
                "c_bad_index": int(bad_row["candidate_index"]),
                "c_good_index": int(good_row["candidate_index"]),
                "rank_bad": int(bad_row["score_rank"]),
                "rank_good": int(good_row["score_rank"]),
                "source_bad": str(bad_row.get("source", "unknown")),
                "source_good": str(good_row.get("source", "unknown")),
                "baseline_score_bad": float(bad_row["fixed_score"]),
                "baseline_score_good": float(good_row["fixed_score"]),
                "delta_score": float(good_row["fixed_score"] - bad_row["fixed_score"]),
                "error_bad": float(bad_ae),
                "error_good": float(good_row["_ae"]),
                "n_candidates": int(len(g)),
                "snr_db": float(g["snr_db"].iloc[0]) if "snr_db" in g.columns else np.nan,
                "distance_km": float(g["distance_km"].iloc[0]),
                "source_depth_km": float(g["source_depth_km"].iloc[0])
                if "source_depth_km" in g.columns
                else np.nan,
                "pred_p_sample": float(g["pred_p_sample"].iloc[0]),
                # feature snapshots
                "prob_bad": float(bad_row["cand_prob"]),
                "prob_good": float(good_row["cand_prob"]),
                "hist_prior_bad": float(bad_row["history_prior"]),
                "hist_prior_good": float(good_row["history_prior"]),
                "abs_hist_dev_bad": float(bad_row["abs_hist_dev_s"]),
                "abs_hist_dev_good": float(good_row["abs_hist_dev_s"]),
                "delta_sp_bad": float(bad_row["delta_sp"]),
                "delta_sp_good": float(good_row["delta_sp"]),
                "tau_s_bad": float(bad_row["tau_s"]),
                "tau_s_good": float(good_row["tau_s"]),
                "resid_s_bad": float(bad_row["resid_s"]),
                "resid_s_good": float(good_row["resid_s"]),
                "resid_sp_bad": float(bad_row["resid_sp"]),
                "resid_sp_good": float(good_row["resid_sp"]),
                "stead_prob_bad": float(bad_row["stead_probability"])
                if np.isfinite(bad_row.get("stead_probability", np.nan))
                else np.nan,
                "stead_prob_good": float(good_row["stead_probability"])
                if np.isfinite(good_row.get("stead_probability", np.nan))
                else np.nan,
                "ida_prob_bad": float(bad_row["ida_probability"])
                if np.isfinite(bad_row.get("ida_probability", np.nan))
                else np.nan,
                "ida_prob_good": float(good_row["ida_probability"])
                if np.isfinite(good_row.get("ida_probability", np.nan))
                else np.nan,
                "top1_top2_margin": float(
                    g.nlargest(2, "fixed_score")["fixed_score"].iloc[0]
                    - (g.nlargest(2, "fixed_score")["fixed_score"].iloc[1] if len(g) > 1 else g["fixed_score"].iloc[0])
                )
                if len(g) > 1
                else 0.0,
            }
        )
    return pd.DataFrame(pairs)


def add_multistation_features(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Same-ring consistency + moveout deltas for good/bad (diagnostic, uses pred neighbors)."""
    # neighbor counts by distance rings using baseline selected samples
    base = table[table.baseline_selected].drop_duplicates("trace_name").copy()
    base["trace_name"] = base["trace_name"].astype(str)
    base["tau_s"] = absolute_travel_time_s(
        origin_time=base["origin_time"],
        trace_start_time=base["trace_start_time"],
        sample=base["candidate_sample"].to_numpy(float),
        sampling_rate_hz=base["sampling_rate_hz"].to_numpy(float),
    )
    base["delta_sp"] = (
        base["candidate_sample"].to_numpy(float) - base["pred_p_sample"].to_numpy(float)
    ) / base["sampling_rate_hz"].to_numpy(float)

    # index by event
    by_event: dict[str, pd.DataFrame] = {str(e): g for e, g in base.groupby("event_id")}

    n2, n5, n10 = [], [], []
    ring_s_good, ring_s_bad, ring_sp_good, ring_sp_bad = [], [], [], []
    nn_resid, med_resid, mad_resid = [], [], []

    sigma_d, sigma_s, sigma_sp = 2.0, 0.8, 0.5
    for _, row in pairs.iterrows():
        eid = str(row.event_id)
        tn = str(row.trace_name)
        g = by_event.get(eid)
        if g is None or tn not in set(g.trace_name.astype(str)):
            n2.append(0)
            n5.append(0)
            n10.append(0)
            ring_s_good.append(np.nan)
            ring_s_bad.append(np.nan)
            ring_sp_good.append(np.nan)
            ring_sp_bad.append(np.nan)
            nn_resid.append(np.nan)
            med_resid.append(np.nan)
            mad_resid.append(np.nan)
            continue
        me = g[g.trace_name.astype(str) == tn].iloc[0]
        d0 = float(me.distance_km)
        dd = np.abs(g.distance_km.to_numpy(float) - d0)
        mask = g.trace_name.astype(str).to_numpy() != tn
        n2.append(int(np.sum(mask & (dd <= 2))))
        n5.append(int(np.sum(mask & (dd <= 5))))
        n10.append(int(np.sum(mask & (dd <= 10))))
        usable = mask & (dd <= 10) & np.isfinite(g.tau_s.to_numpy(float))
        if not usable.any():
            ring_s_good.append(np.nan)
            ring_s_bad.append(np.nan)
            ring_sp_good.append(np.nan)
            ring_sp_bad.append(np.nan)
            nn_resid.append(np.nan)
            med_resid.append(np.nan)
            mad_resid.append(np.nan)
            continue
        wd = np.exp(-0.5 * (dd[usable] / sigma_d) ** 2)
        tau_j = g.tau_s.to_numpy(float)[usable]
        sp_j = g.delta_sp.to_numpy(float)[usable]
        den = float(np.sum(wd)) + 1e-8

        def ring(val, neigh):
            wt = np.exp(-0.5 * ((val - neigh) / (sigma_s if neigh is tau_j else sigma_sp)) ** 2)
            # fix: use correct sigma
            return float(np.sum(wd * wt) / den)

        # absolute-S ring
        wt_sg = np.exp(-0.5 * ((float(row.tau_s_good) - tau_j) / sigma_s) ** 2)
        wt_sb = np.exp(-0.5 * ((float(row.tau_s_bad) - tau_j) / sigma_s) ** 2)
        ring_s_good.append(float(np.sum(wd * wt_sg) / den))
        ring_s_bad.append(float(np.sum(wd * wt_sb) / den))
        wt_pg = np.exp(-0.5 * ((float(row.delta_sp_good) - sp_j) / sigma_sp) ** 2)
        wt_pb = np.exp(-0.5 * ((float(row.delta_sp_bad) - sp_j) / sigma_sp) ** 2)
        ring_sp_good.append(float(np.sum(wd * wt_pg) / den))
        ring_sp_bad.append(float(np.sum(wd * wt_pb) / den))
        # neighbor residuals relative to bad (selected) absolute S
        dtau = np.abs(float(row.tau_s_bad) - tau_j)
        nn_resid.append(float(np.min(dtau)))
        med_resid.append(float(np.median(dtau)))
        mad_resid.append(float(np.median(np.abs(dtau - np.median(dtau)))))

    pairs = pairs.copy()
    pairs["n_neighbors_2km"] = n2
    pairs["n_neighbors_5km"] = n5
    pairs["n_neighbors_10km"] = n10
    pairs["ring_s_good"] = ring_s_good
    pairs["ring_s_bad"] = ring_s_bad
    pairs["ring_sp_good"] = ring_sp_good
    pairs["ring_sp_bad"] = ring_sp_bad
    pairs["nearest_neighbor_abs_dt_s"] = nn_resid
    pairs["median_neighbor_abs_dt_s"] = med_resid
    pairs["mad_neighbor_abs_dt_s"] = mad_resid
    return pairs


def add_moveout_pair_features(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    packs_s = build_moveout_packs(table, mode="absolute_s")
    packs_sp = build_moveout_packs(table, mode="sp")
    # map trace -> (event, index)
    loc_s = {}
    for eid, p in packs_s.items():
        for i, tn in enumerate(p.trace_names):
            loc_s[tn] = (eid, i, p)
    loc_sp = {}
    for eid, p in packs_sp.items():
        for i, tn in enumerate(p.trace_names):
            loc_sp[tn] = (eid, i, p)

    mo_s_g, mo_s_b, mo_sp_g, mo_sp_b = [], [], [], []
    for _, row in pairs.iterrows():
        tn = str(row.trace_name)
        # absolute-S
        if tn in loc_s:
            eid, i, p = loc_s[tn]
            deltas = per_candidate_moveout_delta(p, i, min_neighbors=1)
            # find candidate indices matching samples
            cs = p.cand_sample[i]
            ig = int(np.argmin(np.abs(cs - float(row.c_good_sample))))
            ib = int(np.argmin(np.abs(cs - float(row.c_bad_sample))))
            mo_s_g.append(float(deltas[ig]) if np.isfinite(deltas[ig]) else np.nan)
            mo_s_b.append(float(deltas[ib]) if np.isfinite(deltas[ib]) else np.nan)
        else:
            mo_s_g.append(np.nan)
            mo_s_b.append(np.nan)
        if tn in loc_sp:
            eid, i, p = loc_sp[tn]
            deltas = per_candidate_moveout_delta(p, i, min_neighbors=1)
            cs = p.cand_sample[i]
            ig = int(np.argmin(np.abs(cs - float(row.c_good_sample))))
            ib = int(np.argmin(np.abs(cs - float(row.c_bad_sample))))
            mo_sp_g.append(float(deltas[ig]) if np.isfinite(deltas[ig]) else np.nan)
            mo_sp_b.append(float(deltas[ib]) if np.isfinite(deltas[ib]) else np.nan)
        else:
            mo_sp_g.append(np.nan)
            mo_sp_b.append(np.nan)
    pairs = pairs.copy()
    pairs["moveout_abs_delta_good"] = mo_s_g
    pairs["moveout_abs_delta_bad"] = mo_s_b
    pairs["moveout_sp_delta_good"] = mo_sp_g
    pairs["moveout_sp_delta_bad"] = mo_sp_b
    return pairs


def feature_auc_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Expand pairs to good/bad rows; higher score => prefer good."""
    # features where higher is better for good
    higher_better = {
        "cand_prob": ("prob_good", "prob_bad"),
        "fixed_score": ("baseline_score_good", "baseline_score_bad"),
        "history_prior": ("hist_prior_good", "hist_prior_bad"),
        "ring_s": ("ring_s_good", "ring_s_bad"),
        "ring_sp": ("ring_sp_good", "ring_sp_bad"),
        "stead_prob": ("stead_prob_good", "stead_prob_bad"),
        "ida_prob": ("ida_prob_good", "ida_prob_bad"),
        "neg_abs_hist_dev": ("abs_hist_dev_good", "abs_hist_dev_bad"),  # invert
        "neg_moveout_abs": ("moveout_abs_delta_good", "moveout_abs_delta_bad"),
        "neg_moveout_sp": ("moveout_sp_delta_good", "moveout_sp_delta_bad"),
        "neg_abs_resid_s": ("resid_s_good", "resid_s_bad"),  # will use -abs later
        "neg_abs_resid_sp": ("resid_sp_good", "resid_sp_bad"),
    }
    rows = []
    for name, (cg, cb) in higher_better.items():
        g = pairs[cg].to_numpy(float)
        b = pairs[cb].to_numpy(float)
        if name.startswith("neg_abs_resid"):
            g, b = -np.abs(g), -np.abs(b)
        elif name.startswith("neg_"):
            g, b = -g, -b
        # expand
        scores = np.concatenate([g, b])
        labels = np.concatenate([np.ones(len(g)), np.zeros(len(b))])
        d = g - b  # positive => feature prefers good
        rows.append(
            {
                "feature": name,
                "auc": _auc_safe(scores, labels),
                "median_good": float(np.nanmedian(g)),
                "median_bad": float(np.nanmedian(b)),
                "median_d_good_minus_bad": float(np.nanmedian(d)),
                "mean_d": float(np.nanmean(d)),
                "iqr_d": float(np.nanpercentile(d, 75) - np.nanpercentile(d, 25))
                if np.isfinite(d).sum()
                else float("nan"),
                "p_d_gt_0": float(np.nanmean(d > 0)),
                "effect_size_cohen_d": _cohen_d(g, b),
                "n_pairs": int(len(pairs)),
            }
        )
    # source type: good is stead / ida / both
    for src in ["stead", "ida", "stead+ida"]:
        g = (pairs["source_good"] == src).astype(float)
        b = (pairs["source_bad"] == src).astype(float)
        scores = np.concatenate([g, b])
        labels = np.concatenate([np.ones(len(g)), np.zeros(len(b))])
        d = g - b
        rows.append(
            {
                "feature": f"source_is_{src}",
                "auc": _auc_safe(scores, labels),
                "median_good": float(np.nanmean(g)),
                "median_bad": float(np.nanmean(b)),
                "median_d_good_minus_bad": float(np.nanmedian(d)),
                "mean_d": float(np.nanmean(d)),
                "iqr_d": float("nan"),
                "p_d_gt_0": float(np.nanmean(d > 0)),
                "effect_size_cohen_d": _cohen_d(g, b),
                "n_pairs": int(len(pairs)),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def quadrant_analysis(table: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Q1–Q4 on top1 vs rank2 within 0.5s of label."""
    true = meta.set_index(meta.trace_name.astype(str))["s_arrival_sample"]
    sr = meta.set_index(meta.trace_name.astype(str))["sampling_rate_hz"]
    q = {"Q1_top1_ok_rank2_bad": 0, "Q2_top1_bad_rank2_ok": 0, "Q3_both_bad": 0, "Q4_both_ok": 0, "n_with_ge2": 0}
    q2_rows = []
    for tn, g in table.groupby("trace_name"):
        if len(g) < 2:
            continue
        q["n_with_ge2"] += 1
        g2 = g.sort_values("fixed_score", ascending=False).head(2)
        ts = float(true.loc[str(tn)])
        srate = float(sr.loc[str(tn)])
        ae = np.abs(g2["candidate_sample"].to_numpy(float) - ts) / srate
        t1_ok = ae[0] <= 0.5
        t2_ok = ae[1] <= 0.5
        if t1_ok and not t2_ok:
            q["Q1_top1_ok_rank2_bad"] += 1
        elif (not t1_ok) and t2_ok:
            q["Q2_top1_bad_rank2_ok"] += 1
            q2_rows.append(str(tn))
        elif (not t1_ok) and (not t2_ok):
            q["Q3_both_bad"] += 1
        else:
            q["Q4_both_ok"] += 1
    q["q2_trace_names"] = q2_rows
    return q


def main() -> None:
    out = ensure_dir(OUT)
    print("loading enriched table...", flush=True)
    meta, table, fixed = load_enriched_table()
    # cache enriched for moveout eval
    table.to_parquet(out / "phaseB_enriched_candidate_table.parquet", index=False)

    print("building recoverable pairs...", flush=True)
    pairs = build_pairs(table, fixed, meta)
    print(f"n_recoverable_pairs={len(pairs)}", flush=True)
    print("adding multistation features...", flush=True)
    pairs = add_multistation_features(pairs, table)
    print("adding moveout pair features...", flush=True)
    pairs = add_moveout_pair_features(pairs, table)
    pairs.to_csv(out / "ranking_gap_pairs.csv", index=False)

    # rank distribution
    rg = pairs["rank_good"].to_numpy(int)
    dist = {
        "n_recoverable": int(len(pairs)),
        "rank_good_eq_2": int(np.sum(rg == 2)),
        "rank_good_eq_3": int(np.sum(rg == 3)),
        "rank_good_eq_4": int(np.sum(rg == 4)),
        "rank_good_eq_5": int(np.sum(rg == 5)),
        "rank_good_gt_5": int(np.sum(rg > 5)),
        "P_rank_good_le_2": float(np.mean(rg <= 2)),
        "P_rank_good_le_3": float(np.mean(rg <= 3)),
        "P_rank_good_le_5": float(np.mean(rg <= 5)),
        "mean_rank_good": float(np.mean(rg)),
        "median_rank_good": float(np.median(rg)),
        "mean_n_candidates": float(pairs["n_candidates"].mean()),
        "median_delta_score_good_minus_bad": float(pairs["delta_score"].median()),
    }
    pd.DataFrame([dist]).to_csv(out / "ranking_gap_rank_distribution.csv", index=False)
    save_json(dist, out / "ranking_gap_rank_distribution.json")
    print(json.dumps(dist, indent=2), flush=True)

    # feature AUC
    print("feature AUC...", flush=True)
    auc_df = feature_auc_table(pairs)
    auc_df.to_csv(out / "ranking_gap_feature_auc.csv", index=False)
    # effects separately (same table)
    auc_df.to_csv(out / "ranking_gap_feature_effects.csv", index=False)
    print(auc_df.head(12).to_string(index=False), flush=True)

    # quadrants
    print("quadrants...", flush=True)
    q = quadrant_analysis(table, meta)
    q2_set = set(q.pop("q2_trace_names"))
    save_json(q, out / "ranking_gap_quadrants.json")
    print(q, flush=True)
    q2 = pairs[pairs.trace_name.isin(q2_set)].copy()
    q2.to_csv(out / "ranking_gap_q2_top1wrong_top2correct.csv", index=False)
    if len(q2):
        q2_auc = feature_auc_table(q2)
        q2_auc.to_csv(out / "ranking_gap_q2_feature_auc.csv", index=False)
        print("Q2 top features:\n", q2_auc.head(10).to_string(index=False), flush=True)

    # summary comparison same-ring vs moveout AUC
    key_auc = {
        "same_ring_absolute_s_auc": float(auc_df.loc[auc_df.feature == "ring_s", "auc"].iloc[0])
        if (auc_df.feature == "ring_s").any()
        else float("nan"),
        "same_ring_sp_auc": float(auc_df.loc[auc_df.feature == "ring_sp", "auc"].iloc[0])
        if (auc_df.feature == "ring_sp").any()
        else float("nan"),
        "moveout_absolute_s_auc": float(auc_df.loc[auc_df.feature == "neg_moveout_abs", "auc"].iloc[0])
        if (auc_df.feature == "neg_moveout_abs").any()
        else float("nan"),
        "moveout_sp_auc": float(auc_df.loc[auc_df.feature == "neg_moveout_sp", "auc"].iloc[0])
        if (auc_df.feature == "neg_moveout_sp").any()
        else float("nan"),
        "quadrants": q,
        "rank_distribution": dist,
        "top10_features": auc_df.head(10).to_dict(orient="records"),
        "note": "GT used only to define good/bad; AUCs are forensic diagnostics",
    }
    save_json(key_auc, out / "ranking_gap_summary.json")
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
