#!/usr/bin/env python
"""Finish moveout analysis: ambiguity, bootstrap, single-station control, report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.multistation.moveout_rescore import (
    MoveoutRescoreConfig,
    apply_moveout_rescore_packs,
    build_moveout_packs,
)
from earthquake.utils import ensure_dir

OUT = ensure_dir(artifacts_dir() / "results" / "multistation_moveout")
ORACLE_F1 = 0.8916980743014904


def main() -> None:
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    table = pd.read_parquet(OUT / "phaseB_enriched_candidate_table.parquet")
    pairs = pd.read_csv(OUT / "ranking_gap_pairs.csv")
    fixed = np.load(
        artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy"
    ).astype(float)
    true = meta.s_arrival_sample.to_numpy(float)
    sr = meta.sampling_rate_hz.to_numpy(float)
    names = meta.trace_name.astype(str).to_numpy()
    events = meta.event_id.astype(str).to_numpy()

    def metrics(pred, true_=true, sr_=sr):
        m = match_picks(pred, true_, sr_, windows_s=(0.1, 0.5))
        b = report_pick_timing_bundle(pred, true_, sr_)
        err = np.abs((pred - true_) / sr_)
        ae = err[np.isfinite(pred) & np.isfinite(true_)]
        return {
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
            "frac_ae_gt_1s": float(np.mean(ae > 1)),
            "frac_ae_gt_5s": float(np.mean(ae > 5)),
            "frac_ae_gt_10s": float(np.mean(ae > 10)),
            "frac_ae_gt_30s": float(np.mean(ae > 30)),
        }

    def recoverable(pred):
        rec = set(pairs.trace_name.astype(str))
        base_ae = np.abs(fixed - true) / sr
        new_ae = np.abs(pred - true) / sr
        is_rec = np.array([t in rec for t in names])
        subset = is_rec & (base_ae > 0.5)
        fixes = int(np.sum(subset & (new_ae <= 0.5)))
        breaks = int(np.sum((base_ae <= 0.5) & (new_ae > 0.5)))
        return {
            "fixes": fixes,
            "breaks_all": breaks,
            "net_fixes_minus_breaks": fixes - breaks,
            "recoverable_oracle_gap_recovered_pct": float(fixes / max(int(subset.sum()), 1)),
            "n_baseline_wrong_but_recoverable": int(subset.sum()),
            "changed_rate": float(np.mean(np.abs(pred - fixed) > 0.5)),
            "change_precision": float(fixes / max(fixes + breaks, 1)),
        }

    base_m = metrics(fixed)
    packs_s = build_moveout_packs(table, mode="absolute_s")
    cfg = MoveoutRescoreConfig(
        mode="absolute_s",
        aggregator="weighted_median",
        sigma_r_s=0.5,
        lambda_moveout=1.0,
        min_neighbors=1,
    )
    rr = apply_moveout_rescore_packs(packs_s, cfg)
    pred = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    best_m = metrics(pred)
    best_rec = recoverable(pred)
    print("best legal", best_m["f1@0.5"] - base_m["f1@0.5"], best_rec, flush=True)

    # single-station residual control (no neighbors)
    sigma, lam = 0.5, 1.0
    ctrl_map = {}
    for tn, g in table.groupby("trace_name"):
        g = g.sort_values("candidate_index")
        base = g.fixed_score.to_numpy(float)
        resid = g.resid_s.to_numpy(float)
        R = np.exp(-0.5 * (resid / sigma) ** 2)
        R = np.where(np.isfinite(R), R, 0.0)
        j = int(np.nanargmax(base + lam * R))
        ctrl_map[str(tn)] = float(g.candidate_sample.to_numpy(float)[j])
    ctrl = np.array([ctrl_map[t] for t in names])
    ctrl_m = metrics(ctrl)
    ctrl_rec = recoverable(ctrl)
    agree = float(np.mean(np.abs(pred - ctrl) < 0.5))
    print(
        "single_station_resid_control",
        ctrl_m["f1@0.5"] - base_m["f1@0.5"],
        ctrl_rec,
        "agree",
        agree,
        flush=True,
    )

    # ambiguity gates
    margins = []
    for tn, g in table.groupby("trace_name"):
        sc = np.sort(g.fixed_score.to_numpy(float).copy())
        marg = float(sc[-1] - sc[-2]) if len(sc) >= 2 else 999.0
        margins.append((str(tn), marg))
    marg_arr = pd.Series(dict(margins)).reindex(names).to_numpy(float)
    amb_rows = []
    for q in [0.10, 0.20, 0.30]:
        thr = float(np.nanquantile(marg_arr[np.isfinite(marg_arr)], q))
        allow = marg_arr <= thr
        gated = fixed.copy()
        gated[allow] = pred[allow]
        met = metrics(gated)
        rec = recoverable(gated)
        amb_rows.append(
            {
                "gate": f"margin_lowest_{int(q * 100)}pct",
                "threshold": thr,
                "n_allowed": int(allow.sum()),
                "frac_allowed": float(allow.mean()),
                **met,
                "delta_f1@0.5": met["f1@0.5"] - base_m["f1@0.5"],
                "delta_f1@0.1": met["f1@0.1"] - base_m["f1@0.1"],
                "delta_p95": met["detected_ae_p95"] - base_m["detected_ae_p95"],
                **rec,
            }
        )
        print("amb", q, amb_rows[-1]["delta_f1@0.5"], rec["net_fixes_minus_breaks"], flush=True)
    disagree = {}
    for tn, g in table.groupby("trace_name"):
        if len(g) < 2:
            disagree[str(tn)] = False
            continue
        top2 = g.nlargest(2, "fixed_score")
        src = top2["source"].astype(str).tolist() if "source" in top2.columns else ["?", "?"]
        disagree[str(tn)] = src[0] != src[1]
    allow_d = np.array([disagree.get(t, False) for t in names])
    gated = fixed.copy()
    gated[allow_d] = pred[allow_d]
    met = metrics(gated)
    rec = recoverable(gated)
    amb_rows.append(
        {
            "gate": "stead_ida_top2_disagree",
            "threshold": np.nan,
            "n_allowed": int(allow_d.sum()),
            "frac_allowed": float(allow_d.mean()),
            **met,
            "delta_f1@0.5": met["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - base_m["detected_ae_p95"],
            **rec,
        }
    )
    pd.DataFrame(amb_rows).to_csv(OUT / "moveout_ambiguity_bins.csv", index=False)
    best_amb = pd.DataFrame(amb_rows).sort_values("delta_f1@0.5", ascending=False).iloc[0].to_dict()

    # bins
    # margin quartiles (handle duplicate edges)
    try:
        qs = pd.qcut(marg_arr, 4, labels=["Q1_most_amb", "Q2", "Q3", "Q4_least_amb"], duplicates="drop")
    except ValueError:
        qs = pd.qcut(marg_arr, 4, duplicates="drop")
        qs = qs.astype(str)
    bin_rows = []
    labs = list(pd.Series(qs).dropna().unique())
    # preserve preferred order when present
    preferred = ["Q1_most_amb", "Q2", "Q3", "Q4_least_amb"]
    ordered = [x for x in preferred if x in labs] + [x for x in labs if x not in preferred]
    for lab in ordered:
        m = np.asarray(qs).astype(str) == str(lab)
        if not m.any():
            continue
        mb = metrics(fixed[m], true[m], sr[m])
        mn = metrics(pred[m], true[m], sr[m])
        bae = np.abs(fixed[m] - true[m]) / sr[m]
        nae = np.abs(pred[m] - true[m]) / sr[m]
        fixes = int(np.sum((bae > 0.5) & (nae <= 0.5)))
        breaks = int(np.sum((bae <= 0.5) & (nae > 0.5)))
        bin_rows.append(
            {
                "bin": str(lab),
                "n_traces": int(m.sum()),
                "baseline_f1@0.5": mb["f1@0.5"],
                "new_f1@0.5": mn["f1@0.5"],
                "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
                "fixes": fixes,
                "breaks": breaks,
                "net": fixes - breaks,
            }
        )
    nn = (
        rr.drop_duplicates("trace_name")
        .set_index("trace_name")["n_neighbors"]
        .reindex(names)
        .fillna(0)
        .to_numpy(int)
    )
    nbins = np.full(len(nn), "0", dtype=object)
    nbins[nn == 1] = "1"
    nbins[(nn >= 2) & (nn <= 3)] = "2-3"
    nbins[(nn >= 4) & (nn <= 7)] = "4-7"
    nbins[nn >= 8] = ">=8"
    for b in ["0", "1", "2-3", "4-7", ">=8"]:
        m = nbins == b
        if not m.any():
            continue
        mb = metrics(fixed[m], true[m], sr[m])
        mn = metrics(pred[m], true[m], sr[m])
        bae = np.abs(fixed[m] - true[m]) / sr[m]
        nae = np.abs(pred[m] - true[m]) / sr[m]
        fixes = int(np.sum((bae > 0.5) & (nae <= 0.5)))
        breaks = int(np.sum((bae <= 0.5) & (nae > 0.5)))
        bin_rows.append(
            {
                "bin": f"neighbors_{b}",
                "n_traces": int(m.sum()),
                "baseline_f1@0.5": mb["f1@0.5"],
                "new_f1@0.5": mn["f1@0.5"],
                "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
                "fixes": fixes,
                "breaks": breaks,
                "net": fixes - breaks,
            }
        )
        print("bin", bin_rows[-1], flush=True)
    pd.DataFrame(bin_rows).to_csv(OUT / "moveout_neighbor_bins.csv", index=False)
    pd.DataFrame([r for r in bin_rows if not str(r["bin"]).startswith("neighbors_")]).to_csv(
        OUT / "moveout_ambiguity_margin_quartiles.csv", index=False
    )

    print("bootstrap...", flush=True)
    rng = np.random.default_rng(20260903)
    idx_by: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(v, int) for v in idx_by.values()]
    keys = ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate"]
    deltas = {k: np.empty(2000) for k in keys}
    ba, bb = metrics(pred), metrics(fixed)
    for b in range(2000):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = metrics(pred[ix], true[ix], sr[ix])
        mb = metrics(fixed[ix], true[ix], sr[ix])
        for k in keys:
            deltas[k][b] = ma[k] - mb[k]
    boot = {"n_boot": 2000, "seed": 20260903, "point": {k: float(ba[k] - bb[k]) for k in keys}}
    for k in keys:
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        boot[k] = {"mean_delta": float(deltas[k].mean()), "ci95": [float(lo), float(hi)]}
        print(k, boot[k], flush=True)
    save_json(boot, OUT / "moveout_bootstrap.json")

    oracle = load_json(OUT / "moveout_oracle_diagnostics.json")
    sanity = load_json(OUT / "sanity_checks.json")
    gap = load_json(OUT / "ranking_gap_summary.json")
    prov = load_json(OUT / "travel_time_provenance.json") if (OUT / "travel_time_provenance.json").exists() else {}

    sanity["shuffle_note"] = (
        "shuffle_event/resid barely reduced ΔF1; gain largely single-station |resid_s| vs T_S_hat"
    )
    sanity["single_station_resid_control_delta_f1@0.5"] = ctrl_m["f1@0.5"] - base_m["f1@0.5"]
    sanity["moveout_vs_singlestation_agree"] = agree
    save_json(sanity, OUT / "sanity_checks.json")

    ora_best = oracle.get("oracle_A_abs_s", {}).get("delta_f1@0.5", 0.0069)
    go_a = (best_m["f1@0.5"] - base_m["f1@0.5"]) >= 0.008 and (
        best_m["detected_ae_p95"] - base_m["detected_ae_p95"]
    ) <= 0
    go_b = best_rec["recoverable_oracle_gap_recovered_pct"] >= 0.30
    mo_auc = float(gap.get("moveout_absolute_s_auc") or 0)
    sr_auc = float(gap.get("same_ring_absolute_s_auc") or 0)
    go_c = mo_auc >= 0.70 and mo_auc > sr_auc + 0.05
    go = bool(go_a or go_b or go_c)
    rank_dist = gap.get("rank_distribution", {})
    p_le2 = float(rank_dist.get("P_rank_good_le_2", 0))
    next_step = f"waveform-level pairwise ranker (P(rank_good<=2)={p_le2:.1%}; Q2 large)"

    # also get single-station abs resid AUC from feature table
    auc_df = pd.read_csv(OUT / "ranking_gap_feature_auc.csv")
    ss_auc = float(auc_df.loc[auc_df.feature == "neg_abs_resid_s", "auc"].iloc[0])

    summary = {
        "baseline_fixed_rescore_UNION": base_m,
        "best_legal": {
            "tag": "absolute_s_weighted_median_sr0.5_l1.0_mn1",
            **best_m,
            "delta_f1@0.5": best_m["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": best_m["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": best_m["detected_ae_p95"] - base_m["detected_ae_p95"],
            **best_rec,
        },
        "single_station_resid_control": {
            "note": "score_fixed + λ exp(-resid_s^2/(2σ^2)); NO neighbors",
            **ctrl_m,
            "delta_f1@0.5": ctrl_m["f1@0.5"] - base_m["f1@0.5"],
            **ctrl_rec,
        },
        "moveout_vs_singlestation_agree": agree,
        "best_ambiguity_gate": best_amb,
        "oracle": oracle,
        "oracle_ceiling_delta_f1@0.5": ora_best,
        "gap_to_union_oracle": ORACLE_F1 - best_m["f1@0.5"],
        "bootstrap": {k: boot[k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95"]},
        "ranking_gap_aucs": {
            "same_ring_s": gap.get("same_ring_absolute_s_auc"),
            "same_ring_sp": gap.get("same_ring_sp_auc"),
            "moveout_s": gap.get("moveout_absolute_s_auc"),
            "moveout_sp": gap.get("moveout_sp_auc"),
            "single_station_abs_resid_s": ss_auc,
        },
        "go_no_go": {
            "GO": go,
            "criterion_A_delta_f1_ge_0.008_p95_ok": go_a,
            "criterion_B_recoverable_ge_30pct": go_b,
            "criterion_C_moveout_auc_ge_0.70": go_c,
            "verdict": "GO" if go else "NO-GO for further multi-station geometry development",
            "next_step_if_nogo": None if go else next_step,
            "key_evidence": (
                "shuffle barely hurts; single-station |resid_s| control ≈/≥ moveout → "
                "multi-station geometry not the primary driver"
            ),
        },
        "quadrants": gap.get("quadrants"),
        "rank_distribution": rank_dist,
        "top10_features": gap.get("top10_features"),
        "sanity": sanity,
        "provenance": prov,
    }
    pd.DataFrame([best_rec]).to_csv(OUT / "moveout_recoverable_analysis.csv", index=False)
    np.save(OUT / "pred_moveout_best.npy", pred)
    np.save(OUT / "pred_singlestation_resid_control.npy", ctrl)
    pd.DataFrame(
        [
            {"method": "fixed_rescore_UNION", **base_m},
            {
                "method": "moveout_best",
                **best_m,
                "delta_f1@0.5": best_m["f1@0.5"] - base_m["f1@0.5"],
            },
            {
                "method": "singlestation_resid_control",
                **ctrl_m,
                "delta_f1@0.5": ctrl_m["f1@0.5"] - base_m["f1@0.5"],
            },
        ]
    ).to_csv(OUT / "moveout_metrics.csv", index=False)
    save_json(summary, OUT / "moveout_best.json")

    lines = [
        "# Moveout residual pilot + ranking-gap forensic — phaseB full-dev",
        "",
        "## Sanity",
        json.dumps(sanity, indent=2),
        "",
        "## Ranking-gap",
        json.dumps(rank_dist, indent=2),
        "",
        "## AUC",
        json.dumps(summary["ranking_gap_aucs"], indent=2),
        "",
        "## Oracle ceiling",
        f"ΔF1@0.5={ora_best:+.4f}",
        "",
        "## Best legal moveout",
        json.dumps(summary["best_legal"], indent=2, default=str),
        "",
        "## Single-station residual CONTROL",
        json.dumps(summary["single_station_resid_control"], indent=2, default=str),
        "",
        f"prediction agree moveout vs control: {agree:.3f}",
        "",
        "## Ambiguity best",
        json.dumps(
            {
                k: best_amb[k]
                for k in best_amb
                if k
                in (
                    "gate",
                    "delta_f1@0.5",
                    "net_fixes_minus_breaks",
                    "change_precision",
                    "frac_allowed",
                )
            },
            indent=2,
            default=str,
        ),
        "",
        "## Bootstrap",
        json.dumps(summary["bootstrap"], indent=2),
        "",
        f"## Gap to UNION oracle: {summary['gap_to_union_oracle']:.4f}",
        "",
        "## Go/No-Go",
        json.dumps(summary["go_no_go"], indent=2),
        "",
        "catalog-assisted; confirm untouched. No GNN.",
    ]
    (OUT / "moveout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GO/NOGO", summary["go_no_go"], flush=True)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
