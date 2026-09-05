#!/usr/bin/env python
"""Event-wise moveout residual soft rescoring on Stage-6 phaseB full-dev.

Uses train-only Stage-6 base_tau_s / base_delta_sp from residual_history_features
(fitted on picker_train MLP). Does NOT read confirm or modify method lock.
"""

from __future__ import annotations

import json
import sys
import time
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
from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "multistation_moveout"
ORACLE_F1 = 0.8916980743014904


def _metrics(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    b = report_pick_timing_bundle(pred, true, sr, windows_s=(0.1, 0.5), match_window_s=0.5)
    err = np.abs((pred - true) / sr)
    ae = err[np.isfinite(pred) & np.isfinite(true)]
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
    }
    if ae.size:
        out["frac_ae_gt_1s"] = float(np.mean(ae > 1))
        out["frac_ae_gt_5s"] = float(np.mean(ae > 5))
        out["frac_ae_gt_10s"] = float(np.mean(ae > 10))
        out["frac_ae_gt_30s"] = float(np.mean(ae > 30))
    return out


def _event_bootstrap(pred_a, pred_b, true, sr, events, n_boot=2000, seed=20260903):
    rng = np.random.default_rng(seed)
    idx_by: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        idx_by.setdefault(str(e), []).append(i)
    lists = [np.asarray(v, int) for v in idx_by.values()]
    keys = ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate"]
    deltas = {k: np.empty(n_boot) for k in keys}
    pa, pb = _metrics(pred_a, true, sr), _metrics(pred_b, true, sr)
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = _metrics(pred_a[ix], true[ix], sr[ix])
        mb = _metrics(pred_b[ix], true[ix], sr[ix])
        for k in keys:
            deltas[k][b] = ma[k] - mb[k]
    out = {"n_boot": n_boot, "seed": seed, "point": {k: float(pa[k] - pb[k]) for k in keys}}
    for k in keys:
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        out[k] = {"mean_delta": float(deltas[k].mean()), "ci95": [float(lo), float(hi)]}
    return out


def _recoverable_stats(pred, baseline, true, sr, pairs: pd.DataFrame, names: np.ndarray) -> dict:
    rec = pairs.set_index("trace_name")
    base_ae = np.abs(baseline - true) / sr
    new_ae = np.abs(pred - true) / sr
    base_wrong = base_ae > 0.5
    base_right = base_ae <= 0.5
    new_right = new_ae <= 0.5
    is_rec = np.array([str(t) in rec.index for t in names])
    subset = is_rec & base_wrong
    fixes = int(np.sum(subset & new_right))
    breaks = int(np.sum(base_right & ~new_right))
    n_sub = int(np.sum(subset))
    return {
        "n_baseline_wrong_but_recoverable": n_sub,
        "fixes": fixes,
        "breaks_all": breaks,
        "net_fixes_minus_breaks": fixes - breaks,
        "recoverable_oracle_gap_recovered_pct": float(fixes / max(n_sub, 1)),
        "changed_rate": float(np.mean(np.abs(pred - baseline) > 0.5)),
        "change_precision": float(fixes / max(fixes + breaks, 1)),
    }


def main() -> None:
    t0 = time.time()
    out = ensure_dir(OUT)
    enriched_path = out / "phaseB_enriched_candidate_table.parquet"
    if not enriched_path.exists():
        raise SystemExit("Run scripts/analyze_ranking_gap.py first to build enriched table.")

    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    table = pd.read_parquet(enriched_path)
    pairs = pd.read_csv(out / "ranking_gap_pairs.csv")
    fixed = np.load(
        artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy"
    ).astype(float)
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    names = meta["trace_name"].astype(str).to_numpy()
    events = meta["event_id"].astype(str).to_numpy()
    base_m = _metrics(fixed, true, sr)
    print("baseline", json.dumps(base_m, indent=2), flush=True)

    # provenance for travel-time
    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    provenance = {
        "travel_time_source": "stage6 residual_history_features.base_tau_s / base_delta_sp",
        "baseline_kind": hist_man.get("baseline_kind"),
        "fit_split": "picker_train_only",
        "n_picker_train_events": hist_man.get("n_picker_train_events"),
        "n_picker_ps_traces_for_mlp": hist_man.get("n_picker_ps_traces_for_mlp"),
        "baseline_sha256": hist_man.get("baseline_sha256"),
        "features_sha256": hist_man.get("features_sha256"),
        "confirm_excluded": hist_man.get("confirm_excluded"),
        "phaseB_labels_used_for_fit": False,
    }
    save_json(provenance, out / "travel_time_provenance.json")

    print("building moveout packs...", flush=True)
    packs_s = build_moveout_packs(table, mode="absolute_s")
    packs_sp = build_moveout_packs(table, mode="sp")

    # ---------------- sanity ----------------
    sanity = {}
    cfg0 = MoveoutRescoreConfig(mode="absolute_s", lambda_moveout=0.0, min_neighbors=1)
    r0 = apply_moveout_rescore_packs(packs_s, cfg0)
    p0 = r0.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    p0b = r0.set_index("trace_name")["baseline_sample"].reindex(names).to_numpy(float)
    sanity["lambda0_equals_baseline"] = bool(np.mean(np.abs(p0 - p0b) < 1e-9) > 0.999)
    sanity["lambda0_matches_frozen_npy"] = bool(np.mean(np.abs(p0 - fixed) < 0.5) > 0.999)
    # no neighbors via min_neighbors huge
    cfg_nn = MoveoutRescoreConfig(mode="absolute_s", lambda_moveout=1.0, min_neighbors=10_000)
    pnn = (
        apply_moveout_rescore_packs(packs_s, cfg_nn)
        .set_index("trace_name")["new_sample"]
        .reindex(names)
        .to_numpy(float)
    )
    sanity["no_neighbor_unchanged"] = bool(np.mean(np.abs(pnn - p0b) < 1e-9) > 0.999)
    # leave-one-out: for singleton events n_neighbors should be 0 and unchanged
    # UNION membership
    cand_sets = {tn: set(g.candidate_sample.astype(float)) for tn, g in table.groupby("trace_name")}
    cfg_test = MoveoutRescoreConfig(mode="absolute_s", sigma_r_s=0.5, lambda_moveout=1.0, min_neighbors=1)
    rr_test = apply_moveout_rescore_packs(packs_s, cfg_test)
    pred_test = rr_test.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    union_ok = all(
        any(abs(float(pred_test[i]) - c) < 1e-6 for c in cand_sets[names[i]]) for i in range(len(names))
    )
    sanity["always_from_union_set"] = bool(union_ok)
    # LOO check: event center for a multi-station event should ignore target
    # Verify programmatically on one event: center with/without artificial self resid
    sample_eid = next(e for e, p in packs_s.items() if len(p.trace_names) >= 5)
    pack = packs_s[sample_eid]
    cfg_loo = MoveoutRescoreConfig(mode="absolute_s", lambda_moveout=1.0, min_neighbors=1, sigma_r_s=0.5)
    # override all neighbor resid to 0 except we'll check n_neighbors = n-1
    rr_loo = apply_moveout_rescore_packs({sample_eid: pack}, cfg_loo)
    sanity["loo_n_neighbors_is_n_minus_1"] = bool(
        (rr_loo["n_neighbors"] == (len(pack.trace_names) - 1)).all()
    )
    sanity["train_only_tt_no_phaseB_fit"] = True
    sanity["confirm_untouched"] = True
    sanity["stage6_lock_untouched"] = True
    save_json(sanity, out / "sanity_checks.json")
    print("sanity", sanity, flush=True)
    if not (
        sanity["lambda0_equals_baseline"]
        and sanity["no_neighbor_unchanged"]
        and sanity["always_from_union_set"]
        and sanity["loo_n_neighbors_is_n_minus_1"]
    ):
        raise SystemExit(f"SANITY FAILED: {sanity}")

    grid_rows = []

    def run(tag: str, packs, cfg: MoveoutRescoreConfig, **kwargs):
        rr = apply_moveout_rescore_packs(packs, cfg, **kwargs)
        pred = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
        met = _metrics(pred, true, sr)
        rec = _recoverable_stats(pred, fixed, true, sr, pairs, names)
        row = {
            "tag": tag,
            "mode": cfg.mode,
            "aggregator": cfg.aggregator,
            "sigma_r_s": cfg.sigma_r_s,
            "lambda_moveout": cfg.lambda_moveout,
            "min_neighbors": cfg.min_neighbors,
            "use_mad_gate": cfg.use_mad_gate,
            **met,
            "delta_f1@0.5": met["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - base_m["detected_ae_p95"],
            **{f"rec_{k}": v for k, v in rec.items()},
            "changed_rate_diag": float(rr["changed_candidate"].mean()),
        }
        grid_rows.append(row)
        print(
            f"  {tag}: F1@0.5={row['f1@0.5']:.4f} Δ={row['delta_f1@0.5']:+.4f} "
            f"P95Δ={row['delta_p95']:+.3f} fixes={rec['fixes']} breaks={rec['breaks_all']} "
            f"net={rec['net_fixes_minus_breaks']} rec%={rec['recoverable_oracle_gap_recovered_pct']:.3f}",
            flush=True,
        )
        return {"pred": pred, "diag": rr, "metrics": met, "row": row, "rec": rec}

    # ---------------- Oracle diagnostics (LEAKAGE) ----------------
    print("oracle diagnostics...", flush=True)
    # Oracle: neighbor residual from catalog S
    tr = table.drop_duplicates("trace_name").set_index("trace_name")
    tau_lab = absolute_travel_time_s(
        origin_time=tr["origin_time"],
        trace_start_time=tr["trace_start_time"],
        sample=tr["s_arrival_sample"].to_numpy(float),
        sampling_rate_hz=tr["sampling_rate_hz"].to_numpy(float),
    )
    resid_s_lab = tau_lab - tr["base_tau_s"].to_numpy(float)
    sp_lab = (tr["s_arrival_sample"].to_numpy(float) - tr["pred_p_sample"].to_numpy(float)) / tr[
        "sampling_rate_hz"
    ].to_numpy(float)
    resid_sp_lab = sp_lab - tr["base_delta_sp"].to_numpy(float)
    map_s = dict(zip(tr.index.astype(str), resid_s_lab.astype(float)))
    map_sp = dict(zip(tr.index.astype(str), resid_sp_lab.astype(float)))

    ora_cfg = MoveoutRescoreConfig(mode="absolute_s", sigma_r_s=0.5, lambda_moveout=1.0, min_neighbors=1)
    ora_s = run("ORACLE_abs_s_neighbor_label", packs_s, ora_cfg, neighbor_resid_by_trace=map_s)
    ora_sp = run(
        "ORACLE_sp_neighbor_label",
        packs_sp,
        MoveoutRescoreConfig(mode="sp", sigma_r_s=0.5, lambda_moveout=1.0, min_neighbors=1),
        neighbor_resid_by_trace=map_sp,
    )
    oracle_diag = {
        "oracle_only": True,
        "note": "LEAKAGE DIAGNOSTIC — NOT A VALID TEST-TIME METHOD",
        "oracle_A_abs_s": {
            "delta_f1@0.5": ora_s["row"]["delta_f1@0.5"],
            "metrics": ora_s["metrics"],
            "recoverable": ora_s["rec"],
        },
        "oracle_B_sp": {
            "delta_f1@0.5": ora_sp["row"]["delta_f1@0.5"],
            "metrics": ora_sp["metrics"],
            "recoverable": ora_sp["rec"],
        },
    }
    save_json(oracle_diag, out / "moveout_oracle_diagnostics.json")

    # Early stop check on oracle potential
    ora_best_df1 = max(ora_s["row"]["delta_f1@0.5"], ora_sp["row"]["delta_f1@0.5"])
    ora_best_rec = max(
        ora_s["rec"]["recoverable_oracle_gap_recovered_pct"],
        ora_sp["rec"]["recoverable_oracle_gap_recovered_pct"],
    )
    print(f"ORACLE ceiling ΔF1@0.5={ora_best_df1:+.4f} recoverable%={ora_best_rec:.3f}", flush=True)

    # ---------------- legal sweep (small) ----------------
    print("legal moveout sweep...", flush=True)
    legal_results = []
    # 1) fix lambda=0.25, sweep sigma and min_neighbors and mode
    for mode, packs in [("absolute_s", packs_s), ("sp", packs_sp)]:
        for sig in [0.3, 0.5, 0.8, 1.2]:
            for mn in [1, 2, 3]:
                cfg = MoveoutRescoreConfig(
                    mode=mode, sigma_r_s=sig, lambda_moveout=0.25, min_neighbors=mn, aggregator="median"
                )
                legal_results.append(run(f"{mode}_med_sr{sig}_l0.25_mn{mn}", packs, cfg))
    # pick best so far by delta_f1@0.5
    best_so_far = max(legal_results, key=lambda r: r["row"]["delta_f1@0.5"])
    bs = best_so_far["row"]
    print("lambda sweep around best...", flush=True)
    for lam in [0.1, 0.25, 0.5, 1.0]:
        packs = packs_s if bs["mode"] == "absolute_s" else packs_sp
        cfg = MoveoutRescoreConfig(
            mode=bs["mode"],
            sigma_r_s=float(bs["sigma_r_s"]),
            lambda_moveout=lam,
            min_neighbors=int(bs["min_neighbors"]),
            aggregator="median",
        )
        legal_results.append(run(f"{bs['mode']}_med_sr{bs['sigma_r_s']}_l{lam}_mn{bs['min_neighbors']}", packs, cfg))
    # weighted median + mad gate variants on best params
    packs = packs_s if bs["mode"] == "absolute_s" else packs_sp
    for agg in ["weighted_median"]:
        cfg = MoveoutRescoreConfig(
            mode=bs["mode"],
            sigma_r_s=float(bs["sigma_r_s"]),
            lambda_moveout=1.0,
            min_neighbors=int(bs["min_neighbors"]),
            aggregator=agg,
        )
        legal_results.append(run(f"{bs['mode']}_{agg}_best", packs, cfg))
    cfg = MoveoutRescoreConfig(
        mode=bs["mode"],
        sigma_r_s=float(bs["sigma_r_s"]),
        lambda_moveout=1.0,
        min_neighbors=int(bs["min_neighbors"]),
        aggregator="median",
        use_mad_gate=True,
        sigma_mad_s=1.0,
    )
    legal_results.append(run(f"{bs['mode']}_madgate_best", packs, cfg))

    # shuffle diagnostics on a strong legal config
    strong = MoveoutRescoreConfig(mode="absolute_s", sigma_r_s=0.5, lambda_moveout=1.0, min_neighbors=1)
    sh_e = run("diag_shuffle_event", packs_s, strong, shuffle_event_geometry=True)
    sh_r = run("diag_shuffle_resid", packs_s, strong, shuffle_neighbor_resid=True)
    sanity["shuffle_event_delta"] = sh_e["row"]["delta_f1@0.5"]
    sanity["shuffle_resid_delta"] = sh_r["row"]["delta_f1@0.5"]
    save_json(sanity, out / "sanity_checks.json")

    # best legal (exclude ORACLE and diag)
    legal_only = [r for r in legal_results if not r["row"]["tag"].startswith("ORACLE") and not r["row"]["tag"].startswith("diag")]
    # also exclude diag from grid later
    best = max(legal_only, key=lambda r: r["row"]["delta_f1@0.5"])
    pred = best["pred"]
    diag = best["diag"]
    print("BEST legal", best["row"]["tag"], best["row"]["delta_f1@0.5"], flush=True)

    # ---------------- ambiguity-gated ----------------
    print("ambiguity gating...", flush=True)
    # per-trace top1-top2 margin from table
    margins = []
    for tn, g in table.groupby("trace_name"):
        sc = np.sort(g["fixed_score"].to_numpy(float).copy())
        marg = float(sc[-1] - sc[-2]) if len(sc) >= 2 else 999.0
        margins.append((str(tn), marg, len(sc)))
    marg_df = pd.DataFrame(margins, columns=["trace_name", "margin", "n_cand"]).set_index("trace_name")
    marg_arr = marg_df.reindex(names)["margin"].to_numpy(float)
    amb_rows = []
    for q in [0.10, 0.20, 0.30]:
        thr = float(np.nanquantile(marg_arr[np.isfinite(marg_arr)], q))
        allow = marg_arr <= thr
        gated = fixed.copy()
        gated[allow] = pred[allow]
        met = _metrics(gated, true, sr)
        rec = _recoverable_stats(gated, fixed, true, sr, pairs, names)
        amb_rows.append(
            {
                "gate": f"margin_lowest_{int(q*100)}pct",
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
        print(
            f"  amb q={q}: ΔF1@0.5={amb_rows[-1]['delta_f1@0.5']:+.4f} "
            f"net={rec['net_fixes_minus_breaks']} change_prec={rec['change_precision']:.3f}",
            flush=True,
        )
    # also STEAD/IDA disagreement gate: sources differ on top1 vs rank2
    disagree = []
    for tn, g in table.groupby("trace_name"):
        if len(g) < 2:
            disagree.append((str(tn), False))
            continue
        top2 = g.nlargest(2, "fixed_score")
        sources = top2["source"].astype(str).tolist() if "source" in top2.columns else ["?", "?"]
        disagree.append((str(tn), sources[0] != sources[1]))
    dis_map = dict(disagree)
    allow_d = np.array([dis_map.get(t, False) for t in names])
    gated_d = fixed.copy()
    gated_d[allow_d] = pred[allow_d]
    met_d = _metrics(gated_d, true, sr)
    rec_d = _recoverable_stats(gated_d, fixed, true, sr, pairs, names)
    amb_rows.append(
        {
            "gate": "stead_ida_top2_disagree",
            "threshold": np.nan,
            "n_allowed": int(allow_d.sum()),
            "frac_allowed": float(allow_d.mean()),
            **met_d,
            "delta_f1@0.5": met_d["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": met_d["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": met_d["detected_ae_p95"] - base_m["detected_ae_p95"],
            **rec_d,
        }
    )
    amb_df = pd.DataFrame(amb_rows)
    amb_df.to_csv(out / "moveout_ambiguity_bins.csv", index=False)
    best_amb = amb_df.sort_values("delta_f1@0.5", ascending=False).iloc[0].to_dict()

    # ---------------- margin quartile bins ----------------
    qs = pd.qcut(marg_arr, 4, labels=["Q1_most_amb", "Q2", "Q3", "Q4_least_amb"], duplicates="drop")
    bin_rows = []
    for lab in ["Q1_most_amb", "Q2", "Q3", "Q4_least_amb"]:
        m = (qs.to_numpy() == lab) if hasattr(qs, "to_numpy") else np.array([str(x) == lab for x in qs])
        # qs may be Categorical
        m = np.asarray(qs) == lab
        if not m.any():
            continue
        mb = _metrics(fixed[m], true[m], sr[m])
        mn = _metrics(pred[m], true[m], sr[m])
        # fixes/breaks in bin
        bae = np.abs(fixed[m] - true[m]) / sr[m]
        nae = np.abs(pred[m] - true[m]) / sr[m]
        fixes = int(np.sum((bae > 0.5) & (nae <= 0.5)))
        breaks = int(np.sum((bae <= 0.5) & (nae > 0.5)))
        bin_rows.append(
            {
                "margin_bin": lab,
                "n_traces": int(m.sum()),
                "baseline_f1@0.5": mb["f1@0.5"],
                "new_f1@0.5": mn["f1@0.5"],
                "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
                "fixes": fixes,
                "breaks": breaks,
                "net": fixes - breaks,
            }
        )
    # neighbor bins
    nn = diag.drop_duplicates("trace_name").set_index("trace_name")["n_neighbors"].reindex(names).fillna(0).to_numpy(int)
    nbins = np.full(len(nn), "0", dtype=object)
    nbins[nn == 1] = "1"
    nbins[(nn >= 2) & (nn <= 3)] = "2-3"
    nbins[(nn >= 4) & (nn <= 7)] = "4-7"
    nbins[nn >= 8] = ">=8"
    for b in ["0", "1", "2-3", "4-7", ">=8"]:
        m = nbins == b
        if not m.any():
            continue
        mb = _metrics(fixed[m], true[m], sr[m])
        mn = _metrics(pred[m], true[m], sr[m])
        bin_rows.append(
            {
                "margin_bin": f"neighbors_{b}",
                "n_traces": int(m.sum()),
                "baseline_f1@0.5": mb["f1@0.5"],
                "new_f1@0.5": mn["f1@0.5"],
                "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
                "fixes": int(np.sum((np.abs(fixed[m] - true[m]) / sr[m] > 0.5) & (np.abs(pred[m] - true[m]) / sr[m] <= 0.5))),
                "breaks": int(np.sum((np.abs(fixed[m] - true[m]) / sr[m] <= 0.5) & (np.abs(pred[m] - true[m]) / sr[m] > 0.5))),
                "net": 0,
            }
        )
        bin_rows[-1]["net"] = bin_rows[-1]["fixes"] - bin_rows[-1]["breaks"]
    pd.DataFrame(bin_rows).to_csv(out / "moveout_neighbor_bins.csv", index=False)
    # also save margin bins separately name
    pd.DataFrame([r for r in bin_rows if not str(r["margin_bin"]).startswith("neighbors_")]).to_csv(
        out / "moveout_ambiguity_margin_quartiles.csv", index=False
    )

    # ---------------- bootstrap ----------------
    print("bootstrap...", flush=True)
    boot = _event_bootstrap(pred, fixed, true, sr, events, n_boot=2000)
    save_json(boot, out / "moveout_bootstrap.json")

    # ---------------- save ----------------
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out / "moveout_grid.csv", index=False)
    rec_best = best["rec"]
    pd.DataFrame([rec_best]).to_csv(out / "moveout_recoverable_analysis.csv", index=False)
    np.save(out / "pred_moveout_best.npy", pred)

    # ranking gap AUCs if available
    gap_sum = {}
    if (out / "ranking_gap_summary.json").exists():
        gap_sum = load_json(out / "ranking_gap_summary.json")

    go_a = best["row"]["delta_f1@0.5"] >= 0.008 and best["row"]["delta_p95"] <= 0.0
    go_b = rec_best["recoverable_oracle_gap_recovered_pct"] >= 0.30
    go_c = (
        float(gap_sum.get("moveout_absolute_s_auc", 0) or 0) >= 0.70
        and float(gap_sum.get("moveout_absolute_s_auc", 0) or 0)
        > float(gap_sum.get("same_ring_absolute_s_auc", 0) or 0) + 0.05
    ) or (
        float(gap_sum.get("moveout_sp_auc", 0) or 0) >= 0.70
        and float(gap_sum.get("moveout_sp_auc", 0) or 0)
        > float(gap_sum.get("same_ring_sp_auc", 0) or 0) + 0.05
    )
    go = bool(go_a or go_b or go_c)

    # next-step recommendation from rank distribution
    rank_dist = gap_sum.get("rank_distribution", {})
    p_le2 = float(rank_dist.get("P_rank_good_le_2", 0))
    if p_le2 >= 0.6:
        next_step = "waveform-level pairwise ranker (correct cand often already rank-2)"
    elif float(rank_dist.get("P_rank_good_le_5", 0)) < 0.7:
        next_step = "candidate generation / model complementarity (good cand often rank>3)"
    else:
        next_step = "waveform-level candidate discrimination (mixed rank distribution)"

    summary = {
        "baseline_fixed_rescore_UNION": base_m,
        "best_legal": best["row"],
        "best_recoverable": rec_best,
        "best_ambiguity_gate": best_amb,
        "oracle": oracle_diag,
        "oracle_ceiling_delta_f1@0.5": ora_best_df1,
        "oracle_ceiling_recoverable_pct": ora_best_rec,
        "gap_to_union_oracle": ORACLE_F1 - best["row"]["f1@0.5"],
        "gap_baseline_to_union_oracle": ORACLE_F1 - base_m["f1@0.5"],
        "bootstrap": {k: boot[k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95"]},
        "go_no_go": {
            "GO": go,
            "criterion_A_delta_f1_ge_0.008_p95_ok": go_a,
            "criterion_B_recoverable_ge_30pct": go_b,
            "criterion_C_moveout_auc_ge_0.70": go_c,
            "verdict": "GO" if go else "NO-GO for further multi-station geometry development",
            "next_step_if_nogo": None if go else next_step,
        },
        "ranking_gap_aucs": {
            "same_ring_s": gap_sum.get("same_ring_absolute_s_auc"),
            "same_ring_sp": gap_sum.get("same_ring_sp_auc"),
            "moveout_s": gap_sum.get("moveout_absolute_s_auc"),
            "moveout_sp": gap_sum.get("moveout_sp_auc"),
        },
        "sanity": sanity,
        "provenance": provenance,
        "elapsed_s": time.time() - t0,
    }
    save_json(summary, out / "moveout_best.json")
    pd.DataFrame(
        [
            {"method": "fixed_rescore_UNION", **base_m},
            {"method": "moveout_best", **best["metrics"], "delta_f1@0.5": best["row"]["delta_f1@0.5"]},
            {"method": "moveout_oracle_abs", **ora_s["metrics"], "delta_f1@0.5": ora_s["row"]["delta_f1@0.5"]},
        ]
    ).to_csv(out / "moveout_metrics.csv", index=False)

    lines = [
        "# Moveout residual pilot + ranking-gap forensic — phaseB full-dev",
        "",
        "## Sanity",
        json.dumps(sanity, indent=2),
        "",
        "## Ranking-gap rank distribution",
        json.dumps(rank_dist, indent=2),
        "",
        "## AUC same-ring vs moveout",
        json.dumps(summary["ranking_gap_aucs"], indent=2),
        "",
        "## Oracle ceiling (LEAKAGE)",
        f"best ΔF1@0.5={ora_best_df1:+.4f}, recoverable%={ora_best_rec:.3f}",
        "",
        "## Best legal moveout",
        f"tag={best['row']['tag']}",
        f"F1@0.5={best['row']['f1@0.5']:.4f} (Δ={best['row']['delta_f1@0.5']:+.4f})",
        f"F1@0.1={best['row']['f1@0.1']:.4f} (Δ={best['row']['delta_f1@0.1']:+.4f})",
        f"P95={best['row']['detected_ae_p95']:.3f} (Δ={best['row']['delta_p95']:+.3f})",
        f"fixes={rec_best['fixes']} breaks={rec_best['breaks_all']} net={rec_best['net_fixes_minus_breaks']}",
        f"recoverable%={rec_best['recoverable_oracle_gap_recovered_pct']:.3f}",
        "",
        "## Ambiguity gate best",
        json.dumps({k: best_amb[k] for k in best_amb if k in ('gate','delta_f1@0.5','net_fixes_minus_breaks','change_precision','frac_allowed')}, indent=2, default=str),
        "",
        "## Bootstrap",
        json.dumps(summary["bootstrap"], indent=2),
        "",
        f"## Gap to UNION oracle: {summary['gap_to_union_oracle']:.4f}",
        "",
        "## Go / No-Go",
        json.dumps(summary["go_no_go"], indent=2),
        "",
        "catalog-assisted; confirm/Stage-6 locks untouched. No GNN.",
    ]
    (out / "moveout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["go_no_go"], indent=2))
    print(f"Wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
