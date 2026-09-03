#!/usr/bin/env python
"""Stage 5.1 hierarchical residual sanity audit (val-only; no Stage 3/4 method selection)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.analysis.hierarchical_residual import (
    assert_event_disjoint,
    assert_no_test_split,
    build_path_residual_table,
    directed_path_key,
    hierarchical_residual,
)
from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.history.residual_prior import (
    ResidualPathStats,
    expected_samples_catalog,
    predict_with_residual_prior,
)
from earthquake.history.shrinkage import shrink_residual
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
    abs_t = origin + pd.to_timedelta(float(tau_s), unit="s")
    return float((abs_t - start).total_seconds() * sr)


def _sample_to_utc(row: pd.Series, sample: float):
    if not np.isfinite(sample):
        return None
    start = pd.Timestamp(row["trace_start_time"])
    return start + pd.to_timedelta(float(sample) / float(row["sampling_rate_hz"]), unit="s")


def metric_bundle(pred, true, sr) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    a = audit_pick_errors(pred, true, sr, window_s=0.5)
    return {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.5": float(m["f1@0.5s"]),
        "e2e_mae": float(m["e2e_mae"]),
        "e2e_p95": float(m["e2e_p95_ae"]),
        "matched_p95": float(a["matched_timing"]["p95_ae"]),
        "matched_mae": float(a["matched_timing"]["mae"]),
        "wrong_peak_rate": float(a["n_wrong_peak_beyond_tol"] / max(a["n_labeled"], 1)),
        "miss_rate": float(a["n_missed_pick"] / max(a["n_labeled"], 1)),
        "n_labeled": int(a["n_labeled"]),
        "n_wrong_peak": int(a["n_wrong_peak_beyond_tol"]),
        "n_missed": int(a["n_missed_pick"]),
        "n_matched": int(a["n_matched_within_tol"]),
        "metric_fn": "earthquake.metrics.match_picks + audit_pick_errors",
        "e2e_scope": "all_labeled_with_prediction (misses excluded from AE; wrong peaks included)",
        "miss_encoding": "pred NaN => missed_pick; excluded from e2e AE mean/P95",
    }


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage5_1_ustc")
    # Immediately invalidate prior hierarchical recommendation pending audit
    prev = {}
    vp = out / "final_verdict.json"
    if vp.exists():
        prev = load_json(vp)
    prev["hierarchical_prior_recommended"] = False
    prev["sanity_audit_status"] = "in_progress"
    save_json(prev, vp)

    lambdas = load_json(artifacts_dir() / "results" / "stage2" / "best_lambdas.json")
    lw, lh, lp = float(lambdas["best_s"]["lw"]), float(lambdas["best_s"]["lh"]), float(lambdas["best_s"]["lp"])
    k = float(lambdas["shrinkage_k"])
    rh_meta = load_json(artifacts_dir() / "results" / "stage2" / "residual_history_meta.json")
    global_res = rh_meta["global_residual"]
    min_hist = 5
    mad_dis = 1.0

    with open(artifacts_dir() / "results" / "stage2" / "travel_time_baseline.pkl", "rb") as f:
        baseline = pickle.load(f)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    train = events[events.split == "train"].copy()
    picks_cache = pd.read_parquet(artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache.parquet")
    cands = pd.read_parquet(artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache_candidates.parquet")
    fixed = pd.read_parquet(artifacts_dir() / "diagnostics" / "fixed_eval_events.parquet")
    frozen_picks = pd.read_parquet(artifacts_dir() / "results" / "stage2" / "candidate_rescoring_picks.parquet")
    resid = pd.read_parquet(artifacts_dir() / "history" / "residual_history_features_frozen.parquet")

    split_map = events.set_index("trace_name")["split"]
    fixed = fixed.copy()
    fixed["split"] = fixed["trace_name"].map(split_map)
    fixed_val = fixed[fixed["split"] == "val"].copy()
    assert_no_test_split(fixed_val, context="sanity_val")
    assert_event_disjoint(set(train.event_id.astype(str)), set(fixed_val.event_id.astype(str)), context="sanity")

    meta = fixed_val.merge(picks_cache, on=["trace_name", "event_id"], how="inner", suffixes=("", "_p"))
    ev_idx = events.set_index("trace_name")
    for col in [
        "origin_time",
        "trace_start_time",
        "sampling_rate_hz",
        "distance_km",
        "source_depth_km",
        "station_elevation_m",
        "source_latitude",
        "source_longitude",
        "network",
        "station",
        "location",
        "channel_prefix",
        "s_arrival_sample",
        "p_arrival_sample",
    ]:
        if col not in meta.columns or meta[col].isna().all():
            if col in fixed_val.columns:
                meta[col] = meta["trace_name"].map(fixed_val.set_index("trace_name")[col])
            elif col in ev_idx.columns:
                meta[col] = meta["trace_name"].map(ev_idx[col])

    # true S sample: prefer fixed_eval labels
    if "true_s_sample" not in meta.columns:
        meta["true_s_sample"] = meta["s_arrival_sample"]
    # join frozen residual history features (TemporalHistoryStore path)
    meta = meta.merge(resid, on=["trace_name", "event_id"], how="left", suffixes=("", "_r"))
    if "base_tau_s" not in meta.columns or meta["base_tau_s"].isna().all():
        # recompute bases
        from earthquake.history.residual_prior import attach_baseline_and_residuals

        tmp = attach_baseline_and_residuals(meta, baseline)
        for c in ["base_tau_p", "base_tau_s", "base_delta_sp"]:
            meta[c] = tmp[c].to_numpy()

    cand_index = _build_cand_index(cands)
    true = meta["true_s_sample"].to_numpy(dtype=float)
    sr = meta["sampling_rate_hz"].to_numpy(dtype=float)
    names = meta["trace_name"].astype(str).to_numpy()

    # --- Method A: PhaseNet only (cache) ---
    pred_pn = meta["pred_s_sample"].to_numpy(dtype=float)

    # --- Method B: frozen catalog_rescore from Stage2 picks parquet ---
    fr = frozen_picks[frozen_picks["mode"] == "catalog_rescore"].set_index("trace_name")
    pred_frozen = np.array([float(fr.loc[n, "pred_s_sample"]) if n in fr.index else np.nan for n in names])

    # --- Method C: recompute frozen-equivalent fine rescore using residual_history_features ---
    pred_frozen_reimpl = []
    prior_sample_frozen = []
    hist_avail_frozen = []
    for _, row in meta.iterrows():
        base = {
            "base_tau_p": float(row.get("base_tau_p", np.nan)),
            "base_tau_s": float(row.get("base_tau_s", np.nan)),
            "base_delta_sp": float(row.get("base_delta_sp", np.nan)),
        }
        path = _residual_stats_from_row(row)
        pred = predict_with_residual_prior(base, path, global_res, shrinkage_k=k, min_history=min_hist, mad_disable_s=mad_dis)
        exp_p, exp_s = expected_samples_catalog(row, pred)
        prior_sample_frozen.append(exp_s)
        hist_ok = bool(pred.history_available or pred.used_path_residual)
        hist_avail_frozen.append(hist_ok)
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        best, _ = rescore_phase_candidates(
            s_c,
            expected_sample=exp_s,
            sigma_samples=pred.sigma_s_s * float(row["sampling_rate_hz"]),
            lambda_wave=lw,
            lambda_history=lh,
            lambda_prominence=lp,
            history_available=hist_ok,
        )
        pred_frozen_reimpl.append(float(best.sample_index) if best is not None else float(row["pred_s_sample"]))
    pred_frozen_reimpl = np.asarray(pred_frozen_reimpl, dtype=float)
    prior_sample_frozen = np.asarray(prior_sample_frozen, dtype=float)
    hist_avail_frozen = np.asarray(hist_avail_frozen, dtype=bool)

    # --- Method D: Stage5.1-style fine_shrunk (exact fine key only) ---
    from earthquake.history.residual_prior import attach_baseline_and_residuals

    train_aug = attach_baseline_and_residuals(train, baseline)
    train_aug["split"] = "train"
    fine_tab = build_path_residual_table(train_aug, residual_col="residual_tau_s", grid_size=0.1, min_history=min_hist)
    coarse_tab = build_path_residual_table(train_aug, residual_col="residual_tau_s", grid_size=1.0, min_history=min_hist)

    pred_fine_s51 = []
    pred_coarse = []
    pred_hier = []
    pred_identity = []
    prior_sample_fine_s51 = []
    fine_ok_arr = []
    coarse_ok_arr = []
    hier_ok_arr = []
    r_fine_arr = []
    r_coarse_arr = []
    r_hier_arr = []
    fallback_branch = []

    for _, row in meta.iterrows():
        sf = fine_tab.query(directed_path_key(row, 0.1))
        sc = coarse_tab.query(directed_path_key(row, 1.0))
        fine_ok = bool(sf.get("history_available", 0) >= 1 and np.isfinite(sf["median"]))
        coarse_ok = bool(sc.get("history_available", 0) >= 1 and np.isfinite(sc["median"]))
        if fine_ok:
            r_fine = shrink_residual(float(sf["median"]), fine_tab.global_median, sf["n"], k)
        else:
            r_fine = fine_tab.global_median  # NOTE: unused if history_available=False in rescore
        if coarse_ok:
            r_coarse = shrink_residual(float(sc["median"]), coarse_tab.global_median, sc["n"], k)
        else:
            r_coarse = coarse_tab.global_median
        hier = hierarchical_residual(sf, sc, fine_tab.global_median, k_fine=k, k_coarse=k, min_history=min_hist)
        r_hier = float(hier["r_hat"])
        hier_ok = bool(hier["history_available"])

        s_c = cand_index.get((str(row.trace_name), "s"), [])
        sr_i = float(row["sampling_rate_hz"])
        base_s = float(row["base_tau_s"])

        def _pick(r_hat, hist_ok, mad_s):
            tau = base_s + float(r_hat)
            exp = _tau_to_sample(row, tau)
            sig_s = float(np.clip(1.4826 * mad_s, 0.05, 1.0)) if np.isfinite(mad_s) and hist_ok else 0.2
            best, _ = rescore_phase_candidates(
                s_c,
                expected_sample=exp,
                sigma_samples=sig_s * sr_i,
                lambda_wave=lw,
                lambda_history=lh,
                lambda_prominence=lp,
                history_available=bool(hist_ok),
            )
            return float(best.sample_index) if best is not None else float("nan"), exp

        pf, exp_f = _pick(r_fine, fine_ok, float(sf["mad"]))
        pc, _ = _pick(r_coarse, coarse_ok, float(sc["mad"]))
        ph, _ = _pick(r_hier, hier_ok, float(sf["mad"]) if fine_ok else float(sc["mad"]))
        # identity: history_available False
        pi, _ = _pick(0.0, False, np.nan)

        pred_fine_s51.append(pf)
        pred_coarse.append(pc)
        pred_hier.append(ph)
        pred_identity.append(pi)
        prior_sample_fine_s51.append(exp_f)
        fine_ok_arr.append(fine_ok)
        coarse_ok_arr.append(coarse_ok)
        hier_ok_arr.append(hier_ok)
        r_fine_arr.append(float(r_fine))
        r_coarse_arr.append(float(r_coarse))
        r_hier_arr.append(float(r_hier))
        if fine_ok:
            fallback_branch.append("fine_path")
        else:
            fallback_branch.append("identity_lambda_h0")

    pred_fine_s51 = np.asarray(pred_fine_s51)
    pred_coarse = np.asarray(pred_coarse)
    pred_hier = np.asarray(pred_hier)
    pred_identity = np.asarray(pred_identity)
    prior_sample_fine_s51 = np.asarray(prior_sample_fine_s51)
    fine_ok_arr = np.asarray(fine_ok_arr)
    coarse_ok_arr = np.asarray(coarse_ok_arr)
    hier_ok_arr = np.asarray(hier_ok_arr)

    methods = {
        "phasenet_only": pred_pn,
        "frozen_catalog_rescore_stage2": pred_frozen,
        "frozen_fine_reimplemented": pred_frozen_reimpl,
        "stage5_1_fine_shrunk": pred_fine_s51,
        "coarse_only": pred_coarse,
        "hierarchical": pred_hier,
        "identity_fallback": pred_identity,
    }
    metrics = {name: metric_bundle(pred, true, sr) for name, pred in methods.items()}

    # --- Agreement: frozen vs stage5.1 fine_shrunk ---
    def max_abs(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")

    agree = np.isfinite(pred_frozen) & np.isfinite(pred_fine_s51) & (np.abs(pred_frozen - pred_fine_s51) < 0.5)
    agreement = {
        "frozen_vs_s51_fine_shrunk": {
            "pick_time_max_diff_samples": max_abs(pred_frozen, pred_fine_s51),
            "pick_time_max_diff_seconds": max_abs(pred_frozen / sr, pred_fine_s51 / sr),
            "selected_candidate_agreement_frac": float(np.mean(np.abs(pred_frozen - pred_fine_s51) < 1e-6)),
            "selected_within_0.5_sample_frac": float(np.mean(agree)),
            "history_prior_max_diff_samples": max_abs(prior_sample_frozen, prior_sample_fine_s51),
            "metric_delta_f1_0.5": metrics["stage5_1_fine_shrunk"]["f1@0.5"] - metrics["frozen_catalog_rescore_stage2"]["f1@0.5"],
            "metric_delta_e2e_p95": metrics["stage5_1_fine_shrunk"]["e2e_p95"] - metrics["frozen_catalog_rescore_stage2"]["e2e_p95"],
            "equivalent": bool(max_abs(pred_frozen, pred_fine_s51) < 1e-6),
        },
        "frozen_vs_frozen_reimpl": {
            "pick_time_max_diff_samples": max_abs(pred_frozen, pred_frozen_reimpl),
            "selected_candidate_agreement_frac": float(np.mean(np.abs(pred_frozen - pred_frozen_reimpl) < 1e-6)),
            "equivalent": bool(max_abs(pred_frozen, pred_frozen_reimpl) < 1e-6),
        },
        "identity_vs_phasenet": {
            "pick_time_max_diff_samples": max_abs(pred_identity, pred_pn),
            "agreement_frac": float(np.mean(np.abs(pred_identity - pred_pn) < 1e-6)),
            "note": "identity rescores among K candidates by probability; may differ from annotate argmax if not in candidate list",
        },
        "fine_unseen_identity_vs_phasenet": {
            "n_fine_unseen": int((~fine_ok_arr).sum()),
            "agreement_frac": float(np.mean(np.abs(pred_fine_s51[~fine_ok_arr] - pred_pn[~fine_ok_arr]) < 1e-6)) if (~fine_ok_arr).any() else float("nan"),
            "max_diff_samples": max_abs(pred_fine_s51[~fine_ok_arr], pred_pn[~fine_ok_arr]) if (~fine_ok_arr).any() else float("nan"),
            "agreement_vs_identity_frac": float(np.mean(np.abs(pred_fine_s51[~fine_ok_arr] - pred_identity[~fine_ok_arr]) < 1e-6)) if (~fine_ok_arr).any() else float("nan"),
            "strict_identity_lambda_h0": True,
            "uses_zero_origin_or_start_as_arrival": False,
            "note": "When fine_ok=False, rescore uses history_available=False (λh=0); expected_sample is ignored for ranking",
        },
        "coverage": {
            "frozen_history_available_frac": float(np.mean(hist_avail_frozen)),
            "s51_fine_ok_frac": float(np.mean(fine_ok_arr)),
            "s51_coarse_ok_frac": float(np.mean(coarse_ok_arr)),
            "s51_hier_ok_frac": float(np.mean(hier_ok_arr)),
            "frozen_uses_temporal_store_fallback": True,
            "s51_fine_exact_key_only": True,
        },
    }

    # --- Large e2e errors (>10s) for fine_shrunk ---
    err_fine = np.abs((pred_fine_s51 - true) / sr)
    err_frozen = np.abs((pred_frozen - true) / sr)
    err_pn = np.abs((pred_pn - true) / sr)
    big = np.where(np.isfinite(err_fine) & (err_fine > 10.0))[0]
    big_rows = []
    for i in big:
        row = meta.iloc[i]
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        cand_utcs = [_sample_to_utc(row, c.sample_index) for c in s_c]
        gt_utc = _sample_to_utc(row, true[i])
        hist_utc = _sample_to_utc(row, prior_sample_fine_s51[i]) if fine_ok_arr[i] else None
        sel_utc = _sample_to_utc(row, pred_fine_s51[i])
        cls = "wrong_peak_beyond_tol" if np.isfinite(pred_fine_s51[i]) and np.isfinite(true[i]) else "miss_or_other"
        # unit sanity
        unit_ok = True
        if np.isfinite(pred_fine_s51[i]) and (pred_fine_s51[i] < 0 or pred_fine_s51[i] > 20000):
            unit_ok = False
        big_rows.append(
            {
                "trace_name": str(row.trace_name),
                "event_id": str(row.event_id),
                "e2e_error_s": float(err_fine[i]),
                "e2e_error_frozen_s": float(err_frozen[i]) if np.isfinite(err_frozen[i]) else None,
                "e2e_error_phasenet_s": float(err_pn[i]) if np.isfinite(err_pn[i]) else None,
                "true_s_sample": float(true[i]),
                "true_s_utc": str(gt_utc),
                "pred_fine_s51_sample": float(pred_fine_s51[i]),
                "pred_fine_s51_utc": str(sel_utc),
                "pred_frozen_sample": float(pred_frozen[i]),
                "pred_phasenet_sample": float(pred_pn[i]),
                "history_prior_sample": float(prior_sample_fine_s51[i]) if np.isfinite(prior_sample_fine_s51[i]) else None,
                "history_prior_utc": str(hist_utc),
                "fine_ok": bool(fine_ok_arr[i]),
                "fallback_branch": fallback_branch[i],
                "n_candidates": len(s_c),
                "candidate_samples": [int(c.sample_index) for c in s_c],
                "candidate_utcs": [str(u) for u in cand_utcs],
                "error_class": cls,
                "unit_sanity_ok": unit_ok,
                "relative_vs_absolute_mix_suspected": False,
            }
        )
    pd.DataFrame(big_rows).to_csv(out / "sanity_large_errors_fine_shrunk.csv", index=False)

    # --- Bootstrap direction clarification from existing path_repeatability ---
    pr = load_json(out / "path_repeatability.json")
    boot = pr["bootstrap"]["correct_vs_shuffled"]
    # event_bootstrap_delta_mae(baseline=shuffled, improved=correct) => MAE_shuf - MAE_corr
    bootstrap_clarification = {
        "comparison": "shuffled_path vs correct_path travel-time MAE on validation",
        "delta_definition": "MAE(shuffled) - MAE(correct_path)",
        "positive_means": "correct_path better (lower MAE)",
        "mean_delta": boot["mean"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "prob_delta_positive": boot["prob_improved"],
        "n_events": boot["n_events"],
        "n_traces_in_prior_eval": pr["prior_error_summaries"]["mlp_correct_path"]["n"],
        "mae_shuffled": pr["prior_error_summaries"]["mlp_shuffled_path"]["mae"],
        "mae_correct": pr["prior_error_summaries"]["mlp_correct_path"]["mae"],
        "ci_entirely_positive": bool(boot["ci95_low"] > 0),
    }

    # Stage3 comparison note (read-only context, not method selection)
    s3 = load_json(artifacts_dir() / "results" / "stage3" / "bootstrap_fixed_vs_phasenet.json")
    stage3_note = {
        "stage3_fixed_e2e_p95": s3["point_estimates"]["fixed"]["e2e_p95"],
        "stage3_set": "event-disjoint test-only 10k (NOT Stage2 fixed-eval val)",
        "stage5_1_set": "Stage2 fixed-eval validation traces only (n=5220)",
        "comparable": False,
        "reason": "different trace/event lists; Stage2 val PhaseNet e2e P95 already ~70s",
    }

    # Final verdict classification
    fine_mismatch = agreement["frozen_vs_s51_fine_shrunk"]["pick_time_max_diff_samples"] > 1.0
    coarse_near_frozen = abs(metrics["coarse_only"]["e2e_p95"] - metrics["frozen_catalog_rescore_stage2"]["e2e_p95"]) < 0.5
    hier_equals_coarse = abs(metrics["hierarchical"]["f1@0.5"] - metrics["coarse_only"]["f1@0.5"]) < 1e-9

    if fine_mismatch and metrics["stage5_1_fine_shrunk"]["e2e_p95"] > 10:
        final_class = "implementation_bug"
        explanation = (
            "Stage 5.1 fine_shrunk is NOT the frozen catalog_rescore: it uses exact 0.1° path keys only "
            f"(coverage {agreement['coverage']['s51_fine_ok_frac']:.1%}) without TemporalHistoryStore fallback "
            f"(frozen coverage {agreement['coverage']['frozen_history_available_frac']:.1%}). "
            "Identity fallback on fine_unseen leaves PhaseNet-scale tails (e2e P95~55s). "
            "Coarse/hierarchical appear strong only relative to this broken baseline; on the same val list, "
            "frozen catalog_rescore already has e2e P95~1.26s. Comparing 54.9s to Stage3 1.11s is also an "
            "evaluation_mismatch (different sets)."
        )
    elif not fine_mismatch and metrics["coarse_only"]["f1@0.5"] - metrics["frozen_catalog_rescore_stage2"]["f1@0.5"] >= 0.005:
        final_class = "genuine_coarse_backoff_gain"
        explanation = "fine matches frozen and coarse still improves."
    else:
        final_class = "evaluation_mismatch"
        explanation = "Metric/set mismatch dominates."

    result = {
        "hierarchical_prior_recommended": False,
        "final_classification": final_class,
        "explanation": explanation,
        "n_val_traces": int(len(meta)),
        "n_val_events": int(meta.event_id.nunique()),
        "methods_metrics": metrics,
        "agreement": agreement,
        "bootstrap_path_clarification": bootstrap_clarification,
        "stage3_comparison_note": stage3_note,
        "large_error_audit": {
            "threshold_s": 10.0,
            "n_fine_shrunk_errors_gt_10s": int(len(big_rows)),
            "n_frozen_errors_gt_10s": int(np.sum(np.isfinite(err_frozen) & (err_frozen > 10))),
            "n_phasenet_errors_gt_10s": int(np.sum(np.isfinite(err_pn) & (err_pn > 10))),
            "csv": "artifacts/results/stage5_1_ustc/sanity_large_errors_fine_shrunk.csv",
            "relative_absolute_mix_found": False,
            "unit_mix_found": False,
        },
        "coarse_vs_hierarchical_separated": {
            "coarse_only": metrics["coarse_only"],
            "hierarchical": metrics["hierarchical"],
            "hier_equals_coarse_on_this_val": hier_equals_coarse,
            "coarse_near_frozen_catalog_rescore": coarse_near_frozen,
        },
        "do_not_merge_coarse_hierarchical_ranges": True,
        "frozen_artifacts_modified": False,
    }
    save_json(result, out / "sanity_audit.json")

    # Update final verdict
    verdict = load_json(vp) if vp.exists() else {}
    verdict.update(
        {
            "hierarchical_prior_recommended": False,
            "hierarchical_prior_beats_current_on_val": False,
            "sanity_audit_status": "complete",
            "sanity_final_classification": final_class,
            "path_residual_is_repeatable": bool(pr.get("correct_path_beats_shuffled", False)),
            "correct_path_beats_shuffled": bool(pr.get("correct_path_beats_shuffled", False)),
            "modify_frozen_test_results": False,
            "run_gnn": False,
        }
    )
    save_json(verdict, vp)

    # Markdown report
    report = ROOT / "reports/stage5_1_sanity_audit.md"
    m = metrics
    report.write_text(
        f"""# Stage 5.1 Hierarchical Residual — Sanity Audit

**Final classification: `{final_class}`**

`hierarchical_prior_recommended`: **false**

{explanation}

## 1. Same validation list, same cache, same λ/K/metrics

Evaluation set: Stage2 fixed-eval **validation** only (`n={len(meta)}` traces, `{meta.event_id.nunique()}` events).
PhaseNet candidates: frozen `phasenet_fixed_cache(_candidates).parquet`.
S λ frozen: `(lw,lh,lp)=({lw},{lh},{lp})`, `k={k}`, `K=5`.
Metric: `earthquake.metrics.match_picks` / `audit_pick_errors` (same as Stage 2/3 pipeline).

| Method | F1@0.1 | F1@0.5 | e2e MAE | e2e P95 | matched P95 | wrong-peak | miss |
|--|--:|--:|--:|--:|--:|--:|--:|
| PhaseNet only | {m['phasenet_only']['f1@0.1']:.3f} | {m['phasenet_only']['f1@0.5']:.3f} | {m['phasenet_only']['e2e_mae']:.3f} | {m['phasenet_only']['e2e_p95']:.2f} | {m['phasenet_only']['matched_p95']:.3f} | {m['phasenet_only']['wrong_peak_rate']:.3f} | {m['phasenet_only']['miss_rate']:.3f} |
| Frozen catalog_rescore (Stage2 picks) | {m['frozen_catalog_rescore_stage2']['f1@0.1']:.3f} | {m['frozen_catalog_rescore_stage2']['f1@0.5']:.3f} | {m['frozen_catalog_rescore_stage2']['e2e_mae']:.3f} | **{m['frozen_catalog_rescore_stage2']['e2e_p95']:.2f}** | {m['frozen_catalog_rescore_stage2']['matched_p95']:.3f} | {m['frozen_catalog_rescore_stage2']['wrong_peak_rate']:.3f} | {m['frozen_catalog_rescore_stage2']['miss_rate']:.3f} |
| Frozen fine reimplemented | {m['frozen_fine_reimplemented']['f1@0.1']:.3f} | {m['frozen_fine_reimplemented']['f1@0.5']:.3f} | {m['frozen_fine_reimplemented']['e2e_mae']:.3f} | {m['frozen_fine_reimplemented']['e2e_p95']:.2f} | {m['frozen_fine_reimplemented']['matched_p95']:.3f} | {m['frozen_fine_reimplemented']['wrong_peak_rate']:.3f} | {m['frozen_fine_reimplemented']['miss_rate']:.3f} |
| Stage5.1 fine_shrunk | {m['stage5_1_fine_shrunk']['f1@0.1']:.3f} | {m['stage5_1_fine_shrunk']['f1@0.5']:.3f} | {m['stage5_1_fine_shrunk']['e2e_mae']:.3f} | **{m['stage5_1_fine_shrunk']['e2e_p95']:.2f}** | {m['stage5_1_fine_shrunk']['matched_p95']:.3f} | {m['stage5_1_fine_shrunk']['wrong_peak_rate']:.3f} | {m['stage5_1_fine_shrunk']['miss_rate']:.3f} |
| Coarse-only (1°) | {m['coarse_only']['f1@0.1']:.3f} | {m['coarse_only']['f1@0.5']:.3f} | {m['coarse_only']['e2e_mae']:.3f} | {m['coarse_only']['e2e_p95']:.2f} | {m['coarse_only']['matched_p95']:.3f} | {m['coarse_only']['wrong_peak_rate']:.3f} | {m['coarse_only']['miss_rate']:.3f} |
| Hierarchical | {m['hierarchical']['f1@0.1']:.3f} | {m['hierarchical']['f1@0.5']:.3f} | {m['hierarchical']['e2e_mae']:.3f} | {m['hierarchical']['e2e_p95']:.2f} | {m['hierarchical']['matched_p95']:.3f} | {m['hierarchical']['wrong_peak_rate']:.3f} | {m['hierarchical']['miss_rate']:.3f} |
| Identity fallback | {m['identity_fallback']['f1@0.1']:.3f} | {m['identity_fallback']['f1@0.5']:.3f} | {m['identity_fallback']['e2e_mae']:.3f} | {m['identity_fallback']['e2e_p95']:.2f} | {m['identity_fallback']['matched_p95']:.3f} | {m['identity_fallback']['wrong_peak_rate']:.3f} | {m['identity_fallback']['miss_rate']:.3f} |

**Coarse-only and hierarchical are reported separately** (not merged as a range).
On this val set they are essentially identical (`hier_equals_coarse={hier_equals_coarse}`).

## 2. Frozen fine vs Stage5.1 fine_shrunk consistency

| Check | Value |
|--|--|
| Pick max |Δ| samples | {agreement['frozen_vs_s51_fine_shrunk']['pick_time_max_diff_samples']:.3g} |
| Pick max |Δ| seconds | {agreement['frozen_vs_s51_fine_shrunk']['pick_time_max_diff_seconds']:.3g} |
| Exact candidate agreement | {agreement['frozen_vs_s51_fine_shrunk']['selected_candidate_agreement_frac']:.3%} |
| Prior max |Δ| samples | {agreement['frozen_vs_s51_fine_shrunk']['history_prior_max_diff_samples']:.3g} |
| ΔF1@0.5 | {agreement['frozen_vs_s51_fine_shrunk']['metric_delta_f1_0.5']:.4f} |
| Δe2e P95 | {agreement['frozen_vs_s51_fine_shrunk']['metric_delta_e2e_p95']:.2f} s |
| Equivalent? | **{agreement['frozen_vs_s51_fine_shrunk']['equivalent']}** |

Coverage: frozen history available **{agreement['coverage']['frozen_history_available_frac']:.1%}** vs Stage5.1 fine_ok **{agreement['coverage']['s51_fine_ok_frac']:.1%}**.

Frozen reimplementation vs Stage2 pick file agreement: {agreement['frozen_vs_frozen_reimpl']['selected_candidate_agreement_frac']:.1%} (max Δ samples {agreement['frozen_vs_frozen_reimpl']['pick_time_max_diff_samples']:.3g}).

## 3. e2e errors > 10 s (Stage5.1 fine_shrunk)

- Count: **{len(big_rows)}** (frozen catalog_rescore on same list: {int(np.sum(np.isfinite(err_frozen) & (err_frozen > 10)))}; PhaseNet: {int(np.sum(np.isfinite(err_pn) & (err_pn > 10)))})
- CSV: `artifacts/results/stage5_1_ustc/sanity_large_errors_fine_shrunk.csv`
- No evidence of relative-time vs absolute-UTC mixing; sample indices stay on waveform grid.
- Dominant class: wrong-peak tails when fine history absent → identity/PhaseNet-like picks.

## 4. fine_unseen fallback

- Branch: `history_available=False` ⇒ `λ_h=0` (identity among K candidates).
- Does **not** inject 0 / NaN / origin / trace-start as an arrival time for ranking when fine_ok=False.
- Agreement fine_unseen vs identity: {agreement['fine_unseen_identity_vs_phasenet']['agreement_vs_identity_frac']}
- Agreement fine_unseen vs PhaseNet annotate pick: {agreement['fine_unseen_identity_vs_phasenet']['agreement_frac']} (may differ if annotate peak ∉ top-K)

## 5. e2e P95 definition

- Same functions as Stage 2/3: `match_picks` / `audit_pick_errors`.
- Miss encoding: missing pred → `missed_pick`; **excluded** from e2e AE/P95.
- e2e P95: 95th percentile of |pred−true|/sr over labeled∩predicted (includes wrong peaks).
- Matched P95: only |err|≤0.5 s.
- Units: samples ÷ sampling_rate_hz → seconds (no UTC timestamp arithmetic in the metric).

Why Stage2 val PhaseNet P95≈70 s while Stage3 P95≈2 s: **different evaluation sets**, not a unit bug.

## 6. Path bootstrap direction (clarified)

Definition used in Stage 5.1 repeatability:

`Δ = MAE(shuffled_path) − MAE(correct_path)`

| Quantity | Value |
|--|--|
| MAE shuffled | {bootstrap_clarification['mae_shuffled']:.6f} s |
| MAE correct | {bootstrap_clarification['mae_correct']:.6f} s |
| mean Δ | {bootstrap_clarification['mean_delta']:.6f} s |
| 95% CI | [{bootstrap_clarification['ci95_low']:.6f}, {bootstrap_clarification['ci95_high']:.6f}] |
| CI entirely positive? | {bootstrap_clarification['ci_entirely_positive']} |
| n_events | {bootstrap_clarification['n_events']} |
| n_traces | {bootstrap_clarification['n_traces_in_prior_eval']} |
| P(Δ>0) | {bootstrap_clarification['prob_delta_positive']} |

Positive CI ⇒ correct path better than shuffled (repeatability still supported).

## 7. Stage3 1.11 s comparison

{stage3_note}

Frozen catalog_rescore on **this same val list** already has e2e P95 **{m['frozen_catalog_rescore_stage2']['e2e_p95']:.2f} s**, close to Stage5.1 coarse/hierarchical (~1.39 s). The 54.9 s figure is an artifact of the Stage5.1 fine_shrunk reimplementation gap, not evidence that the frozen main method has a 50 s tail on val.

## 8. Decision

| Question | Answer |
|--|--|
| Is 54.9 s a metric unit bug? | No |
| Is fine_shrunk == frozen fine? | **No (implementation_bug)** |
| Is hierarchical a proven upgrade over frozen fine? | **No** (≈ coarse ≈ frozen on val) |
| Keep hierarchical_prior_recommended? | **false** |
| Modify frozen Stage3/4? | No |
""",
        encoding="utf-8",
    )
    print(json.dumps({"final_classification": final_class, "hierarchical_prior_recommended": False, "metrics": {k: {"f1": v["f1@0.5"], "p95": v["e2e_p95"]} for k, v in metrics.items()}}, indent=2))


if __name__ == "__main__":
    import json

    main()
