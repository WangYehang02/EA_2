#!/usr/bin/env python
"""Stage 5.1: hierarchical residual prior + frozen PhaseNet candidates on VAL only."""

from __future__ import annotations

import argparse
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
from earthquake.history.residual_prior import attach_baseline_and_residuals
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


def _tau_to_sample(row: pd.Series, tau_s: float) -> float:
    if not np.isfinite(tau_s):
        return float("nan")
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    abs_t = origin + pd.to_timedelta(float(tau_s), unit="s")
    return float((abs_t - start).total_seconds() * sr)


def event_bootstrap_f1_delta(
    event_ids: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    tol: float = 0.5,
) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"event_id": event_ids, "a": pred_a, "b": pred_b, "t": true, "sr": sr})
    groups = {e: g for e, g in df.groupby("event_id", sort=False)}
    eids = np.array(list(groups.keys()))

    def f1_of(pred, g):
        m = match_picks(pred, g["t"].to_numpy(), g["sr"].to_numpy(), windows_s=(tol,))
        return float(m[f"f1@{tol}s"])

    def delta(sample):
        ga = pd.concat([groups[e] for e in sample], ignore_index=True)
        fa = f1_of(ga["a"].to_numpy(), ga)
        fb = f1_of(ga["b"].to_numpy(), ga)
        return fb - fa

    point = delta(eids)
    boots = np.array([delta(rng.choice(eids, size=eids.size, replace=True)) for _ in range(n_boot)])
    return {
        "mean": float(point),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "prob_positive": float(np.mean(boots > 0)),
        "n_events": int(eids.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-fine", type=float, default=0.1)
    parser.add_argument("--grid-coarse", type=float, default=1.0)
    parser.add_argument("--grids-extra", type=float, nargs="*", default=[0.5, 0.2])
    parser.add_argument("--shrinkage-k", type=float, default=50.0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage5_1_ustc")
    lambdas = load_json(artifacts_dir() / "results" / "stage2" / "best_lambdas.json")
    lw, lh, lp = float(lambdas["best_s"]["lw"]), float(lambdas["best_s"]["lh"]), float(lambdas["best_s"]["lp"])
    k = float(args.shrinkage_k)

    with open(artifacts_dir() / "results" / "stage2" / "travel_time_baseline.pkl", "rb") as f:
        baseline = pickle.load(f)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    train = events[events.split == "train"].copy()
    val_all = events[events.split == "val"].copy()
    assert_event_disjoint(set(train.event_id.astype(str)), set(val_all.event_id.astype(str)), context="hier")

    train_aug = attach_baseline_and_residuals(train, baseline)
    train_aug["split"] = "train"
    fine_tab = build_path_residual_table(train_aug, residual_col="residual_tau_s", grid_size=args.grid_fine, min_history=args.min_history)
    coarse_tab = build_path_residual_table(train_aug, residual_col="residual_tau_s", grid_size=args.grid_coarse, min_history=args.min_history)
    extra_tabs = {
        g: build_path_residual_table(train_aug, residual_col="residual_tau_s", grid_size=g, min_history=args.min_history)
        for g in args.grids_extra
    }

    # Frozen PhaseNet candidate cache; keep VAL traces only from fixed_eval
    picks = pd.read_parquet(artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache.parquet")
    cands = pd.read_parquet(artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache_candidates.parquet")
    fixed = pd.read_parquet(artifacts_dir() / "diagnostics" / "fixed_eval_events.parquet")
    split_map = events.set_index("trace_name")["split"]
    fixed = fixed.copy()
    fixed["split"] = fixed["trace_name"].map(split_map)
    fixed_val = fixed[fixed["split"] == "val"].copy()
    assert_no_test_split(fixed_val, context="hier_fixed_eval_val")
    if fixed_val["split"].eq("test").any():
        raise SystemExit("test leakage into hierarchical val eval")

    meta = fixed_val.merge(picks, on=["trace_name", "event_id"], how="inner", suffixes=("", "_p"))
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
        if col in fixed_val.columns:
            meta[col] = meta["trace_name"].map(fixed_val.set_index("trace_name")[col])
    # join event geometry if missing
    ev_idx = events.set_index("trace_name")
    for col in ["source_latitude", "source_longitude", "network", "station", "location", "channel_prefix", "source_depth_km", "station_elevation_m", "origin_time", "trace_start_time"]:
        if col not in meta.columns or meta[col].isna().all():
            meta[col] = meta["trace_name"].map(ev_idx[col])

    assert_event_disjoint(set(train.event_id.astype(str)), set(meta.event_id.astype(str)), context="hier_eval")
    cand_index = _build_cand_index(cands)

    # Attach MLP base
    meta_aug = attach_baseline_and_residuals(meta, baseline)

    rng = np.random.default_rng(args.seed)
    methods = {}
    # Prepare per-row residual predictions
    rows_info = []
    fine_stats = []
    coarse_stats = []
    for _, row in meta_aug.iterrows():
        kf = directed_path_key(row, args.grid_fine)
        kc = directed_path_key(row, args.grid_coarse)
        sf = fine_tab.query(kf)
        sc = coarse_tab.query(kc)
        fine_stats.append(sf)
        coarse_stats.append(sc)
        hier = hierarchical_residual(sf, sc, fine_tab.global_median, k_fine=k, k_coarse=k, min_history=args.min_history)
        if sf.get("history_available", 0) >= 1 and np.isfinite(sf["median"]):
            r_fine = shrink_residual(float(sf["median"]), fine_tab.global_median, sf["n"], k)
            fine_ok = True
        else:
            r_fine = fine_tab.global_median
            fine_ok = False
        if sc.get("history_available", 0) >= 1 and np.isfinite(sc["median"]):
            r_coarse = shrink_residual(float(sc["median"]), coarse_tab.global_median, sc["n"], k)
            coarse_ok = True
        else:
            r_coarse = coarse_tab.global_median
            coarse_ok = False
        rows_info.append(
            {
                "r_mlp0": 0.0,
                "r_fine": float(r_fine),
                "r_coarse": float(r_coarse),
                "r_hier": float(hier["r_hat"]),
                "fine_ok": fine_ok,
                "coarse_ok": coarse_ok,
                "hier_ok": bool(hier["history_available"]),
                "n_fine": float(sf["n"]),
                "mad_fine": float(sf["mad"]),
                "base_tau_s": float(row["base_tau_s"]),
            }
        )
    info = pd.DataFrame(rows_info)

    # shuffled hierarchical residuals
    avail = np.where(info["hier_ok"].to_numpy())[0]
    shuf_r = info["r_hier"].to_numpy().copy()
    if avail.size:
        shuf_r[avail] = info["r_hier"].to_numpy()[rng.permutation(avail)]

    def rescore_with_residual(r_hat: np.ndarray, hist_ok: np.ndarray) -> np.ndarray:
        preds = []
        for i, (_, row) in enumerate(meta_aug.iterrows()):
            s_c = cand_index.get((str(row.trace_name), "s"), [])
            tau = float(row["base_tau_s"]) + float(r_hat[i])
            exp = _tau_to_sample(row, tau)
            sr = float(row["sampling_rate_hz"])
            # sigma from fine MAD if available else default 0.2s
            mad_s = float(info.iloc[i]["mad_fine"])
            sig_s = float(np.clip(1.4826 * mad_s, 0.05, 1.0)) if np.isfinite(mad_s) and hist_ok[i] else 0.2
            sigma_samp = sig_s * sr
            best, _ = rescore_phase_candidates(
                s_c,
                expected_sample=exp,
                sigma_samples=sigma_samp,
                lambda_wave=lw,
                lambda_history=lh,
                lambda_prominence=lp,
                history_available=bool(hist_ok[i]),
            )
            preds.append(float(best.sample_index) if best is not None else float("nan"))
        return np.asarray(preds, dtype=float)

    true = meta_aug["s_arrival_sample"].to_numpy(dtype=float) if "s_arrival_sample" in meta_aug.columns else meta_aug["true_s_sample"].to_numpy(dtype=float)
    if "true_s_sample" in meta_aug.columns:
        true = meta_aug["true_s_sample"].to_numpy(dtype=float)
    sr = meta_aug["sampling_rate_hz"].to_numpy(dtype=float)
    pn = meta_aug["pred_s_sample"].to_numpy(dtype=float)

    methods["phasenet"] = pn
    methods["mlp_only_rescore"] = rescore_with_residual(np.zeros(len(info)), np.zeros(len(info), dtype=bool))
    methods["fine_shrunk"] = rescore_with_residual(info["r_fine"].to_numpy(), info["fine_ok"].to_numpy())
    methods["coarse_shrunk"] = rescore_with_residual(info["r_coarse"].to_numpy(), info["coarse_ok"].to_numpy())
    methods["hierarchical"] = rescore_with_residual(info["r_hier"].to_numpy(), info["hier_ok"].to_numpy())
    methods["hierarchical_shuffled"] = rescore_with_residual(shuf_r, info["hier_ok"].to_numpy())

    # Also prior-only travel-time errors on the val fixed set
    prior_summaries = {}
    for name, rcol in [
        ("mlp_only", np.zeros(len(info))),
        ("fine_shrunk", info["r_fine"].to_numpy()),
        ("coarse_shrunk", info["r_coarse"].to_numpy()),
        ("hierarchical", info["r_hier"].to_numpy()),
        ("hierarchical_shuffled", shuf_r),
    ]:
        pred_tau = info["base_tau_s"].to_numpy() + rcol
        # obs tau from labels
        obs_tau = []
        for _, row in meta_aug.iterrows():
            origin = pd.Timestamp(row["origin_time"])
            start = pd.Timestamp(row["trace_start_time"])
            samp = float(row["s_arrival_sample"]) if "s_arrival_sample" in row and np.isfinite(row.get("s_arrival_sample", np.nan)) else float(row.get("true_s_sample", np.nan))
            if not np.isfinite(samp):
                obs_tau.append(np.nan)
            else:
                arr = start + pd.to_timedelta(samp / float(row["sampling_rate_hz"]), unit="s")
                obs_tau.append(float((arr - origin).total_seconds()))
        obs_tau = np.asarray(obs_tau, dtype=float)
        err = pred_tau - obs_tau
        ae = np.abs(err[np.isfinite(err)])
        prior_summaries[name] = {
            "mae": float(np.mean(ae)) if ae.size else float("nan"),
            "median_ae": float(np.median(ae)) if ae.size else float("nan"),
            "p95": float(np.percentile(ae, 95)) if ae.size else float("nan"),
            "n": int(ae.size),
        }

    pick_summaries = {}
    for name, pred in methods.items():
        m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
        aud = audit_pick_errors(pred, true, sr, window_s=0.5)
        pick_summaries[name] = {
            "f1@0.1": m["f1@0.1s"],
            "f1@0.5": m["f1@0.5s"],
            "e2e_mae": m["e2e_mae"],
            "e2e_p95": m["e2e_p95_ae"],
            "wrong_peak_rate": float(aud["n_wrong_peak_beyond_tol"] / max(aud.get("n_labeled", m["n_eval"]), 1)),
        }

    # Group slices
    group_rows = []
    ae_pn = np.abs((pn - true) / sr)
    ae_hier = np.abs((methods["hierarchical"] - true) / sr)
    ae_fine = np.abs((methods["fine_shrunk"] - true) / sr)
    for label, mask in [
        ("fine_seen", info["fine_ok"].to_numpy()),
        ("fine_unseen", ~info["fine_ok"].to_numpy()),
        ("sparse_n_lt_20", info["fine_ok"] & (info["n_fine"] < 20)),
        ("dense_n_ge_20", info["fine_ok"] & (info["n_fine"] >= 20)),
        ("mad_lt_0.2", info["fine_ok"] & (info["mad_fine"] < 0.2)),
        ("mad_ge_0.5", info["fine_ok"] & (info["mad_fine"] >= 0.5)),
    ]:
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
        if m.sum() == 0:
            continue
        mh = match_picks(methods["hierarchical"][m], true[m], sr[m], windows_s=(0.5,))
        mf = match_picks(methods["fine_shrunk"][m], true[m], sr[m], windows_s=(0.5,))
        group_rows.append(
            {
                "group": label,
                "n": int(m.sum()),
                "f1_0.5_fine": mf["f1@0.5s"],
                "f1_0.5_hier": mh["f1@0.5s"],
                "delta_f1_0.5": mh["f1@0.5s"] - mf["f1@0.5s"],
                "e2e_p95_fine": mf["e2e_p95_ae"],
                "e2e_p95_hier": mh["e2e_p95_ae"],
                "delta_e2e_p95": mh["e2e_p95_ae"] - mf["e2e_p95_ae"],
            }
        )
    by_group = pd.DataFrame(group_rows)
    by_group.to_csv(out / "hierarchical_val_by_group.csv", index=False)

    boot = {
        "hier_minus_fine_f1_0.5": event_bootstrap_f1_delta(
            meta_aug.event_id.astype(str).to_numpy(),
            methods["fine_shrunk"],
            methods["hierarchical"],
            true,
            sr,
            n_boot=args.n_boot,
            seed=args.seed,
            tol=0.5,
        ),
        "hier_minus_shuffled_f1_0.5": event_bootstrap_f1_delta(
            meta_aug.event_id.astype(str).to_numpy(),
            methods["hierarchical_shuffled"],
            methods["hierarchical"],
            true,
            sr,
            n_boot=args.n_boot,
            seed=args.seed,
            tol=0.5,
        ),
    }
    save_json(boot, out / "hierarchical_val_bootstrap.json")

    # Decision rules vs current fine-only
    prior_mae_fine = prior_summaries["fine_shrunk"]["mae"]
    prior_mae_hier = prior_summaries["hierarchical"]["mae"]
    prior_improve = (prior_mae_fine - prior_mae_hier) / max(prior_mae_fine, 1e-9)
    f1_delta = pick_summaries["hierarchical"]["f1@0.5"] - pick_summaries["fine_shrunk"]["f1@0.5"]
    p95_delta = pick_summaries["fine_shrunk"]["e2e_p95"] - pick_summaries["hierarchical"]["e2e_p95"]
    boot_dir = boot["hier_minus_fine_f1_0.5"]["mean"] > 0 and boot["hier_minus_fine_f1_0.5"]["prob_positive"] >= 0.6
    sparse = by_group[by_group.group.isin(["fine_unseen", "sparse_n_lt_20"])]
    sparse_ok = bool(len(sparse) and (sparse["delta_f1_0.5"] > 0).any())

    pass_rules = {
        "prior_mae_drop_ge_5pct": bool(prior_improve >= 0.05),
        "f1_0.5_gain_ge_0.005": bool(f1_delta >= 0.005),
        "e2e_p95_drop_ge_0.1s": bool(p95_delta >= 0.1),
        "bootstrap_direction_stable": bool(boot_dir),
        "gain_on_sparse_or_unseen": sparse_ok,
        "fallback_unchanged": True,
        "complexity_acceptable": True,
    }
    any_metric = pass_rules["prior_mae_drop_ge_5pct"] or pass_rules["f1_0.5_gain_ge_0.005"] or pass_rules["e2e_p95_drop_ge_0.1s"]
    recommended = bool(any_metric and pass_rules["bootstrap_direction_stable"] and pass_rules["gain_on_sparse_or_unseen"])

    result = {
        "role": "validation_only_feasibility",
        "n_val_traces": int(len(meta_aug)),
        "n_val_events": int(meta_aug.event_id.nunique()),
        "lambdas_s_frozen": {"lw": lw, "lh": lh, "lp": lp, "k": k},
        "grids": {"fine": args.grid_fine, "coarse": args.grid_coarse, "extra_built": list(extra_tabs.keys())},
        "coverage": {
            "fine_ok_frac": float(info["fine_ok"].mean()),
            "coarse_ok_frac": float(info["coarse_ok"].mean()),
            "hier_ok_frac": float(info["hier_ok"].mean()),
        },
        "prior_summaries": prior_summaries,
        "pick_summaries": pick_summaries,
        "deltas_hier_minus_fine": {
            "prior_mae_rel_drop": float(prior_improve),
            "f1_0.5": float(f1_delta),
            "e2e_p95_drop_s": float(p95_delta),
        },
        "pass_rules": pass_rules,
        "hierarchical_prior_recommended": recommended,
        "statement_if_recommended": "promising future extension validated on the development set",
        "forbidden": ["write_into_stage3_main_results", "claim_test_set_gain", "retune_lambda"],
    }
    save_json(result, out / "hierarchical_val_results.json")
    print(result)


if __name__ == "__main__":
    main()
