#!/usr/bin/env python
"""Stage 5.1: path-residual repeatability on train→val only (no Stage 3/4 test)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.analysis.hierarchical_residual import (
    ae_summary,
    assert_event_disjoint,
    assert_no_test_split,
    build_path_residual_table,
    directed_path_key,
    event_bootstrap_delta_mae,
    hierarchical_residual,
)
from earthquake.config import artifacts_dir, save_json
from earthquake.history.residual_prior import attach_baseline_and_residuals
from earthquake.history.shrinkage import shrink_residual
from earthquake.utils import ensure_dir, mad

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42})


def _station_col(df: pd.DataFrame) -> pd.Series:
    if "station_id" in df.columns:
        return df["station_id"].astype(str)
    return (
        df["network"].astype(str)
        + "."
        + df["station"].astype(str)
        + "."
        + df.get("location", pd.Series([""] * len(df))).fillna("").astype(str)
        + "."
        + df["channel_prefix"].astype(str)
    )


def distance_bin_residual(train: pd.DataFrame, residual_col: str, edges=(0, 50, 100, 200, 400, 1e9)) -> dict:
    out = {}
    x = train[np.isfinite(train[residual_col])].copy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x["distance_km"] >= lo) & (x["distance_km"] < hi)
        sub = x.loc[m, residual_col].to_numpy(dtype=float)
        out[(lo, hi)] = {
            "n": int(sub.size),
            "median": float(np.median(sub)) if sub.size else float("nan"),
            "mad": float(mad(sub)) if sub.size else float("nan"),
        }
    return out


def lookup_dist_bin(d: float, table: dict) -> float:
    if not np.isfinite(d):
        return float("nan")
    for (lo, hi), st in table.items():
        if lo <= d < hi:
            return float(st["median"])
    return float("nan")


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-fine", type=float, default=0.1)
    parser.add_argument("--grid-coarse", type=float, default=1.0)
    parser.add_argument("--shrinkage-k", type=float, default=50.0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage5_1_ustc")
    figs = ensure_dir(out / "figures")

    with open(artifacts_dir() / "results" / "stage2" / "travel_time_baseline.pkl", "rb") as f:
        baseline = pickle.load(f)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    train = events[events.split == "train"].copy()
    val = events[events.split == "val"].copy()
    assert_no_test_split(pd.concat([train, val], ignore_index=True), context="repeatability")
    assert_event_disjoint(set(train.event_id.astype(str)), set(val.event_id.astype(str)), context="repeatability")

    train_aug = attach_baseline_and_residuals(train, baseline)
    val_aug = attach_baseline_and_residuals(val, baseline)
    train_aug["split"] = "train"
    val_aug["split"] = "val"

    # Global residual distribution (train)
    r_train = train_aug["residual_tau_s"].to_numpy(dtype=float)
    r_train = r_train[np.isfinite(r_train)]

    fine_tab = build_path_residual_table(
        train_aug, residual_col="residual_tau_s", grid_size=args.grid_fine, min_history=args.min_history
    )
    coarse_tab = build_path_residual_table(
        train_aug, residual_col="residual_tau_s", grid_size=args.grid_coarse, min_history=args.min_history
    )
    dist_tab = distance_bin_residual(train_aug, "residual_tau_s")

    # Path-level summary on train
    path_rows = []
    for key, st in fine_tab.stats.items():
        path_rows.append({"path_key": key, "n": st["n"], "median": st["median"], "mad": st["mad"], "grid": args.grid_fine})
    path_df = pd.DataFrame(path_rows)
    between_mad = float(mad(path_df.loc[path_df.n >= args.min_history, "median"].to_numpy())) if (path_df.n >= args.min_history).any() else float("nan")
    within_mad = float(np.nanmedian(path_df.loc[path_df.n >= args.min_history, "mad"])) if (path_df.n >= args.min_history).any() else float("nan")
    variance_ratio = float(between_mad / within_mad) if np.isfinite(between_mad) and np.isfinite(within_mad) and within_mad > 1e-9 else float("nan")

    rng = np.random.default_rng(args.seed)
    # Precompute val keys and queries
    val_keys_f = [directed_path_key(r, args.grid_fine) for _, r in val_aug.iterrows()]
    val_keys_c = [directed_path_key(r, args.grid_coarse) for _, r in val_aug.iterrows()]
    fine_stats = [fine_tab.query(k) for k in val_keys_f]
    coarse_stats = [coarse_tab.query(k) for k in val_keys_c]

    # Shuffled fine path correspondence among rows with fine history available
    avail_idx = [i for i, st in enumerate(fine_stats) if st.get("history_available", 0) >= 1]
    shuffle_map = {}
    if avail_idx:
        perm = rng.permutation(avail_idx)
        for a, b in zip(avail_idx, perm):
            shuffle_map[a] = fine_stats[b]

    preds = {
        "mlp_only": [],
        "mlp_correct_path": [],
        "mlp_shuffled_path": [],
        "mlp_distance_bin": [],
        "mlp_shrunk_path": [],
        "mlp_hierarchical": [],
    }
    obs = []
    meta_rows = []
    for i, (_, row) in enumerate(val_aug.iterrows()):
        base = float(row["base_tau_s"])
        y = float(row["obs_tau_s"])
        obs.append(y - base)  # residual target for correlation; also keep abs travel err separately
        st_f = fine_stats[i]
        st_c = coarse_stats[i]
        # correct path (raw median if available else 0)
        if st_f.get("history_available", 0) >= 1 and np.isfinite(st_f["median"]):
            r_corr = float(st_f["median"])
            r_shrunk = shrink_residual(r_corr, fine_tab.global_median, st_f["n"], args.shrinkage_k)
        else:
            r_corr = fine_tab.global_median
            r_shrunk = fine_tab.global_median
        if i in shuffle_map:
            st_s = shuffle_map[i]
            r_shuf = float(st_s["median"]) if np.isfinite(st_s["median"]) else fine_tab.global_median
        else:
            r_shuf = fine_tab.global_median
        r_dist = lookup_dist_bin(float(row["distance_km"]), dist_tab)
        if not np.isfinite(r_dist):
            r_dist = fine_tab.global_median
        hier = hierarchical_residual(
            st_f, st_c, fine_tab.global_median, k_fine=args.shrinkage_k, k_coarse=args.shrinkage_k, min_history=args.min_history
        )
        preds["mlp_only"].append(base)
        preds["mlp_correct_path"].append(base + r_corr)
        preds["mlp_shuffled_path"].append(base + r_shuf)
        preds["mlp_distance_bin"].append(base + r_dist)
        preds["mlp_shrunk_path"].append(base + r_shrunk)
        preds["mlp_hierarchical"].append(base + float(hier["r_hat"]))
        meta_rows.append(
            {
                "trace_name": row["trace_name"],
                "event_id": row["event_id"],
                "station_id": str(_station_col(val_aug).iloc[i]),
                "distance_km": float(row["distance_km"]),
                "obs_tau_s": y,
                "base_tau_s": base,
                "obs_residual_s": y - base,
                "n_fine": float(st_f["n"]),
                "mad_fine": float(st_f["mad"]),
                "fine_seen": bool(st_f.get("history_available", 0) >= 1),
                "pred_residual_correct": r_corr,
                "pred_residual_shrunk": r_shrunk,
                "pred_residual_hier": float(hier["r_hat"]),
            }
        )

    obs_tau = val_aug["obs_tau_s"].to_numpy(dtype=float)
    event_ids = val_aug["event_id"].astype(str).to_numpy()
    summaries = {}
    errs = {}
    for name, pred in preds.items():
        pred = np.asarray(pred, dtype=float)
        err = pred - obs_tau
        errs[name] = err
        summaries[name] = ae_summary(err)

    # Correlation path residual predictor vs val residual
    meta = pd.DataFrame(meta_rows)
    m_ok = meta["fine_seen"] & np.isfinite(meta["obs_residual_s"]) & np.isfinite(meta["pred_residual_correct"])
    if m_ok.sum() >= 10:
        corr = float(np.corrcoef(meta.loc[m_ok, "pred_residual_correct"], meta.loc[m_ok, "obs_residual_s"])[0, 1])
    else:
        corr = float("nan")

    boot = {
        "correct_vs_shuffled": event_bootstrap_delta_mae(
            event_ids, errs["mlp_shuffled_path"], errs["mlp_correct_path"], n_boot=args.n_boot, seed=args.seed
        ),
        "correct_vs_mlp": event_bootstrap_delta_mae(
            event_ids, errs["mlp_only"], errs["mlp_correct_path"], n_boot=args.n_boot, seed=args.seed
        ),
        "shrunk_vs_mlp": event_bootstrap_delta_mae(
            event_ids, errs["mlp_only"], errs["mlp_shrunk_path"], n_boot=args.n_boot, seed=args.seed
        ),
        "hier_vs_shrunk": event_bootstrap_delta_mae(
            event_ids, errs["mlp_shrunk_path"], errs["mlp_hierarchical"], n_boot=args.n_boot, seed=args.seed
        ),
    }

    # Dominance: leave-one-station contribution to MAE reduction (correct vs shuffled)
    sta = _station_col(val_aug).to_numpy()
    ae_shuf = np.abs(errs["mlp_shuffled_path"])
    ae_corr = np.abs(errs["mlp_correct_path"])
    overall = float(np.nanmean(ae_shuf - ae_corr))
    station_contrib = []
    for s, sub in pd.DataFrame({"sta": sta, "d": ae_shuf - ae_corr}).groupby("sta"):
        # contribution ≈ how much overall drops if station removed
        mask = sta != s
        without = float(np.nanmean((ae_shuf - ae_corr)[mask]))
        station_contrib.append({"station_id": s, "n": int(len(sub)), "mean_delta": float(sub["d"].mean()), "overall_without": without})
    sc = pd.DataFrame(station_contrib).sort_values("n", ascending=False)
    top_share = float(sc.head(5)["n"].sum() / max(len(val_aug), 1))

    # Reliability vs count / MAD
    bins = []
    meta["ae_correct"] = np.abs(errs["mlp_correct_path"])
    meta["ae_mlp"] = np.abs(errs["mlp_only"])
    for label, m in [
        ("unseen_or_sparse", ~meta["fine_seen"]),
        ("n_5_19", meta["fine_seen"] & (meta["n_fine"] < 20)),
        ("n_ge_20", meta["fine_seen"] & (meta["n_fine"] >= 20)),
        ("mad_lt_0.2", meta["fine_seen"] & (meta["mad_fine"] < 0.2)),
        ("mad_ge_0.5", meta["fine_seen"] & (meta["mad_fine"] >= 0.5)),
    ]:
        if m.sum() == 0:
            continue
        bins.append(
            {
                "group": label,
                "n": int(m.sum()),
                "mae_mlp": float(meta.loc[m, "ae_mlp"].mean()),
                "mae_correct_path": float(meta.loc[m, "ae_correct"].mean()),
                "delta_mae": float(meta.loc[m, "ae_mlp"].mean() - meta.loc[m, "ae_correct"].mean()),
            }
        )
    by_group = pd.DataFrame(bins)
    by_group.to_csv(out / "path_repeatability_by_group.csv", index=False)

    # Distance exploratory on val (not Stage3/4)
    dist_rows = []
    for lo, hi, name in [(0, 50, "<50"), (50, 100, "50-100"), (100, 200, "100-200"), (200, 1e9, ">=200")]:
        m = (meta["distance_km"] >= lo) & (meta["distance_km"] < hi)
        if m.sum() == 0:
            continue
        dist_rows.append(
            {
                "distance_bin": name,
                "n": int(m.sum()),
                "mae_mlp": float(meta.loc[m, "ae_mlp"].mean()),
                "mae_path": float(meta.loc[m, "ae_correct"].mean()),
                "role": "validation_exploratory",
            }
        )
    pd.DataFrame(dist_rows).to_csv(out / "path_repeatability_by_distance_val.csv", index=False)

    correct_beats_shuffled = bool(
        summaries["mlp_correct_path"]["mae"] < summaries["mlp_shuffled_path"]["mae"]
        and boot["correct_vs_shuffled"]["ci95_low"] > 0
    )

    result = {
        "protocol": {
            "history": "train_only",
            "eval": "validation_only",
            "baseline": "frozen Stage2 distance MLP",
            "grid_fine": args.grid_fine,
            "grid_coarse": args.grid_coarse,
            "shrinkage_k": args.shrinkage_k,
            "forbidden": ["stage3_test", "stage4_holdout", "hyperparameter_search"],
        },
        "n_train": int(len(train_aug)),
        "n_val": int(len(val_aug)),
        "global_residual_train": {
            "median": float(np.median(r_train)),
            "mad": float(mad(r_train)),
            "p05": float(np.percentile(r_train, 5)),
            "p95": float(np.percentile(r_train, 95)),
        },
        "path_stats_fine": {
            "n_paths": int(len(path_df)),
            "n_paths_ge_min_history": int((path_df.n >= args.min_history).sum()),
            "between_path_median_mad": between_mad,
            "within_path_mad_median": within_mad,
            "between_within_mad_ratio": variance_ratio,
            "note": "robust between/within MAD ratio; not a formal ICC",
        },
        "prior_error_summaries": summaries,
        "path_residual_corr_val": corr,
        "bootstrap": boot,
        "correct_path_beats_shuffled": correct_beats_shuffled,
        "dominance": {
            "top5_stations_trace_share": top_share,
            "overall_mean_ae_reduction_shuffled_minus_correct": overall,
        },
        "coverage_val": {
            "fine_seen_frac": float(meta["fine_seen"].mean()),
            "median_n_fine_when_seen": float(meta.loc[meta.fine_seen, "n_fine"].median()) if meta.fine_seen.any() else 0.0,
        },
    }
    save_json(result, out / "path_repeatability.json")
    save_json(boot, out / "path_repeatability_bootstrap.json")
    meta.to_parquet(out / "path_repeatability_val_rows.parquet", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.hist(r_train, bins=80, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Train S residual (obs - MLP) [s]")
    ax.set_ylabel("Count")
    ax.set_title("Global residual distribution (train)")
    save_fig(fig, figs / "fig_residual_hist_train")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    names = ["mlp_only", "mlp_distance_bin", "mlp_shuffled_path", "mlp_correct_path", "mlp_shrunk_path", "mlp_hierarchical"]
    ax.bar(range(len(names)), [summaries[n]["mae"] for n in names], color="#55A868")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("mlp_", "") for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Val travel-time MAE [s]")
    ax.set_title("Path prior MAE on validation")
    save_fig(fig, figs / "fig_prior_mae_val")

    if m_ok.sum() >= 10:
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.scatter(meta.loc[m_ok, "pred_residual_correct"], meta.loc[m_ok, "obs_residual_s"], s=4, alpha=0.25)
        ax.set_xlabel("Train path residual median [s]")
        ax.set_ylabel("Val observed residual [s]")
        ax.set_title(f"Seen-path residual correlation={corr:.3f}")
        lim = np.nanpercentile(np.abs(meta.loc[m_ok, ["pred_residual_correct", "obs_residual_s"]]), 98)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        save_fig(fig, figs / "fig_path_residual_scatter_val")

    print({"correct_beats_shuffled": correct_beats_shuffled, "mae": {k: v["mae"] for k, v in summaries.items()}}, flush=True)


if __name__ == "__main__":
    main()
