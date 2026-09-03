#!/usr/bin/env python
from __future__ import annotations

"""Re-run PhaseNet / history / fixed-alpha fusion on the fixed eval set after alignment fix."""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.history.prior import build_priors_for_row
from earthquake.history.temporal_store import PathStats
from earthquake.metrics import match_picks, noise_false_positive_rate
from earthquake.models.fixed_fusion import linear_fusion, log_space_fusion
from earthquake.models.phasenet_wrapper import PhaseNetWrapper
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir, haversine_km


def row_to_stats(h: pd.Series) -> PathStats:
    return PathStats(
        history_count=int(h.get("history_count", 0)),
        tau_p_median=float(h.get("tau_p_median", np.nan)),
        tau_p_mad=float(h.get("tau_p_mad", np.nan)),
        tau_s_median=float(h.get("tau_s_median", np.nan)),
        tau_s_mad=float(h.get("tau_s_mad", np.nan)),
        delta_sp_median=float(h.get("delta_sp_median", np.nan)),
        delta_sp_mad=float(h.get("delta_sp_mad", np.nan)),
        last_history_time=h.get("last_history_time"),
        nearest_historical_source_distance_km=float(h.get("nearest_historical_source_distance_km", np.nan)),
        history_available=bool(h.get("history_available", False)),
        matched_key=h.get("matched_key"),
        fallback_level=int(h.get("fallback_level", -1)) if pd.notna(h.get("fallback_level", np.nan)) else -1,
    )


def distance_bin_prior_sample(row: pd.Series, travel_table: pd.DataFrame, phase: str) -> float:
    """Median travel time by epicentral-distance bin -> sample on waveform."""
    if travel_table is None or len(travel_table) == 0 or not np.isfinite(row.get("distance_km", np.nan)):
        return float("nan")
    d = float(row["distance_km"])
    # find nearest bin center
    idx = (travel_table["dist_center"] - d).abs().idxmin()
    tau = float(travel_table.loc[idx, f"tau_{phase}_median"])
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    arrival = origin + pd.to_timedelta(tau, unit="s")
    return float((arrival - start).total_seconds() * sr)


def build_distance_table(train_events: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    df = train_events.dropna(subset=["distance_km", "path_travel_time_p_s", "path_travel_time_s_s"]).copy()
    if len(df) == 0:
        # derive from samples
        df = train_events.dropna(subset=["distance_km", "p_arrival_sample", "s_arrival_sample", "origin_time", "trace_start_time"]).copy()
        df["path_travel_time_p_s"] = (
            pd.to_datetime(df["trace_start_time"], utc=True)
            + pd.to_timedelta(df["p_arrival_sample"] / df["sampling_rate_hz"], unit="s")
            - pd.to_datetime(df["origin_time"], utc=True)
        ).dt.total_seconds()
        df["path_travel_time_s_s"] = (
            pd.to_datetime(df["trace_start_time"], utc=True)
            + pd.to_timedelta(df["s_arrival_sample"] / df["sampling_rate_hz"], unit="s")
            - pd.to_datetime(df["origin_time"], utc=True)
        ).dt.total_seconds()
    df["dist_bin"] = pd.qcut(df["distance_km"], q=min(n_bins, max(df["distance_km"].nunique(), 1)), duplicates="drop")
    g = df.groupby("dist_bin", observed=True).agg(
        dist_center=("distance_km", "median"),
        tau_p_median=("path_travel_time_p_s", "median"),
        tau_s_median=("path_travel_time_s_s", "median"),
        n=("distance_km", "size"),
    )
    return g.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="stead")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-traces", type=int, default=2000, help="limit for runtime; use -1 for all fixed eval events")
    parser.add_argument("--ckpt", default=None, help="optional finetuned checkpoint")
    parser.add_argument("--alpha-step", type=float, default=0.1)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage15")
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    hist = pd.read_parquet(artifacts_dir() / "history" / "history_features_frozen.parquet")
    fixed_ev = pd.read_parquet(artifacts_dir() / "diagnostics" / "fixed_eval_events.parquet")
    if args.max_traces > 0:
        fixed_ev = fixed_ev.head(args.max_traces)

    # merge history
    df = fixed_ev.merge(hist, on=["trace_name", "event_id"], how="left", suffixes=("", "_h"))
    train_events = events[events["split"] == "train"]
    dist_table = build_distance_table(train_events)

    wrap = PhaseNetWrapper(weight=args.weight, device=args.device)
    if args.ckpt:
        import torch, seisbench.models as sbm
        ckpt = torch.load(args.ckpt, map_location=args.device)
        # keep architecture from init weight, load finetuned weights
        wrap.model.load_state_dict(ckpt["model"])
        wrap._ref.model = wrap.model

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    records = []
    with InstanceHDF5Reader(h5) as reader:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="stage15-eval"):
            wave = reader.read_waveform(str(row.trace_name))
            pn = wrap.predict_row(wave, row)
            stats = row_to_stats(row)
            # shuffled stats: keep count/MAD, replace medians from a random other row with history
            records.append(
                {
                    "row": row,
                    "pn": pn,
                    "stats": stats,
                    "wave_len": wave.shape[-1],
                }
            )

    # Build shuffled correspondence
    rng = np.random.default_rng(0)
    hist_ok = [r for r in records if r["stats"].history_available]
    shuffled_stats = []
    for r in records:
        if r["stats"].history_available and len(hist_ok) > 1:
            j = int(rng.integers(0, len(hist_ok)))
            shuffled_stats.append(hist_ok[j]["stats"])
        else:
            shuffled_stats.append(r["stats"])

    alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 2).tolist()

    def evaluate(mode: str, alpha_p: float, alpha_s: float, fusion: str, prior_mode: str, use_shuffle: bool, use_distance: bool):
        pred_p, true_p, pred_s, true_s, srs = [], [], [], [], []
        for i, r in enumerate(records):
            row = r["row"]
            pn = r["pn"]
            stats = shuffled_stats[i] if use_shuffle else r["stats"]
            n = r["wave_len"]
            sr = float(row["sampling_rate_hz"])
            if use_distance:
                # distance-bin history replacement
                cp = distance_bin_prior_sample(row, dist_table, "p")
                cs = distance_bin_prior_sample(row, dist_table, "s")
                from earthquake.history.prior import gaussian_prior, sigma_from_mad
                priors = {
                    "prior_p": gaussian_prior(n, cp, sigma_from_mad(0.2, sr)),
                    "prior_s": gaussian_prior(n, cs, sigma_from_mad(0.2, sr)),
                    "force_phasenet": not np.isfinite(cp),
                    "history_available": np.isfinite(cp),
                }
            else:
                priors = build_priors_for_row(
                    row,
                    stats,
                    n_samples=n,
                    mode=prior_mode,
                    phasenet_p_sample=float(pn["p_pred_sample_on_waveform"]),
                )
            fuse = linear_fusion if fusion == "linear" else log_space_fusion
            if mode == "phasenet":
                fp, fs = pn["p"], pn["s"]
            elif mode == "history":
                fp, fs = priors["prior_p"], priors["prior_s"]
                if prior_mode == "blind_s":
                    fp = pn["p"]
            else:
                force = bool(priors.get("force_phasenet", False))
                fp = fuse(pn["p"], priors["prior_p"], alpha_p, force_phasenet=force or prior_mode == "blind_s")
                if prior_mode == "blind_s":
                    fp = pn["p"]
                fs = fuse(pn["s"], priors["prior_s"], alpha_s, force_phasenet=force)
            pp = pick_from_prob(fp, threshold=0.3)
            ss = pick_from_prob(fs, threshold=0.3)
            pred_p.append(pp["peak_sample"])
            pred_s.append(ss["peak_sample"])
            true_p.append(float(row["p_arrival_sample"]) if pd.notna(row["p_arrival_sample"]) else np.nan)
            true_s.append(float(row["s_arrival_sample"]) if pd.notna(row["s_arrival_sample"]) else np.nan)
            srs.append(sr)
        return {
            "P": match_picks(np.array(pred_p), np.array(true_p), np.array(srs)),
            "S": match_picks(np.array(pred_s), np.array(true_s), np.array(srs)),
            "n": len(records),
            "mode": mode,
            "alpha_p": alpha_p,
            "alpha_s": alpha_s,
            "fusion": fusion,
            "prior_mode": prior_mode,
            "shuffled": use_shuffle,
            "distance_bin": use_distance,
        }

    # baselines
    results = []
    results.append(evaluate("phasenet", 1, 1, "linear", "catalog_assisted", False, False))
    results.append(evaluate("history", 0, 0, "linear", "catalog_assisted", False, False))
    results.append(evaluate("history", 0, 0, "linear", "catalog_assisted", True, False))  # shuffled
    results.append(evaluate("history", 0, 0, "linear", "catalog_assisted", False, True))  # distance bin

    # alpha search on a validation subset of records (first half of fixed eval that is val split)
    val_idx = [i for i, r in enumerate(records) if str(r["row"].get("split", "")) == "val"]
    if not val_idx:
        val_idx = list(range(min(500, len(records))))

    # Temporarily evaluate alpha on all records but mark; for speed search on val_idx only via subsetting
    # Simpler: search using evaluate() on full set is expensive; restrict records temporarily
    all_records = records
    records = [all_records[i] for i in val_idx]
    shuffled_stats_all = shuffled_stats
    shuffled_stats = [shuffled_stats_all[i] for i in val_idx]

    best = None
    best_score = -1
    search_rows = []
    for fusion, ap, as_ in itertools.product(["linear", "log"], alpha_grid, alpha_grid):
        m = evaluate("fusion", ap, as_, fusion, "catalog_assisted", False, False)
        score = 0.5 * (m["P"]["f1@0.1s"] + m["S"]["f1@0.1s"])
        row = {"fusion": fusion, "alpha_p": ap, "alpha_s": as_, "score": score, "p_f1_0.1": m["P"]["f1@0.1s"], "s_f1_0.1": m["S"]["f1@0.1s"]}
        search_rows.append(row)
        if score > best_score:
            best_score = score
            best = (fusion, ap, as_, m)
    pd.DataFrame(search_rows).to_csv(out / "alpha_search_val.csv", index=False)

    # restore full records for test-like reporting
    records = all_records
    shuffled_stats = shuffled_stats_all
    if best:
        fusion, ap, as_, _ = best
        results.append(evaluate("fusion", ap, as_, fusion, "catalog_assisted", False, False))
        results.append(evaluate("fusion", ap, as_, fusion, "catalog_assisted", True, False))
        results.append(evaluate("fusion", ap, as_, fusion, "blind_s", False, False))
        save_json({"fusion": fusion, "alpha_p": ap, "alpha_s": as_, "score": best_score}, out / "best_alpha.json")

    # flatten results
    flat = []
    for m in results:
        flat.append(
            {
                "mode": m["mode"],
                "fusion": m.get("fusion"),
                "alpha_p": m.get("alpha_p"),
                "alpha_s": m.get("alpha_s"),
                "prior_mode": m.get("prior_mode"),
                "shuffled": m.get("shuffled"),
                "distance_bin": m.get("distance_bin"),
                "n": m["n"],
                "p_f1@0.1s": m["P"]["f1@0.1s"],
                "p_f1@0.5s": m["P"]["f1@0.5s"],
                "p_mae": m["P"]["mae"],
                "p_median_ae": m["P"]["median_ae"],
                "s_f1@0.1s": m["S"]["f1@0.1s"],
                "s_f1@0.5s": m["S"]["f1@0.5s"],
                "s_mae": m["S"]["mae"],
                "s_median_ae": m["S"]["median_ae"],
            }
        )
    pd.DataFrame(flat).to_csv(out / "stage15_comparison.csv", index=False)
    save_json(flat, out / "stage15_comparison.json")
    print(pd.DataFrame(flat).to_string(index=False))


if __name__ == "__main__":
    main()
