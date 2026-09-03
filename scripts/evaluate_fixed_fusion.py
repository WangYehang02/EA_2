#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.history.temporal_store import PathStats
from earthquake.metrics import groupby_history_metrics, metrics_from_pick_frame
from earthquake.models.fixed_fusion import linear_fusion, log_space_fusion
from earthquake.models.historical_picker import HistoricalPicker
from earthquake.utils import ensure_dir


def load_history_map(protocol: str = "frozen") -> pd.DataFrame:
    path = artifacts_dir() / "history" / f"history_features_{protocol}.parquet"
    return pd.read_parquet(path)


def row_to_stats(h: pd.Series) -> PathStats:
    return PathStats(
        history_count=int(h["history_count"]),
        tau_p_median=float(h["tau_p_median"]),
        tau_p_mad=float(h["tau_p_mad"]),
        tau_s_median=float(h["tau_s_median"]),
        tau_s_mad=float(h["tau_s_mad"]),
        delta_sp_median=float(h["delta_sp_median"]),
        delta_sp_mad=float(h["delta_sp_mad"]),
        last_history_time=h.get("last_history_time"),
        nearest_historical_source_distance_km=float(h.get("nearest_historical_source_distance_km", float("nan"))),
        history_available=bool(h["history_available"]),
        matched_key=h.get("matched_key"),
        fallback_level=int(h.get("fallback_level", -1)) if pd.notna(h.get("fallback_level", np.nan)) else -1,
    )


def evaluate_alpha(
    events: pd.DataFrame,
    hist: pd.DataFrame,
    split: str,
    alpha_p: float,
    alpha_s: float,
    fusion: str,
    cfg: dict,
    max_traces: int,
) -> tuple[dict, pd.DataFrame]:
    sub_h = hist[hist["split"] == split].copy()
    sub_e = events.merge(sub_h, on=["trace_name", "event_id"], how="inner", suffixes=("", "_hist"))
    sub_e = sub_e.head(max_traces)

    # Use cached PhaseNet probs if present
    zpath = artifacts_dir() / "phasenet" / f"probs_{split}.zarr"
    names_path = artifacts_dir() / "phasenet" / f"probs_{split}_trace_names.json"
    pn_picks = pd.read_parquet(artifacts_dir() / "phasenet" / "phasenet_picks.parquet")
    pn_picks = pn_picks[pn_picks["split"] == split]
    if zpath.exists() and names_path.exists():
        names = load_json(names_path)
        name_to_idx = {str(n): i for i, n in enumerate(names)}
        z = zarr.open(str(zpath), mode="r")
    else:
        name_to_idx, z = {}, None

    picker = HistoricalPicker(
        alpha_p=alpha_p,
        alpha_s=alpha_s,
        fusion=fusion,
        prior_mode=cfg.get("prior_mode", "catalog_assisted"),
        min_sigma_s=float(cfg.get("min_sigma_s", 0.05)),
        max_sigma_s=float(cfg.get("max_sigma_s", 1.0)),
        pick_threshold=float(cfg.get("pick_threshold", 0.3)),
    )

    rows = []
    for _, row in sub_e.iterrows():
        tname = str(row["trace_name"])
        if z is not None and tname in name_to_idx:
            i = name_to_idx[tname]
            phasenet_out = {"p": np.asarray(z["p"][i]), "s": np.asarray(z["s"][i]), "noise": np.asarray(z["noise"][i])}
        else:
            # fallback: peak-only approximate using stored picks (should rarely happen in debug)
            continue
        stats = row_to_stats(row)
        # Need original event fields for prior center
        ev = events[events["trace_name"].astype(str) == tname].iloc[0]
        fused = picker.fuse(phasenet_out, ev, stats)
        rows.append(
            {
                "trace_name": tname,
                "event_id": str(row["event_id"]),
                "split": split,
                "sampling_rate_hz": float(ev["sampling_rate_hz"]),
                "true_p_sample": float(ev["p_arrival_sample"]) if pd.notna(ev["p_arrival_sample"]) else np.nan,
                "true_s_sample": float(ev["s_arrival_sample"]) if pd.notna(ev["s_arrival_sample"]) else np.nan,
                "pred_p_sample": fused["p_pick"]["peak_sample"],
                "pred_s_sample": fused["s_pick"]["peak_sample"],
                "history_available": fused["history_available"],
                "history_count": fused["history_count"],
                "alpha_p": alpha_p,
                "alpha_s": alpha_s,
                "fusion": fusion,
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "P": metrics_from_pick_frame(df, phase="p") if len(df) else {},
        "S": metrics_from_pick_frame(df, phase="s") if len(df) else {},
        "n": len(df),
        "alpha_p": alpha_p,
        "alpha_s": alpha_s,
        "fusion": fusion,
        "split": split,
    }
    return metrics, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_fixed.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    protocol = cfg.get("protocol", "frozen")
    hist = load_history_map(protocol)
    alpha_grid = cfg.get("alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    fusion_modes = ["linear", "log"] if cfg.get("fusion", "both") == "both" else [cfg.get("fusion", "linear")]
    max_traces = int(cfg.get("max_eval_traces", cfg.get("max_traces_per_split", 64)))

    out = ensure_dir(artifacts_dir() / "results")
    search_rows = []
    best = None
    best_score = -1.0

    # Search alphas on validation
    for fusion, ap, as_ in itertools.product(fusion_modes, alpha_grid, alpha_grid):
        metrics, _ = evaluate_alpha(events, hist, "val", ap, as_, fusion, cfg, max_traces)
        score = 0.5 * (metrics["P"].get("f1@0.1s", 0.0) + metrics["S"].get("f1@0.1s", 0.0))
        row = {
            "fusion": fusion,
            "alpha_p": ap,
            "alpha_s": as_,
            "score_f1_0.1": score,
            "p_f1_0.1": metrics["P"].get("f1@0.1s"),
            "s_f1_0.1": metrics["S"].get("f1@0.1s"),
            "p_f1_0.5": metrics["P"].get("f1@0.5s"),
            "s_f1_0.5": metrics["S"].get("f1@0.5s"),
            "p_mae": metrics["P"].get("mae"),
            "s_mae": metrics["S"].get("mae"),
            "n": metrics["n"],
        }
        search_rows.append(row)
        if score > best_score:
            best_score = score
            best = row

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(out / "fixed_fusion_metrics.csv", index=False)
    save_json(best or {}, out / "fixed_fusion_best_val.json")

    # Identity check alpha=1
    m1, df1 = evaluate_alpha(events, hist, "val", 1.0, 1.0, "linear", cfg, max_traces)
    pn = pd.read_parquet(artifacts_dir() / "phasenet" / "phasenet_picks.parquet")
    pn = pn[pn["split"] == "val"].set_index("trace_name")
    if len(df1):
        joined = df1.set_index("trace_name").join(pn[["pred_p_sample", "pred_s_sample"]], rsuffix="_pn")
        # Compare finite picks
        dp = np.nanmax(np.abs(joined["pred_p_sample"] - joined["pred_p_sample_pn"]))
        ds = np.nanmax(np.abs(joined["pred_s_sample"] - joined["pred_s_sample_pn"]))
        identity = {"max_abs_diff_p": float(dp) if np.isfinite(dp) else None, "max_abs_diff_s": float(ds) if np.isfinite(ds) else None}
    else:
        identity = {"max_abs_diff_p": None, "max_abs_diff_s": None}
    save_json(identity, out / "alpha_identity_check.json")

    # Test with best alphas
    if best:
        test_metrics, test_df = evaluate_alpha(
            events, hist, "test", best["alpha_p"], best["alpha_s"], best["fusion"], cfg, max_traces
        )
        save_json(test_metrics, out / "fixed_fusion_test_metrics.json")
        test_df.to_parquet(out / "fixed_fusion_test_picks.parquet", index=False)
        by_hist = pd.concat(
            [groupby_history_metrics(test_df, phase="p"), groupby_history_metrics(test_df, phase="s")],
            ignore_index=True,
        )
        by_hist.to_csv(out / "metrics_by_history_count.csv", index=False)
        print({"best_val": best, "test": test_metrics, "identity": identity})
    else:
        print({"warning": "no best alpha found", "identity": identity})


if __name__ == "__main__":
    main()
