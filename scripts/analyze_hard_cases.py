#!/usr/bin/env python
"""Hard-subset analysis for Stage 2 candidate re-scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.metrics import match_picks
from earthquake.utils import ensure_dir


def phase_metrics(df: pd.DataFrame, pred_col: str, true_col: str) -> dict:
    if len(df) == 0:
        return {"n": 0}
    m = match_picks(df[pred_col].to_numpy(), df[true_col].to_numpy(), df["sampling_rate_hz"].to_numpy())
    return {"n": len(df), "f1@0.1s": m["f1@0.1s"], "f1@0.5s": m["f1@0.5s"], "mae": m["mae"], "p95": m["p95_ae"], "median_ae": m["median_ae"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_fixed.yaml")
    parser.add_argument("--baseline-mode", default="phasenet")
    parser.add_argument("--improved-mode", default="catalog_rescore")
    args = parser.parse_args()
    _ = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage2")

    picks = pd.read_parquet(out / "candidate_rescoring_picks.parquet")
    meta = pd.read_parquet(out / "fixed_eval_joined_meta.parquet")
    base = picks[picks["mode"] == args.baseline_mode].rename(columns={"pred_p_sample": "base_p", "pred_s_sample": "base_s"})
    imp = picks[picks["mode"] == args.improved_mode].rename(columns={"pred_p_sample": "imp_p", "pred_s_sample": "imp_s"})
    df = meta.merge(base[["trace_name", "base_p", "base_s"]], on="trace_name").merge(
        imp[["trace_name", "imp_p", "imp_s"]], on="trace_name"
    )

    # multi-peak flag
    df["p_multipeak"] = df["p_n_cands"].fillna(1) > 1
    df["s_multipeak"] = df["s_n_cands"].fillna(1) > 1
    mad = df["residual_s_mad"].fillna(df["tau_s_mad"])

    groups = {
        "snr_<0": df["snr_db"] < 0,
        "snr_0_5": (df["snr_db"] >= 0) & (df["snr_db"] < 5),
        "snr_5_10": (df["snr_db"] >= 5) & (df["snr_db"] < 10),
        "snr_>10": df["snr_db"] >= 10,
        "s_single_peak": ~df["s_multipeak"],
        "s_multi_peak": df["s_multipeak"],
        "s_prob_<0.3": df["s_peak_probability"] < 0.3,
        "s_prob_0.3_0.7": (df["s_peak_probability"] >= 0.3) & (df["s_peak_probability"] <= 0.7),
        "s_prob_>0.7": df["s_peak_probability"] > 0.7,
        "hist_<5": df["history_count"].fillna(0) < 5,
        "hist_5_9": (df["history_count"] >= 5) & (df["history_count"] <= 9),
        "hist_10_19": (df["history_count"] >= 10) & (df["history_count"] <= 19),
        "hist_>=20": df["history_count"] >= 20,
        "mad_<0.2": mad < 0.2,
        "mad_0.2_0.5": (mad >= 0.2) & (mad <= 0.5),
        "mad_0.5_1.0": (mad > 0.5) & (mad <= 1.0),
        "mad_>1.0": mad > 1.0,
        "dist_<50": df["distance_km"] < 50,
        "dist_50_100": (df["distance_km"] >= 50) & (df["distance_km"] < 100),
        "dist_>=100": df["distance_km"] >= 100,
    }

    rows = []
    for name, mask in groups.items():
        sub = df[mask]
        b = phase_metrics(sub, "base_s", "true_s_sample")
        i = phase_metrics(sub, "imp_s", "true_s_sample")
        rows.append(
            {
                "subset": name,
                "n": len(sub),
                "base_s_f1@0.5": b.get("f1@0.5s"),
                "imp_s_f1@0.5": i.get("f1@0.5s"),
                "delta_s_f1@0.5": (i.get("f1@0.5s") - b.get("f1@0.5s")) if b.get("f1@0.5s") is not None and i.get("f1@0.5s") is not None else None,
                "base_s_p95": b.get("p95"),
                "imp_s_p95": i.get("p95"),
                "delta_s_p95": (i.get("p95") - b.get("p95")) if b.get("p95") is not None and i.get("p95") is not None else None,
                "base_s_mae": b.get("mae"),
                "imp_s_mae": i.get("mae"),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "hard_cases_s.csv", index=False)

    # When does history hurt? subsets where delta F1 < 0
    hurt = tab[tab["delta_s_f1@0.5"].fillna(0) < -0.01].sort_values("delta_s_f1@0.5")
    help_ = tab[tab["delta_s_f1@0.5"].fillna(0) > 0.01].sort_values("delta_s_f1@0.5", ascending=False)
    summary = {
        "helps_most": help_.head(8).to_dict(orient="records"),
        "hurts_most": hurt.head(8).to_dict(orient="records"),
        "recommend_disable_when": [
            "history_count < 5",
            "residual/tau MAD > 1.0s",
            "history unavailable (already identity)",
        ],
    }
    save_json(summary, out / "hard_cases_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
