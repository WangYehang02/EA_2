#!/usr/bin/env python
"""Analyze learned gate behavior vs history / PhaseNet confidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.utils import ensure_dir


def _bin_stats(df: pd.DataFrame, key: str, bins, labels) -> list[dict]:
    out = []
    cat = pd.cut(df[key], bins=bins, labels=labels, include_lowest=True)
    for lab in labels:
        sub = df[cat == lab]
        if len(sub) == 0:
            continue
        g = sub["gate"].to_numpy(dtype=np.float64)
        g = g[np.isfinite(g)]
        out.append(
            {
                "feature": key,
                "bin": str(lab),
                "n": int(len(sub)),
                "gate_mean": float(np.mean(g)) if g.size else float("nan"),
                "gate_std": float(np.std(g)) if g.size else float("nan"),
                "learned_minus_fixed_abserr_mean": float(
                    np.nanmean(sub["err_gate"] - sub["err_fixed"])
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_catalog.yaml")
    parser.add_argument("--picks", default="artifacts/results/stage3/learned_gate_picks_test.parquet")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage3")
    picks = pd.read_parquet(ROOT / args.picks)
    if "seed" in picks.columns:
        picks = picks[picks.seed == int(picks.seed.mode().iloc[0])].copy()

    tag = "debug" if "debug" in Path(args.config).stem else str(cfg.get("mode", "catalog"))
    base = out / f"gate_baselines_{tag}.parquet"
    meta = pd.read_parquet(base) if base.exists() else None
    if meta is not None:
        meta = meta[meta.split == "test"] if "split" in meta.columns else meta
        picks = picks.merge(
            meta[["trace_name", "abcd_class", "history_count", "history_mad", "fallback_level"]],
            on="trace_name",
            how="left",
        )

    sr = picks["sampling_rate_hz"].to_numpy()
    true = picks["true_s_sample"].to_numpy()
    picks["err_pn"] = np.abs(picks["pred_s_phasenet"] - true) / sr
    picks["err_fixed"] = np.abs(picks["pred_s_fixed_rescore"] - true) / sr
    picks["err_gate"] = np.abs(picks["pred_s_learned_gate"] - true) / sr
    picks["pn_better_than_hist"] = picks["err_pn"] <= picks["err_fixed"]

    g = picks["gate"].to_numpy(dtype=np.float64)
    g = g[np.isfinite(g)]
    report = {
        "gate_mean": float(np.mean(g)) if g.size else float("nan"),
        "gate_std": float(np.std(g)) if g.size else float("nan"),
        "gate_collapse": bool(g.size and float(np.std(g)) < 0.02),
        "n": int(len(picks)),
    }
    if "abcd_class" in picks.columns:
        report["gate_by_abcd"] = (
            picks.groupby("abcd_class")["gate"].agg(["mean", "std", "count"]).reset_index().to_dict(orient="records")
        )

    rows = []
    if "history_mad" in picks.columns:
        rows += _bin_stats(picks.dropna(subset=["history_mad"]), "history_mad", [-0.01, 0.2, 0.5, 1.0, 100], ["<0.2", "0.2-0.5", "0.5-1.0", ">1.0"])
    if "history_count" in picks.columns:
        rows += _bin_stats(picks, "history_count", [-0.1, 5, 10, 20, 1e9], ["<5", "5-9", "10-19", ">=20"])

    # calibration-ish: bin gate vs fraction where PhaseNet better
    if np.isfinite(g).any():
        qs = pd.qcut(picks["gate"].rank(method="first"), q=10, labels=False, duplicates="drop")
        cal = []
        for b in sorted(pd.Series(qs).dropna().unique()):
            sub = picks[qs == b]
            cal.append(
                {
                    "bin": int(b),
                    "gate_mean": float(sub["gate"].mean()),
                    "frac_phasenet_better": float(sub["pn_better_than_hist"].mean()),
                    "n": int(len(sub)),
                }
            )
        report["reliability_bins"] = cal
        # ECE proxy
        ece = float(np.mean([abs(c["gate_mean"] - c["frac_phasenet_better"]) for c in cal])) if cal else float("nan")
        report["ece_proxy"] = ece

    # agreement with oracle selector
    ora = []
    for _, r in picks.iterrows():
        e_pn = r["err_pn"]
        e_fr = r["err_fixed"]
        prefer_pn = e_pn <= e_fr
        # gate>0.5 => prefer pn
        agree = (r["gate"] >= 0.5) == prefer_pn if np.isfinite(r["gate"]) else False
        ora.append(agree)
    report["agree_oracle_selector_rate"] = float(np.mean(ora)) if ora else float("nan")
    pd.DataFrame(rows).to_csv(out / "gate_behavior_bins.csv", index=False)
    save_json(report, out / "gate_behavior.json")
    print(report, flush=True)


if __name__ == "__main__":
    main()
