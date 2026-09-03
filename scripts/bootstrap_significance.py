#!/usr/bin/env python
"""Event-level paired bootstrap for PhaseNet vs catalog candidate re-scoring."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_fixed.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--baseline-mode", default="phasenet")
    parser.add_argument("--improved-mode", default="catalog_rescore")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    n_boot = int(args.n_bootstrap or cfg.get("n_bootstrap", 2000))
    out = ensure_dir(artifacts_dir() / "results" / "stage2")

    picks = pd.read_parquet(out / "candidate_rescoring_picks.parquet")
    base = picks[picks["mode"] == args.baseline_mode][
        ["trace_name", "event_id", "pred_p_sample", "pred_s_sample", "true_p_sample", "true_s_sample", "sampling_rate_hz"]
    ].rename(columns={"pred_p_sample": "base_p", "pred_s_sample": "base_s"})
    imp = picks[picks["mode"] == args.improved_mode][["trace_name", "pred_p_sample", "pred_s_sample"]].rename(
        columns={"pred_p_sample": "imp_p", "pred_s_sample": "imp_s"}
    )
    df = base.merge(imp, on="trace_name", how="inner").reset_index(drop=True)

    events = df["event_id"].astype(str).to_numpy()
    uniq, inv = np.unique(events, return_inverse=True)
    # rows belonging to each event
    event_rows = [np.where(inv == i)[0] for i in range(len(uniq))]

    base_p = df["base_p"].to_numpy(dtype=np.float64)
    base_s = df["base_s"].to_numpy(dtype=np.float64)
    imp_p = df["imp_p"].to_numpy(dtype=np.float64)
    imp_s = df["imp_s"].to_numpy(dtype=np.float64)
    true_p = df["true_p_sample"].to_numpy(dtype=np.float64)
    true_s = df["true_s_sample"].to_numpy(dtype=np.float64)
    sr = df["sampling_rate_hz"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    keys = ["p_f1_0.1", "p_f1_0.5", "p_mae", "p_p95", "s_f1_0.1", "s_f1_0.5", "s_mae", "s_p95"]
    deltas = {k: np.empty(n_boot, dtype=np.float64) for k in keys}

    for b in range(n_boot):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([event_rows[i] for i in chosen])
        mb = match_picks(base_p[idx], true_p[idx], sr[idx])
        ms_b = match_picks(base_s[idx], true_s[idx], sr[idx])
        mp_i = match_picks(imp_p[idx], true_p[idx], sr[idx])
        ms_i = match_picks(imp_s[idx], true_s[idx], sr[idx])
        deltas["p_f1_0.1"][b] = mp_i["f1@0.1s"] - mb["f1@0.1s"]
        deltas["p_f1_0.5"][b] = mp_i["f1@0.5s"] - mb["f1@0.5s"]
        deltas["p_mae"][b] = mp_i["mae"] - mb["mae"]
        deltas["p_p95"][b] = mp_i["p95_ae"] - mb["p95_ae"]
        deltas["s_f1_0.1"][b] = ms_i["f1@0.1s"] - ms_b["f1@0.1s"]
        deltas["s_f1_0.5"][b] = ms_i["f1@0.5s"] - ms_b["f1@0.5s"]
        deltas["s_mae"][b] = ms_i["mae"] - ms_b["mae"]
        deltas["s_p95"][b] = ms_i["p95_ae"] - ms_b["p95_ae"]
        if (b + 1) % 200 == 0:
            print({"bootstrap": b + 1, "s_f1_0.5_mean": float(np.nanmean(deltas["s_f1_0.5"][: b + 1]))}, flush=True)

    report = {
        "baseline": args.baseline_mode,
        "improved": args.improved_mode,
        "n_bootstrap": n_boot,
        "n_events": int(len(uniq)),
        "n_traces": int(len(df)),
        "deltas": {},
    }
    for k, arr in deltas.items():
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        report["deltas"][k] = {
            "mean": float(np.nanmean(arr)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "probability_improved": float(np.mean(arr > 0)) if "f1" in k else float(np.mean(arr < 0)),
        }

    s_f1 = report["deltas"]["s_f1_0.5"]
    s_p95 = report["deltas"]["s_p95"]
    report["significant_s_f1_0.5"] = bool(s_f1["ci95_low"] > 0)
    report["significant_s_p95_reduction"] = bool(s_p95["ci95_high"] < 0)
    # gate decision inputs
    report["learned_gate_criteria"] = {
        "overall_s_f1_0.5_significant": report["significant_s_f1_0.5"],
        "overall_s_p95_reduction_significant": report["significant_s_p95_reduction"],
    }
    save_json(report, out / "bootstrap_significance.json")
    print(report, flush=True)


if __name__ == "__main__":
    main()
