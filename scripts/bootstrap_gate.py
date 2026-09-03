#!/usr/bin/env python
"""Event-level paired bootstrap for Stage-3 learned gate."""

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


def _bootstrap(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int, seed: int) -> dict:
    events = df["event_id"].astype(str).to_numpy()
    uniq, inv = np.unique(events, return_inverse=True)
    event_rows = [np.where(inv == i)[0] for i in range(len(uniq))]
    a = df[col_a].to_numpy(dtype=np.float64)
    b = df[col_b].to_numpy(dtype=np.float64)
    true = df["true_s_sample"].to_numpy(dtype=np.float64)
    sr = df["sampling_rate_hz"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    keys = ["delta_f1_0.1", "delta_f1_0.5", "delta_e2e_mae", "delta_e2e_p95"]
    deltas = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([event_rows[j] for j in chosen])
        ma = match_picks(a[idx], true[idx], sr[idx])
        mb = match_picks(b[idx], true[idx], sr[idx])
        # b improved over a => positive delta_f1 for (b-a)
        deltas["delta_f1_0.1"][i] = mb["f1@0.1s"] - ma["f1@0.1s"]
        deltas["delta_f1_0.5"][i] = mb["f1@0.5s"] - ma["f1@0.5s"]
        deltas["delta_e2e_mae"][i] = mb["mae"] - ma["mae"]
        deltas["delta_e2e_p95"][i] = mb["p95_ae"] - ma["p95_ae"]
    out = {}
    for k, arr in deltas.items():
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        if "f1" in k:
            p_imp = float(np.mean(arr > 0))
        else:
            p_imp = float(np.mean(arr < 0))
        out[k] = {"mean": float(np.nanmean(arr)), "ci95_low": float(lo), "ci95_high": float(hi), "probability_improved": p_imp}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_catalog.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--picks", default="artifacts/results/stage3/learned_gate_picks_test.parquet")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    n_boot = int(args.n_bootstrap or cfg.get("n_bootstrap", 2000))
    out = ensure_dir(artifacts_dir() / "results" / "stage3")
    picks = pd.read_parquet(ROOT / args.picks)
    # use one seed (first) or mean picks — prefer seed matching config
    if "seed" in picks.columns:
        seeds = sorted(picks.seed.unique())
        use = int(args.seed) if int(args.seed) in set(seeds) else int(seeds[0])
        df = picks[picks.seed == use].copy()
    else:
        df = picks.copy()
        use = args.seed

    report = {
        "seed_used": use,
        "n_bootstrap": n_boot,
        "n_traces": int(len(df)),
        "n_events": int(df.event_id.nunique()),
        "comparisons": {},
    }
    pairs = [
        ("phasenet", "pred_s_phasenet", "learned_gate", "pred_s_learned_gate"),
        ("fixed_catalog_rescore", "pred_s_fixed_rescore", "learned_gate", "pred_s_learned_gate"),
    ]
    if "pred_s_direct_ranker" in df.columns:
        pairs.append(("learned_gate", "pred_s_learned_gate", "direct_ranker", "pred_s_direct_ranker"))

    for a_name, a_col, b_name, b_col in pairs:
        key = f"{b_name}_vs_{a_name}"
        report["comparisons"][key] = _bootstrap(df, a_col, b_col, n_boot, seed=int(cfg.get("seed", 42)))
        print(key, report["comparisons"][key]["delta_f1_0.5"], flush=True)

    # superiority criteria vs fixed
    vs = report["comparisons"].get("learned_gate_vs_fixed_catalog_rescore", {})
    d05 = vs.get("delta_f1_0.5", {})
    d01 = vs.get("delta_f1_0.1", {})
    dp95 = vs.get("delta_e2e_p95", {})
    beats = False
    reasons = []
    if d05 and d05.get("ci95_low", -1) > 0:
        beats = True
        reasons.append("delta_f1_0.5_ci_all_positive")
    if d05 and d05.get("ci95_low", -1) >= -0.005 and d01 and d01.get("ci95_low", -1) > 0:
        beats = True
        reasons.append("noninferior_f1_0.5_and_f1_0.1_ci_positive")
    if d05 and d05.get("ci95_low", -1) >= -0.005 and dp95 and dp95.get("ci95_high", 1) < 0:
        beats = True
        reasons.append("noninferior_f1_and_p95_drop")
    report["learned_gate_beats_fixed_rescore"] = beats
    report["reasons"] = reasons
    if not beats:
        report["recommendation"] = "Keep fixed catalog_rescore as main method; report learned gate as ablation."
    else:
        report["recommendation"] = "Learned gate may be preferred; still report fixed rescore baseline."
    save_json(report, out / "bootstrap_gate.json")
    print({"beats_fixed": beats, "reasons": reasons}, flush=True)


if __name__ == "__main__":
    main()
