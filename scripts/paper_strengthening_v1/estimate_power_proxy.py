#!/usr/bin/env python
"""Estimate detectable ΔF1@0.5 given event counts (paired event bootstrap proxy).

Uses heldout_eval event-level paired differences between E_scalar and main strong control
as an empirical effect distribution scale — for planning only, not a new claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.paper_strengthening import metrics_from_pred


def main() -> None:
    OUT = artifacts_dir() / "results" / "paper_strengthening_v1"
    held = pd.read_parquet(artifacts_dir() / "results/pairwise_pilot/pairs_ranker_train.parquet")
    held = held[held["split"] == "heldout_eval"].reset_index(drop=True)
    pred_e = np.load(OUT / "predictions" / "heldout_eval__E_scalar.npy")
    pred_c = np.load(OUT / "predictions" / "heldout_eval__C_best.npy")
    true = held["true_s_sample"].to_numpy(float)
    sr = held["sampling_rate_hz"].to_numpy(float)
    events = held["event_id"].astype(str).to_numpy()
    uniq = np.unique(events)
    # per-event F1 difference (approximate by restricting to that event's traces)
    deltas = []
    for e in uniq:
        m = events == e
        if m.sum() < 1:
            continue
        de = metrics_from_pred(pred_e[m], true[m], sr[m])["f1@0.5"]
        dc = metrics_from_pred(pred_c[m], true[m], sr[m])["f1@0.5"]
        deltas.append(de - dc)
    deltas = np.asarray(deltas, float)
    rng = np.random.default_rng(0)
    plan = {}
    for n_events in [500, 1000, 2000, 3000, 5000]:
        means = []
        for _ in range(2000):
            samp = rng.choice(deltas, size=min(n_events, len(deltas)), replace=True)
            # scale variance as if sampling new events with similar per-event noise
            # crude: mean of n_events draws from empirical per-event deltas
            means.append(samp.mean())
        means = np.asarray(means)
        plan[str(n_events)] = {
            "assumed_effect_like_heldout_mean_delta": float(deltas.mean()),
            "sim_mean_of_means": float(means.mean()),
            "sim_sd_of_means": float(means.std()),
            "frac_mean_gt_0": float((means > 0).mean()),
            "frac_mean_gt_0.003": float((means > 0.003).mean()),
            "note": "Resamples heldout per-event Δ; not a guarantee for INGV 2021-22.",
        }
    out = {
        "n_heldout_events": int(len(uniq)),
        "heldout_point_delta": float(
            metrics_from_pred(pred_e, true, sr)["f1@0.5"] - metrics_from_pred(pred_c, true, sr)["f1@0.5"]
        ),
        "per_event_delta_mean": float(deltas.mean()),
        "per_event_delta_std": float(deltas.std()),
        "plans": plan,
        "practical_scale": 0.003,
    }
    save_json(out, OUT / "dev" / "power_proxy_heldout.json")
    print(json.dumps(out, indent=2)[:1500])


if __name__ == "__main__":
    main()
