#!/usr/bin/env python
"""Fit Stage-6 picker_train-only distance MLP + TemporalHistoryStore snapshot.

No ranker_train/dev/confirm events update the store. Shrinkage_k frozen at 50 (method lock).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.history.residual_prior import attach_baseline_and_residuals, path_stats_to_residual_stats
from earthquake.history.temporal_store import TemporalHistoryStore
from earthquake.history.travel_time_baseline import fit_mlp
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, load_full_event_ids, load_full_trace_names
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir

SHRINKAGE_K = 50.0  # frozen from Stage-4 method lock; do not re-search on Stage-6
GRID_SIZE = 0.2
MIN_HISTORY = 5


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_history")
        raise SystemExit("method_lock present — refuse")
    except RuntimeError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/index_full/events.parquet")
    parser.add_argument("--out-dir", default="artifacts/models/stage6/history_picker_train")
    args = parser.parse_args()
    _guard()

    out = ensure_dir(ROOT / args.out_dir)
    events = pd.read_parquet(ROOT / args.index)
    pe = set(str(x) for x in load_full_event_ids("stage6_picker_train"))
    re = set(str(x) for x in load_full_event_ids("stage6_ranker_train"))
    de = set(str(x) for x in load_full_event_ids("stage6_dev"))
    ce = set(str(x) for x in load_full_event_ids("stage6_internal_confirm"))
    assert pe.isdisjoint(re) and pe.isdisjoint(de) and pe.isdisjoint(ce)
    assert re.isdisjoint(de) and re.isdisjoint(ce) and de.isdisjoint(ce)

    picker = events[events["event_id"].astype(str).isin(pe)].copy()
    # PS both for MLP fit
    p = pd.to_numeric(picker["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(picker["s_arrival_sample"], errors="coerce")
    picker_ps = picker.loc[p.notna() & s.notna()].copy()
    # Fit MLP on stratified sample for speed (still picker_train-only; no ranker/dev leak)
    n_fit = min(80000, len(picker_ps))
    picker_fit = picker_ps.sample(n=n_fit, random_state=42) if len(picker_ps) > n_fit else picker_ps
    print({"mlp_fit_n": int(len(picker_fit)), "picker_ps_n": int(len(picker_ps))}, flush=True)
    baseline = fit_mlp(picker_fit, seed=42)
    with open(out / "travel_time_baseline_mlp.pkl", "wb") as f:
        pickle.dump(baseline, f)
    print({"mlp_fit_done": True}, flush=True)

    # Global residual from larger sample (still picker_train only)
    picker_aug = attach_baseline_and_residuals(picker_ps.sample(n=min(100000, len(picker_ps)), random_state=0), baseline)
    print({"global_residual_done": True}, flush=True)
    global_res = {
        "residual_p_median": float(picker_aug["residual_tau_p"].median(skipna=True)),
        "residual_s_median": float(picker_aug["residual_tau_s"].median(skipna=True)),
        "residual_sp_median": float(picker_aug["residual_delta_sp"].median(skipna=True)),
    }

    # Fill TemporalHistoryStore with picker_train PS-both only (time order). Faster than all traces.
    store = TemporalHistoryStore(grid_size=GRID_SIZE, min_history=MIN_HISTORY)
    picker_ps_sorted = picker_ps.sort_values(["origin_time", "event_id", "trace_name"])
    # Precompute relative times to avoid repeated pandas work in update_row
    n_upd = 0
    for eid, g in tqdm(picker_ps_sorted.groupby("event_id", sort=False), desc="history_update_picker_ps"):
        store.update_event(g)
        n_upd += len(g)
        if n_upd % 20000 < len(g):
            print({"updated_traces": n_upd, "n_paths": store.n_paths()}, flush=True)
    print({"history_update_done": n_upd, "n_paths": store.n_paths()}, flush=True)

    # Query only S-labelled ranker_train + stage6_dev (never update store).
    # Picker coverage: sample 5k S-labelled picker traces for audit only.
    def _s_labelled(df: pd.DataFrame) -> pd.DataFrame:
        ss = pd.to_numeric(df["s_arrival_sample"], errors="coerce")
        return df.loc[ss.notna()].copy()

    targets = {
        "stage6_ranker_train": _s_labelled(events[events["event_id"].astype(str).isin(re)]),
        "stage6_dev": _s_labelled(events[events["event_id"].astype(str).isin(de)]),
        "stage6_picker_train_audit": _s_labelled(picker).sample(n=min(5000, len(picker_ps)), random_state=42),
    }
    feat_rows = []
    coverage = {}
    for subset, sub in targets.items():
        sub = sub.sort_values(["origin_time", "event_id", "trace_name"]).reset_index(drop=True)
        # vectorized baseline predictions
        base_df = baseline.predict_frame(sub)
        n_ok = 0
        for i, row in tqdm(sub.iterrows(), total=len(sub), desc=f"query_{subset}"):
            stats = store.query_row(row)
            base = {
                "base_tau_p": float(base_df.loc[i, "base_tau_p"]),
                "base_tau_s": float(base_df.loc[i, "base_tau_s"]),
                "base_delta_sp": float(base_df.loc[i, "base_delta_sp"]),
            }
            rs = path_stats_to_residual_stats(stats, base)
            if rs.history_available:
                n_ok += 1
            feat_rows.append(
                {
                    "subset": subset,
                    "trace_name": str(row["trace_name"]),
                    "event_id": str(row["event_id"]),
                    "base_tau_p": base["base_tau_p"],
                    "base_tau_s": base["base_tau_s"],
                    "base_delta_sp": base["base_delta_sp"],
                    "history_count": rs.history_count,
                    "history_available": bool(rs.history_available),
                    "residual_p_median": rs.residual_p_median,
                    "residual_p_mad": rs.residual_p_mad,
                    "residual_s_median": rs.residual_s_median,
                    "residual_s_mad": rs.residual_s_mad,
                    "residual_sp_median": rs.residual_sp_median,
                    "residual_sp_mad": rs.residual_sp_mad,
                    "tau_p_median": rs.tau_p_median,
                    "tau_s_median": rs.tau_s_median,
                    "delta_sp_median": rs.delta_sp_median,
                    "tau_p_mad": rs.tau_p_mad,
                    "tau_s_mad": rs.tau_s_mad,
                    "delta_sp_mad": rs.delta_sp_mad,
                    "matched_key": rs.matched_key,
                    "fallback_level": rs.fallback_level,
                    "shrinkage_k": SHRINKAGE_K,
                }
            )
        coverage[subset] = {
            "n_traces": int(len(sub)),
            "n_history_available": int(n_ok),
            "frac_history_available": float(n_ok / max(len(sub), 1)),
        }

    feat = pd.DataFrame(feat_rows)
    feat_path = out / "residual_history_features.parquet"
    feat.to_parquet(feat_path, index=False)

    # Serialize store path counts (not full obs for size) + pickle store for audit
    with open(out / "temporal_history_store.pkl", "wb") as f:
        pickle.dump(store, f)

    manifest = {
        "protocol": "picker_train_only_frozen_snapshot",
        "shrinkage_k": SHRINKAGE_K,
        "shrinkage_k_researched_on_dev": False,
        "grid_size": GRID_SIZE,
        "min_history": MIN_HISTORY,
        "baseline_kind": baseline.kind,
        "n_picker_train_events": len(pe),
        "n_picker_ps_traces_for_mlp": int(len(picker_ps)),
        "global_residual": global_res,
        "coverage": coverage,
        "event_disjoint": True,
        "confirm_excluded": True,
        "ranker_dev_never_update_store": True,
        "features_path": str(feat_path),
        "features_sha256": sha256_file(feat_path),
        "baseline_sha256": sha256_file(out / "travel_time_baseline_mlp.pkl"),
        "lambdas_s_frozen": {"lw": 0.5, "lh": 2.0, "lp": 0.0},
        "note": "Do not use Stage2/3 history artifacts for Phase C.",
    }
    save_json(manifest, out / "manifest.json")
    save_json(manifest, artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    print(json.dumps({"coverage": coverage, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
