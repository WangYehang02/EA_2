#!/usr/bin/env python
"""Build Phase C union caches + labeled feature tables for ranker_train and dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage6.ranker.features import assemble_feature_table, feature_names
from earthquake.stage6.ranker.labels import attach_analysis_classes, label_union_candidates
from earthquake.stage6.ranker.schema import CANDIDATE_SCHEMA, schema_sha256
from earthquake.stage6.ranker.union_schema import union_candidates_phaseC
from earthquake.utils import ensure_dir


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_dataset")
        raise SystemExit("method_lock present")
    except RuntimeError:
        pass


def build_side(
    *,
    name: str,
    stead_path: Path,
    ida_path: Path,
    meta: pd.DataFrame,
    hist: pd.DataFrame,
    global_res: dict,
    out_dir: Path,
) -> dict:
    stead = pd.read_parquet(stead_path)
    ida = pd.read_parquet(ida_path)
    # restrict to meta traces
    mset = set(meta["trace_name"].astype(str))
    stead = stead[stead["trace_name"].astype(str).isin(mset)].copy()
    ida = ida[ida["trace_name"].astype(str).isin(mset)].copy()
    union = union_candidates_phaseC(stead, ida, stead_k=5, ida_k=5, max_union=10)
    assert int(union.groupby("trace_name").size().max()) <= 10
    labeled = label_union_candidates(union)
    labeled = attach_analysis_classes(labeled)
    # attach meta columns needed for features
    keep = [
        c
        for c in [
            "trace_name",
            "event_id",
            "source_depth_km",
            "distance_km",
            "hyp_distance_km",
            "azimuth_deg",
            "station_elevation_m",
            "s_arrival_sample",
            "sampling_rate_hz",
        ]
        if c in meta.columns
    ]
    labeled = labeled.drop(columns=[c for c in ["source_depth_km", "distance_km"] if c in labeled.columns], errors="ignore")
    labeled = labeled.merge(meta[keep], on=["trace_name"], how="left", suffixes=("", "_m"))

    union_path = out_dir / f"{name}_union.parquet"
    labeled.to_parquet(union_path, index=False)

    feat_r1 = assemble_feature_table(labeled, hist, meta, global_res=global_res, variant="R1")
    feat_r2 = assemble_feature_table(labeled, hist, meta, global_res=global_res, variant="R2")
    feat_r1.to_parquet(out_dir / f"{name}_features_R1.parquet", index=False)
    feat_r2.to_parquet(out_dir / f"{name}_features_R2.parquet", index=False)

    # class rates
    tr = labeled.drop_duplicates("trace_name")
    class_rates = {}
    for cls in [
        "easy_top1_correct",
        "recoverable_stead_wrong",
        "ida_only_recoverable",
        "stead_only_recoverable",
        "both_models_recoverable",
        "none_of_k",
        "multi_peak",
    ]:
        class_rates[cls] = float(tr["analysis_classes"].astype(str).str.contains(cls).mean())

    return {
        "name": name,
        "n_traces": int(tr.shape[0]),
        "n_candidates": int(len(labeled)),
        "mean_n_cand": float(tr["candidate_count"].mean()),
        "max_n_cand": int(tr["candidate_count"].max()),
        "none_of_k_rate": float(tr["label_none_of_k"].mean()),
        "class_rates": class_rates,
        "union_sha256": sha256_file(union_path),
        "schema_sha256": schema_sha256(),
        "schema_version": CANDIDATE_SCHEMA["version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-manifest", default="artifacts/results/stage6/history_picker_train_manifest.json")
    parser.add_argument("--history-features", default="artifacts/models/stage6/history_picker_train/residual_history_features.parquet")
    parser.add_argument("--dev-manifest", default="artifacts/results/stage6/phaseB_eval_manifest.csv")
    parser.add_argument("--train-manifest", default="artifacts/results/stage6/phaseC/ranker_train_s_manifest.csv")
    args = parser.parse_args()
    _guard()

    hist_man = load_json(ROOT / args.history_manifest)
    global_res = hist_man["global_residual"]
    hist = pd.read_parquet(ROOT / args.history_features)
    # drop audit subset name noise — keep all rows keyed by trace
    hist = hist.drop_duplicates("trace_name", keep="last")

    out = ensure_dir(artifacts_dir() / "cache" / "stage6" / "phaseC")
    results = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC")

    train_meta = pd.read_csv(ROOT / args.train_manifest)
    dev_meta = pd.read_csv(ROOT / args.dev_manifest)

    summaries = {}
    summaries["ranker_train"] = build_side(
        name="ranker_train",
        stead_path=ROOT / "artifacts/cache/stage6/phaseC/ranker_train_stead_top10/stead_top10.parquet",
        ida_path=ROOT / "artifacts/cache/stage6/phaseC/ranker_train_ida_top10/ida_top10.parquet",
        meta=train_meta,
        hist=hist[hist["subset"] == "stage6_ranker_train"] if "subset" in hist.columns else hist,
        global_res=global_res,
        out_dir=out,
    )
    summaries["dev"] = build_side(
        name="dev",
        stead_path=ROOT / "artifacts/cache/stage6/phaseB/stead_top10/stead_top10.parquet",
        ida_path=ROOT / "artifacts/cache/stage6/phaseB/ida_top10/ida_top10.parquet",
        meta=dev_meta,
        hist=hist[hist["subset"] == "stage6_dev"] if "subset" in hist.columns else hist,
        global_res=global_res,
        out_dir=out,
    )
    save_json(summaries, results / "ranker_dataset_summary.json")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
