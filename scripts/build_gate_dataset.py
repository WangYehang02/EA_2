#!/usr/bin/env python
"""Build Stage-3 gate train/val/test candidate caches (zarr) + PhaseNet parquet cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.fusion.peak_candidates import extract_candidates, candidates_to_records
from earthquake.gating.cache_io import (
    abcd_class,
    attach_expected_s,
    build_cand_index,
    file_hash,
    fixed_rescore_s,
    phasenet_s_pick,
    sample_traces_by_event,
    save_gate_zarr,
)
from earthquake.gating.features import (
    FEATURE_COLUMNS_BLIND,
    FEATURE_COLUMNS_CATALOG,
    assert_no_forbidden_columns,
    build_trace_features,
    fit_scaler,
    save_scaler,
)
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir


def _ensure_phasenet_cache(
    traces: pd.DataFrame,
    *,
    cache_path: Path,
    cands_path: Path,
    cfg: dict,
    device: str,
    reuse_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume PhaseNet picks/candidates; reuse existing stage2 caches when possible."""
    ensure_dir(cache_path.parent)
    pick_rows: list[dict] = []
    cand_rows: list[dict] = []
    done: set[str] = set()

    # seed from reuse caches
    for rp in reuse_paths:
        cp = Path(rp)
        cdp = cp.with_name(cp.stem + "_candidates.parquet")
        if cp.exists():
            prev = pd.read_parquet(cp)
            pick_rows.extend(prev.to_dict(orient="records"))
            done |= set(prev["trace_name"].astype(str))
        if cdp.exists():
            cand_rows.extend(pd.read_parquet(cdp).to_dict(orient="records"))

    if cache_path.exists():
        prev = pd.read_parquet(cache_path)
        # replace overlapping
        names = set(prev["trace_name"].astype(str))
        pick_rows = [r for r in pick_rows if str(r["trace_name"]) not in names] + prev.to_dict(orient="records")
        done |= names
        if cands_path.exists():
            prev_c = pd.read_parquet(cands_path)
            # drop old cands for these traces then append
            keep = [r for r in cand_rows if str(r["trace_name"]) not in names]
            cand_rows = keep + prev_c.to_dict(orient="records")

    need = traces[~traces["trace_name"].astype(str).isin(done)].copy()
    # Also recompute if K insufficient in existing cands
    k_need = int(cfg.get("candidate_k_build", cfg.get("candidate_k", 5)))
    if cand_rows and len(done):
        cdf = pd.DataFrame(cand_rows)
        s_counts = cdf[cdf["phase"] == "s"].groupby("trace_name").size()
        # if we need K=10 but only have <=5 for many, leave as-is for traces already cached at stage2 K=5;
        # for oracle we may re-run missing higher-K only on target set later.
        _ = (s_counts, k_need)

    thr = float(cfg.get("pick_threshold", 0.3))
    if len(need) > 0:
        ref = SeisBenchPhaseNetReference(weight=cfg.get("phasenet_weight", "stead"), device=device)
        h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
        pending_picks: list[dict] = []
        pending_cands: list[dict] = []
        flush_every = int(cfg.get("flush_every", 100))
        with InstanceHDF5Reader(h5) as reader:
            for i, (_, row) in enumerate(tqdm(need.iterrows(), total=len(need), desc="phasenet-stage3", file=sys.stderr)):
                wave = reader.read_waveform(str(row.trace_name))
                pred = ref.predict_row(wave, row, remap_to_waveform=True)
                p_pick = pick_from_prob(pred["p"], threshold=thr)
                s_pick = pick_from_prob(pred["s"], threshold=thr)
                p_cands = extract_candidates(
                    pred["p"],
                    phase="p",
                    k=int(cfg.get("candidate_k", 5)),
                    min_distance=int(cfg.get("min_peak_distance", 50)),
                    min_prominence=float(cfg.get("min_peak_prominence", 0.05)),
                    min_probability=float(cfg.get("min_peak_probability", 0.1)),
                    sampling_rate=float(row.sampling_rate_hz),
                    waveform_starttime=row.trace_start_time,
                )
                s_cands = extract_candidates(
                    pred["s"],
                    phase="s",
                    k=max(int(cfg.get("candidate_k", 5)), max(cfg.get("candidate_k_grid", [5]))),
                    min_distance=int(cfg.get("min_peak_distance", 50)),
                    min_prominence=float(cfg.get("min_peak_prominence", 0.05)),
                    min_probability=float(cfg.get("min_peak_probability", 0.1)),
                    sampling_rate=float(row.sampling_rate_hz),
                    waveform_starttime=row.trace_start_time,
                )
                pending_picks.append(
                    {
                        "trace_name": str(row.trace_name),
                        "event_id": str(row.event_id),
                        "sampling_rate_hz": float(row.sampling_rate_hz),
                        "true_p_sample": float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan,
                        "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                        "pred_p_sample": p_pick["peak_sample"],
                        "pred_s_sample": s_pick["peak_sample"],
                        "p_peak_probability": p_pick["peak_probability"],
                        "s_peak_probability": s_pick["peak_probability"],
                        "p_n_cands": len(p_cands),
                        "s_n_cands": len(s_cands),
                    }
                )
                for c in p_cands + s_cands:
                    pending_cands.append(
                        {
                            "trace_name": str(row.trace_name),
                            "event_id": str(row.event_id),
                            "phase": c.phase,
                            "sample_index": c.sample_index,
                            "peak_probability": c.peak_probability,
                            "prominence": c.prominence,
                            "peak_width": c.peak_width,
                            "local_entropy": c.local_entropy,
                            "rank": c.rank,
                            "fallback_peak": c.fallback_peak,
                        }
                    )
                if (i + 1) % flush_every == 0 or (i + 1) == len(need):
                    pick_rows.extend(pending_picks)
                    cand_rows.extend(pending_cands)
                    # keep only target traces in stage3 cache files
                    tgt = set(traces["trace_name"].astype(str))
                    pdf = pd.DataFrame(pick_rows)
                    cdf = pd.DataFrame(cand_rows)
                    pdf = pdf[pdf.trace_name.astype(str).isin(tgt)].drop_duplicates("trace_name", keep="last")
                    cdf = cdf[cdf.trace_name.astype(str).isin(tgt)]
                    # for cands: if duplicate ranks from reuse+new, keep last batch — drop_duplicates
                    cdf = cdf.drop_duplicates(["trace_name", "phase", "rank"], keep="last")
                    pdf.to_parquet(cache_path, index=False)
                    cdf.to_parquet(cands_path, index=False)
                    pick_rows = pdf.to_dict(orient="records")
                    cand_rows = cdf.to_dict(orient="records")
                    pending_picks.clear()
                    pending_cands.clear()
                    print({"flushed_stage3_cache": len(pdf), "target": len(traces)}, flush=True)

    tgt = set(traces["trace_name"].astype(str))
    picks = pd.DataFrame(pick_rows)
    cands = pd.DataFrame(cand_rows)
    if len(picks) == 0:
        raise SystemExit("No PhaseNet cache rows")
    picks = picks[picks.trace_name.astype(str).isin(tgt)].drop_duplicates("trace_name", keep="last")
    cands = cands[cands.trace_name.astype(str).isin(tgt)].drop_duplicates(["trace_name", "phase", "rank"], keep="last")
    picks.to_parquet(cache_path, index=False)
    cands.to_parquet(cands_path, index=False)
    return picks, cands


def _hard_score(row: pd.Series) -> float:
    """Higher = harder for oversampling on train only."""
    s = 0.0
    sp = float(row.get("s_peak_probability", np.nan))
    if np.isfinite(sp) and sp < 0.3:
        s += 2.0
    n = float(row.get("s_n_cands", 0) or 0)
    if n >= 3:
        s += 1.5
    mad = float(row.get("residual_s_mad", np.nan))
    if np.isfinite(mad) and mad < 0.2:
        s += 1.0  # history confident — useful for C cases
    return s


def build_split_records(
    meta: pd.DataFrame,
    cand_index: dict,
    *,
    mode: str,
    max_k: int,
    global_res: dict,
    cfg: dict,
    lambdas: dict,
) -> list[dict]:
    cols = FEATURE_COLUMNS_CATALOG if mode == "catalog" else FEATURE_COLUMNS_BLIND
    records = []
    lw, lh, lp = float(lambdas["lw"]), float(lambdas["lh"]), float(lambdas["lp"])
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="features"):
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        p_c = cand_index.get((str(row.trace_name), "p"), [])
        exp = attach_expected_s(
            row,
            global_res=global_res,
            shrink_k=float(cfg.get("shrinkage_k", 50)),
            min_history=int(cfg.get("min_history", 5)),
            mad_disable_s=float(cfg.get("mad_threshold", 1.0)),
            min_sigma_s=float(cfg.get("min_sigma_s", 0.05)),
            max_sigma_s=float(cfg.get("max_sigma_s", 1.0)),
        )
        hist_avail = bool(exp["gate_history_available"]) and bool(row.get("history_available", False))
        # if expected nan, force unavailable
        if not np.isfinite(exp["expected_s_sample"]):
            hist_avail = False

        s_dicts = [
            {
                "sample_index": c.sample_index,
                "peak_probability": c.peak_probability,
                "prominence": c.prominence,
                "peak_width": c.peak_width,
                "local_entropy": c.local_entropy,
                "rank": c.rank,
                "fallback_peak": c.fallback_peak,
            }
            for c in s_c[:max_k]
        ]
        # p peak extras on row
        row2 = row.copy()
        if p_c:
            row2["p_peak_probability"] = p_c[0].peak_probability
            row2["p_peak_entropy"] = p_c[0].local_entropy
        row2["pred_p_sample"] = float(row.get("pred_p_sample", np.nan))
        row2["shrinkage_k"] = float(cfg.get("shrinkage_k", 50))
        feats = build_trace_features(
            row2,
            s_dicts,
            expected_s_sample=exp["expected_s_sample"],
            history_sigma_samples=exp["history_sigma_samples"],
            sampling_rate=float(row.sampling_rate_hz),
            mode=mode,
        )
        feats["history_available"] = 1.0 if hist_avail else 0.0

        pn = phasenet_s_pick(s_c, float(row.get("pred_s_sample", np.nan)))
        fr = fixed_rescore_s(
            s_c,
            expected_s=exp["expected_s_sample"],
            sigma_samples=exp["history_sigma_samples"],
            history_available=hist_avail,
            lw=lw,
            lh=lh,
            lp=lp,
        )
        true_s = float(row["true_s_sample"]) if pd.notna(row.get("true_s_sample", np.nan)) else float("nan")
        sr = float(row.sampling_rate_hz)
        pn_err = abs(pn - true_s) / sr if np.isfinite(pn) and np.isfinite(true_s) else float("nan")
        # history pick = argmin |c - expected|
        if hist_avail and s_c and np.isfinite(exp["expected_s_sample"]):
            hpick = min(s_c, key=lambda c: abs(c.sample_index - exp["expected_s_sample"])).sample_index
            herr = abs(hpick - true_s) / sr if np.isfinite(true_s) else float("nan")
        else:
            herr = float("nan")

        records.append(
            {
                "feats": feats,
                "s_cands": s_dicts,
                "expected_s_sample": exp["expected_s_sample"],
                "history_sigma_samples": exp["history_sigma_samples"],
                "history_available": hist_avail,
                "true_s_sample": true_s,
                "true_p_sample": float(row.get("true_p_sample", np.nan)),
                "pred_p_sample": float(row.get("pred_p_sample", np.nan)),
                "pred_s_phasenet": pn,
                "pred_s_fixed_rescore": fr,
                "history_count": float(row.get("history_count", 0) or 0),
                "history_mad": float(row.get("residual_s_mad", np.nan)),
                "fallback_level": int(row.get("fallback_level", -1)) if pd.notna(row.get("fallback_level", np.nan)) else -1,
                "sampling_rate": sr,
                "trace_name": str(row.trace_name),
                "event_id": str(row.event_id),
                "abcd_class": abcd_class(pn_err, herr),
                "split": str(row.get("split", "")),
                "feature_columns": cols,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_debug.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    mode = str(cfg.get("mode", "catalog"))
    device = args.device or cfg.get("device", "cuda")
    seed = int(cfg.get("seed", 42))
    max_k = int(cfg.get("candidate_k", 5))
    out_cand = ensure_dir(artifacts_dir() / "candidates")
    out_model = ensure_dir(artifacts_dir() / "models" / "learned_gate")
    out_res = ensure_dir(artifacts_dir() / "results" / "stage3")

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    resid = pd.read_parquet(artifacts_dir() / "history" / "residual_history_features_frozen.parquet")
    # join meta columns needed for expected times
    need_cols = [
        "trace_name",
        "event_id",
        "split",
        "origin_time",
        "trace_start_time",
        "sampling_rate_hz",
        "p_arrival_sample",
        "s_arrival_sample",
        "distance_km",
        "source_depth_km",
        "station_elevation_m",
        "snr_db",
        "network",
        "station",
        "location",
        "channel_prefix",
        "station_id",
    ]
    meta_all = events[[c for c in need_cols if c in events.columns]].copy()
    meta_all = meta_all.merge(resid, on=["trace_name", "event_id"], how="left", suffixes=("", "_r"))
    if "split" not in meta_all.columns or meta_all["split"].isna().all():
        meta_all["split"] = meta_all.get("split_r", meta_all["split"])

    test_list = ROOT / cfg.get("test_list", "artifacts/diagnostics/fixed_eval_test_only_traces.txt")
    test_names = [ln.strip() for ln in test_list.read_text().splitlines() if ln.strip()]
    n_test = int(cfg.get("n_test_traces", len(test_names)))
    test_df = meta_all[meta_all.trace_name.astype(str).isin(test_names)].copy()
    # preserve list order then head
    order = {n: i for i, n in enumerate(test_names)}
    test_df["_ord"] = test_df.trace_name.astype(str).map(order)
    test_df = test_df.sort_values("_ord").head(n_test).drop(columns=["_ord"])

    train_pool = meta_all[meta_all["split"] == "train"].copy()
    val_pool = meta_all[meta_all["split"] == "val"].copy()
    # event disjoint from test
    te_events = set(test_df.event_id.astype(str))
    train_pool = train_pool[~train_pool.event_id.astype(str).isin(te_events)]
    val_pool = val_pool[~val_pool.event_id.astype(str).isin(te_events)]

    # provisional hard score from residual only (refined after cache)
    train_pool["hard_score"] = train_pool.apply(_hard_score, axis=1)
    train_df = sample_traces_by_event(
        train_pool,
        n_traces=int(cfg.get("n_train_traces", 2000)),
        max_per_event=int(cfg.get("max_traces_per_event", 5)),
        seed=seed,
        prefer_hard=True,
    )
    val_df = sample_traces_by_event(
        val_pool,
        n_traces=int(cfg.get("n_val_traces", 1000)),
        max_per_event=int(cfg.get("max_traces_per_event", 5)),
        seed=seed + 7,
        prefer_hard=False,
    )

    # ensure event disjoint across train/val
    tr_e = set(train_df.event_id.astype(str))
    val_df = val_df[~val_df.event_id.astype(str).isin(tr_e)]
    if len(val_df) < int(cfg.get("n_val_traces", 1000)) * 0.5:
        # refill from remaining val pool
        rem = val_pool[~val_pool.event_id.astype(str).isin(tr_e | set(val_df.event_id.astype(str)))]
        extra = sample_traces_by_event(
            rem,
            n_traces=int(cfg.get("n_val_traces", 1000)) - len(val_df),
            max_per_event=int(cfg.get("max_traces_per_event", 5)),
            seed=seed + 9,
        )
        val_df = pd.concat([val_df, extra], ignore_index=True)

    all_need = pd.concat([train_df, val_df, test_df], ignore_index=True).drop_duplicates("trace_name")
    print(
        {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_unique_cache_target": len(all_need),
            "train_events": train_df.event_id.nunique(),
            "val_events": val_df.event_id.nunique(),
            "test_events": test_df.event_id.nunique(),
        },
        flush=True,
    )

    cache_path = out_cand / "stage3_phasenet_cache.parquet"
    cands_path = out_cand / "stage3_phasenet_cache_candidates.parquet"
    reuse = [
        artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache.parquet",
        cache_path,
    ]
    picks, cands = _ensure_phasenet_cache(
        all_need,
        cache_path=cache_path,
        cands_path=cands_path,
        cfg=cfg,
        device=device,
        reuse_paths=reuse,
    )

    # merge picks into meta
    pick_cols = ["trace_name", "pred_p_sample", "pred_s_sample", "p_peak_probability", "s_peak_probability", "p_n_cands", "s_n_cands", "true_p_sample", "true_s_sample"]
    picks = picks[[c for c in pick_cols if c in picks.columns]]

    def _join(split_df: pd.DataFrame) -> pd.DataFrame:
        m = split_df.merge(picks, on="trace_name", how="inner", suffixes=("", "_pk"))
        for c in ["true_p_sample", "true_s_sample"]:
            if c + "_pk" in m.columns:
                m[c] = m[c].fillna(m[c + "_pk"])
        # prefer index labels
        if "p_arrival_sample" in m.columns:
            m["true_p_sample"] = m["p_arrival_sample"]
        if "s_arrival_sample" in m.columns:
            m["true_s_sample"] = m["s_arrival_sample"]
        return m

    train_m = _join(train_df)
    val_m = _join(val_df)
    test_m = _join(test_df)
    cand_index = build_cand_index(cands)

    rh_meta = load_json(artifacts_dir() / "results" / "stage2" / "residual_history_meta.json")
    global_res = rh_meta["global_residual"]
    best = load_json(artifacts_dir() / "results" / "stage2" / "best_lambdas.json")
    lambdas = {"lw": best["best_s"]["lw"], "lh": best["best_s"]["lh"], "lp": best["best_s"]["lp"]}

    train_rec = build_split_records(train_m, cand_index, mode=mode, max_k=max_k, global_res=global_res, cfg=cfg, lambdas=lambdas)
    val_rec = build_split_records(val_m, cand_index, mode=mode, max_k=max_k, global_res=global_res, cfg=cfg, lambdas=lambdas)
    test_rec = build_split_records(test_m, cand_index, mode=mode, max_k=max_k, global_res=global_res, cfg=cfg, lambdas=lambdas)

    cols = FEATURE_COLUMNS_CATALOG if mode == "catalog" else FEATURE_COLUMNS_BLIND
    assert_no_forbidden_columns(cols)
    feat_df = pd.DataFrame([r["feats"] for r in train_rec])
    scaler = fit_scaler(feat_df, cols)
    save_scaler(scaler, out_model / "feature_scaler.pkl")

    def _apply(recs):
        out = []
        fdf = pd.DataFrame([r["feats"] for r in recs])
        X, M = scaler.transform(fdf)
        for i, r in enumerate(recs):
            rr = dict(r)
            rr["x"] = X[i]
            rr["missing"] = M[i]
            out.append(rr)
        return out

    train_rec = _apply(train_rec)
    val_rec = _apply(val_rec)
    test_rec = _apply(test_rec)

    tag = "debug" if "debug" in Path(args.config).stem else mode
    z_train = out_cand / f"stage3_train.zarr"
    z_val = out_cand / f"stage3_val.zarr"
    z_test = out_cand / f"stage3_test.zarr"
    save_gate_zarr(z_train, train_rec, cols, max_k)
    save_gate_zarr(z_val, val_rec, cols, max_k)
    save_gate_zarr(z_test, test_rec, cols, max_k)

    # also store pick baselines for eval
    def _base_df(recs, split):
        return pd.DataFrame(
            [
                {
                    "trace_name": r["trace_name"],
                    "event_id": r["event_id"],
                    "split": split,
                    "pred_p_sample": r["pred_p_sample"],
                    "pred_s_phasenet": r["pred_s_phasenet"],
                    "pred_s_fixed_rescore": r["pred_s_fixed_rescore"],
                    "true_p_sample": r["true_p_sample"],
                    "true_s_sample": r["true_s_sample"],
                    "sampling_rate_hz": r["sampling_rate"],
                    "abcd_class": r["abcd_class"],
                    "history_available": r["history_available"],
                    "history_count": r["history_count"],
                    "history_mad": r["history_mad"],
                    "fallback_level": r["fallback_level"],
                    "gate": np.nan,
                    "pred_s_learned_gate": np.nan,
                }
                for r in recs
            ]
        )

    pd.concat(
        [_base_df(train_rec, "train"), _base_df(val_rec, "val"), _base_df(test_rec, "test")],
        ignore_index=True,
    ).to_parquet(out_res / f"gate_baselines_{tag}.parquet", index=False)

    summary = {
        "config": args.config,
        "mode": mode,
        "tag": tag,
        "n_train": len(train_rec),
        "n_val": len(val_rec),
        "n_test": len(test_rec),
        "n_train_events": int(pd.Series([r["event_id"] for r in train_rec]).nunique()),
        "n_val_events": int(pd.Series([r["event_id"] for r in val_rec]).nunique()),
        "n_test_events": int(pd.Series([r["event_id"] for r in test_rec]).nunique()),
        "feature_columns": cols,
        "schema_hash": scaler.schema_hash,
        "candidate_k": max_k,
        "paths": {
            "train": str(z_train),
            "val": str(z_val),
            "test": str(z_test),
            "phasenet_cache": str(cache_path),
            "scaler": str(out_model / "feature_scaler.pkl"),
        },
        "hashes": {
            "residual_history": file_hash(artifacts_dir() / "history" / "residual_history_features_frozen.parquet"),
            "phasenet_cache": file_hash(cache_path) if cache_path.exists() else None,
            "test_list": file_hash(test_list),
        },
        "abcd_train": pd.Series([r["abcd_class"] for r in train_rec]).value_counts().to_dict(),
        "event_disjoint": {
            "train_val": len(set(r["event_id"] for r in train_rec) & set(r["event_id"] for r in val_rec)) == 0,
            "train_test": len(set(r["event_id"] for r in train_rec) & set(r["event_id"] for r in test_rec)) == 0,
            "val_test": len(set(r["event_id"] for r in val_rec) & set(r["event_id"] for r in test_rec)) == 0,
        },
    }
    save_json(summary, out_res / "gate_dataset_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
