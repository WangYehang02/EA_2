#!/usr/bin/env python
"""Audit PhaseNet pick metrics: e2e MAE vs matched timing, misses, wrong peaks, multi-peak ceiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.fusion.peak_candidates import extract_candidates
from earthquake.metrics import audit_pick_errors, match_picks
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir


def _attach_cand_lists(cache: pd.DataFrame, cands_df: pd.DataFrame) -> pd.DataFrame:
    g = cands_df.groupby(["trace_name", "phase"])["sample_index"].apply(lambda s: s.to_numpy(dtype=np.float64))
    out = cache.copy()
    out["p_cand_samples"] = out["trace_name"].map(lambda t: g.get((t, "p"), np.array([], dtype=np.float64)))
    out["s_cand_samples"] = out["trace_name"].map(lambda t: g.get((t, "s"), np.array([], dtype=np.float64)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", default="artifacts/results/stage2/phasenet_fixed_cache.parquet")
    parser.add_argument("--max-traces", type=int, default=None, help="override; default=all fixed eval")
    parser.add_argument("--flush-every", type=int, default=200)
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    device = args.device or cfg.get("device", "cpu")
    thr = float(cfg.get("pick_threshold", 0.3))
    out = ensure_dir(artifacts_dir() / "results" / "stage2")

    fixed = pd.read_parquet(artifacts_dir() / "diagnostics" / "fixed_eval_events.parquet")
    if args.max_traces is not None:
        fixed = fixed.head(int(args.max_traces))
    elif int(cfg.get("max_eval_traces", -1)) > 0:
        fixed = fixed.head(int(cfg["max_eval_traces"]))

    cache_path = ROOT / args.cache if not Path(args.cache).is_absolute() else Path(args.cache)
    cands_path = cache_path.with_name(cache_path.stem + "_candidates.parquet")
    ensure_dir(cache_path.parent)

    done = set()
    pick_rows: list[dict] = []
    cand_rows: list[dict] = []
    if cache_path.exists():
        prev = pd.read_parquet(cache_path)
        done = set(prev["trace_name"].astype(str))
        pick_rows = prev.to_dict(orient="records")
        if cands_path.exists():
            cand_rows = pd.read_parquet(cands_path).to_dict(orient="records")
        print({"resuming_cache": str(cache_path), "already_done": len(done)}, flush=True)

    todo = fixed[~fixed["trace_name"].astype(str).isin(done)].copy()
    if len(todo) > 0:
        ref = SeisBenchPhaseNetReference(weight=cfg.get("phasenet_weight", "stead"), device=device)
        h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
        pending_picks: list[dict] = []
        pending_cands: list[dict] = []
        with InstanceHDF5Reader(h5) as reader:
            for i, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="phasenet-audit", file=sys.stderr)):
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
                    k=int(cfg.get("candidate_k", 5)),
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
                        "snr_db": float(row.snr_db) if "snr_db" in row and pd.notna(row.snr_db) else np.nan,
                        "distance_km": float(row.distance_km) if pd.notna(row.distance_km) else np.nan,
                        "true_p_sample": float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan,
                        "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                        "pred_p_sample": p_pick["peak_sample"],
                        "pred_s_sample": s_pick["peak_sample"],
                        "p_peak_probability": p_pick["peak_probability"],
                        "s_peak_probability": s_pick["peak_probability"],
                        "p_n_cands": len(p_cands),
                        "s_n_cands": len(s_cands),
                        "p_n_cands_nonfallback": int(sum(not c.fallback_peak for c in p_cands)),
                        "s_n_cands_nonfallback": int(sum(not c.fallback_peak for c in s_cands)),
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
                if (i + 1) % max(int(args.flush_every), 1) == 0 or (i + 1) == len(todo):
                    pick_rows.extend(pending_picks)
                    cand_rows.extend(pending_cands)
                    pd.DataFrame(pick_rows).to_parquet(cache_path, index=False)
                    pd.DataFrame(cand_rows).to_parquet(cands_path, index=False)
                    print({"flushed": len(pick_rows), "of_target": len(fixed)}, flush=True)
                    pending_picks.clear()
                    pending_cands.clear()
        if pending_picks:
            pick_rows.extend(pending_picks)
            cand_rows.extend(pending_cands)
            pd.DataFrame(pick_rows).to_parquet(cache_path, index=False)
            pd.DataFrame(cand_rows).to_parquet(cands_path, index=False)

    cache = pd.read_parquet(cache_path)
    cache = cache[cache["trace_name"].astype(str).isin(fixed["trace_name"].astype(str))].copy()
    cands_df = pd.read_parquet(cands_path)
    cache = _attach_cand_lists(cache, cands_df)
    print({"saved_cache": str(cache_path), "saved_candidates": str(cands_path), "n": len(cache)}, flush=True)

    report = {"threshold": thr, "n_traces": len(cache), "phases": {}}
    cat_frames = []
    for phase in ("p", "s"):
        pred = cache[f"pred_{phase}_sample"].to_numpy()
        true = cache[f"true_{phase}_sample"].to_numpy()
        sr = cache["sampling_rate_hz"].to_numpy()
        multi = list(cache[f"{phase}_cand_samples"].to_numpy())
        m = match_picks(pred, true, sr)
        a = audit_pick_errors(pred, true, sr, window_s=0.5, multi_pred_samples=multi)
        a_json = {k: v for k, v in a.items() if k not in ("categories", "abs_err_s", "err_s")}
        report["phases"][phase] = {"match_picks": m, "audit": a_json}
        cat_frames.append(
            pd.DataFrame(
                {
                    "trace_name": cache["trace_name"],
                    "event_id": cache["event_id"],
                    "phase": phase,
                    "category": a["categories"],
                    "abs_err_s": a["abs_err_s"],
                    "err_s": a["err_s"],
                    "true_sample": true,
                    "pred_sample": pred,
                    "snr_db": cache["snr_db"],
                    "distance_km": cache["distance_km"],
                    "n_cands": cache[f"{phase}_n_cands"],
                }
            )
        )
    cats = pd.concat(cat_frames, ignore_index=True)
    cats.to_parquet(out / "error_categories.parquet", index=False)
    save_json(report, out / "metric_audit.json")
    print(
        {
            "p_e2e_mae": report["phases"]["p"]["audit"]["e2e_mae"],
            "p_e2e_p95": report["phases"]["p"]["audit"]["e2e_p95_ae"],
            "p_matched_mae": report["phases"]["p"]["audit"]["matched_timing_mae"],
            "p_wrong": report["phases"]["p"]["audit"]["n_wrong_peak_beyond_tol"],
            "p_miss": report["phases"]["p"]["audit"]["n_missed_pick"],
            "s_e2e_mae": report["phases"]["s"]["audit"]["e2e_mae"],
            "s_matched_mae": report["phases"]["s"]["audit"]["matched_timing_mae"],
            "s_multi_recall": report["phases"]["s"]["audit"]["multi_peak_protocol"].get("recall_any_cand"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
