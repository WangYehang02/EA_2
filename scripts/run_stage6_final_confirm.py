#!/usr/bin/env python
"""One-shot Stage-6 final confirm evaluation for locked fixed_rescore_UNION.

Progress/errors only until CONFIRM.CONSUMED. No mid-run F1/P95 prints.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, resolve_instance_root, save_json
from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.gating.cache_io import attach_expected_s
from earthquake.history.residual_prior import path_stats_to_residual_stats
from earthquake.stage6.confirm_state import (
    assert_same_lock_for_resume,
    final_confirm_dir,
    mark_consumed,
    mark_running,
    refuse_if_consumed,
)
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics
from earthquake.stage6.ranker.union_schema import union_candidates_phaseC
from earthquake.utils import ensure_dir


def _progress(msg: str, log: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with log.open("a") as f:
        f.write(line + "\n")


def _build_confirm_s_manifest(out: Path) -> pd.DataFrame:
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    tr = set(load_full_trace_names("stage6_internal_confirm"))
    m = events[events.trace_name.astype(str).isin(tr)].copy()
    s = pd.to_numeric(m["s_arrival_sample"], errors="coerce")
    m = m.loc[s.notna()].copy()
    m = m.sort_values(["origin_time", "event_id", "trace_name"]).reset_index(drop=True)
    csv = out / "confirm_s_eval_manifest.csv"
    m.to_csv(csv, index=False)
    man = {
        "n_events": int(m.event_id.nunique()),
        "n_traces": int(len(m)),
        "n_s_labelled_traces": int(len(m)),
        "csv": str(csv),
        "csv_sha256": sha256_file(csv),
        "population": "all_s_labelled_internal_confirm",
        "max_traces_cap": None,
    }
    save_json(man, out / "confirm_s_eval_manifest.json")
    return m


def _query_confirm_history(meta: pd.DataFrame, out: Path, log: Path) -> pd.DataFrame:
    hist_dir = artifacts_dir() / "models" / "stage6" / "history_picker_train"
    with open(hist_dir / "temporal_history_store.pkl", "rb") as f:
        store = pickle.load(f)
    with open(hist_dir / "travel_time_baseline_mlp.pkl", "rb") as f:
        baseline = pickle.load(f)
    feat_rows = []
    base_df = baseline.predict_frame(meta)
    n_ok = 0
    for i, row in tqdm(meta.iterrows(), total=len(meta), desc="confirm_history_query"):
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
                "subset": "stage6_internal_confirm",
                "trace_name": str(row.trace_name),
                "event_id": str(row.event_id),
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
                "shrinkage_k": 50.0,
            }
        )
        if (i + 1) % 5000 == 0:
            _progress(f"history_query {i+1}/{len(meta)} available={n_ok}", log)
    feat = pd.DataFrame(feat_rows)
    path = out / "confirm_history_features.parquet"
    feat.to_parquet(path, index=False)
    cov = {
        "n_traces": int(len(feat)),
        "n_history_available": int(n_ok),
        "frac_history_available": float(n_ok / max(len(feat), 1)),
        "fallback_level_counts": feat["fallback_level"].value_counts(dropna=False).to_dict(),
        "sha256": sha256_file(path),
        "store_updated": False,
    }
    save_json(cov, out / "confirm_history_coverage.json")
    _progress(f"history_query_done available_frac={cov['frac_history_available']:.4f}", log)
    return feat


def _launch_candidate_caches(manifest_csv: Path, manifest_json: Path, out: Path, log: Path) -> None:
    import subprocess

    gpus = list(range(8))
    for source in ("stead", "ida"):
        cdir = ensure_dir(out / "cache" / f"{source}_top10")
        done = cdir / f"{source}_top10.parquet"
        if done.exists():
            _progress(f"cache_{source}_already_merged", log)
            continue
        pids = []
        for rank, gpu in enumerate(gpus):
            slog = artifacts_dir() / "results" / "stage6" / "logs" / f"confirm_cache_{source}_rank{rank}.log"
            cmd = [
                "bash",
                "-lc",
                f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate PS && "
                f"CUDA_VISIBLE_DEVICES={gpu} python scripts/cache_stage6_phaseB_candidates.py "
                f"--source {source} --manifest {manifest_csv} --manifest-json {manifest_json} "
                f"--out-dir {cdir} --rank {rank} --world-size {len(gpus)} --device cuda:0 "
                f">>{slog} 2>&1",
            ]
            p = subprocess.Popen(cmd, cwd=str(ROOT))
            pids.append(p)
            _progress(f"launch_cache source={source} rank={rank} gpu={gpu} pid={p.pid}", log)
        fail = False
        for p in pids:
            rc = p.wait()
            if rc != 0:
                fail = True
        if fail:
            raise SystemExit(f"cache_{source}_failed")
        # merge
        subprocess.check_call(
            [
                "bash",
                "-lc",
                f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate PS && "
                f"python scripts/cache_stage6_phaseB_candidates.py --source {source} --out-dir {cdir} "
                f"--world-size {len(gpus)} --merge-only",
            ],
            cwd=str(ROOT),
        )
        _progress(f"cache_{source}_merged rows={len(pd.read_parquet(done))}", log)


def _fixed_rescore(union: pd.DataFrame, meta: pd.DataFrame, hist: pd.DataFrame, global_res: dict, lw: float, lh: float, lp: float) -> np.ndarray:
    hist_ix = hist.drop_duplicates("trace_name").set_index("trace_name")
    by = {str(tn): g for tn, g in union.groupby(union.trace_name.astype(str), sort=False)}
    preds = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="fixed_rescore"):
        tn = str(row.trace_name)
        g = by.get(tn)
        if g is None or len(g) == 0:
            preds.append(np.nan)
            continue
        merged = row.copy()
        if tn in hist_ix.index:
            h = hist_ix.loc[tn]
            for c in h.index:
                merged[c] = h[c]
        exp = attach_expected_s(
            merged,
            global_res=global_res,
            shrink_k=50.0,
            min_history=5,
            mad_disable_s=1.0,
            min_sigma_s=0.05,
            max_sigma_s=1.0,
        )
        cands = []
        for _, r in g.iterrows():
            ps = float(r["stead_probability"]) if pd.notna(r.get("stead_probability")) else -1
            pi = float(r["ida_probability"]) if pd.notna(r.get("ida_probability")) else -1
            p = max(ps, pi, 1e-6)
            cands.append(
                PeakCandidate(
                    sample_index=int(r["candidate_sample"]),
                    absolute_utc=None,
                    peak_probability=p,
                    prominence=p,
                    peak_width=float("nan"),
                    local_entropy=0.0,
                    rank=int(r.get("candidate_index", 0)),
                    fallback_peak=False,
                    phase="S",
                )
            )
        best, _ = rescore_phase_candidates(
            cands,
            expected_sample=float(exp["expected_s_sample"]),
            sigma_samples=float(exp["history_sigma_samples"]),
            lambda_wave=lw,
            lambda_history=lh,
            lambda_prominence=lp,
            history_available=bool(exp["gate_history_available"]),
        )
        preds.append(float(best.sample_index) if best is not None else np.nan)
    return np.asarray(preds, float)


def _top1_from_cache(cache: pd.DataFrame, names: np.ndarray) -> np.ndarray:
    g = cache.sort_values("candidate_rank").groupby("trace_name").first()
    col = "top1_s_sample" if "top1_s_sample" in g.columns else "candidate_sample"
    return g[col].reindex(names).to_numpy(float)


def _oracle(union: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    by = {str(tn): g for tn, g in union.groupby(union.trace_name.astype(str), sort=False)}
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    out = []
    for i, tn in enumerate(meta.trace_name.astype(str)):
        g = by.get(tn)
        if g is None or not np.isfinite(true[i]):
            out.append(np.nan)
            continue
        samp = g["candidate_sample"].to_numpy(float)
        ae = np.abs((samp - true[i]) / sr[i])
        out.append(float(samp[int(np.argmin(ae))]))
    return np.asarray(out, float)


def _bootstrap(pred_a, pred_b, true, sr, event_ids, n_boot=5000, seed=20260817):
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    idx_by = {}
    for i, e in enumerate(event_ids.astype(str)):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(idx_by[e], int) for e in events]

    def bundle(pred, ix=None):
        if ix is None:
            p, t, s = pred, true, sr
        else:
            p, t, s = pred[ix], true[ix], sr[ix]
        return comprehensive_pick_metrics(p, t, s)

    keys = ["f1@0.5", "f1@0.1", "precision@0.5", "recall@0.5", "miss_rate", "wrong_peak_rate", "detected_ae_p95"]
    base_a, base_b = bundle(pred_a), bundle(pred_b)
    deltas = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma, mb = bundle(pred_a, ix), bundle(pred_b, ix)
        for k in keys:
            deltas[k][b] = float(ma[k] - mb[k])
    out = {"n_boot": n_boot, "seed": seed, "point_delta": {k: float(base_a[k] - base_b[k]) for k in keys}}
    for k in keys:
        arr = deltas[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {"mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)]}
    return out


def _decide(union_m, stead_m, boot_vs_stead) -> str:
    d05 = union_m["f1@0.5"] - stead_m["f1@0.5"]
    d01 = union_m["f1@0.1"] - stead_m["f1@0.1"]
    dp95 = union_m["detected_ae_p95"] - stead_m["detected_ae_p95"]
    drec = union_m["recall@0.5"] - stead_m["recall@0.5"]
    ci_lo = boot_vs_stead["f1@0.5"]["ci95"][0]
    if d05 <= 0 or drec < -0.02 or dp95 > 0.5:
        return "not_confirmed"
    if d05 >= 0.01 and ci_lo > 0 and d01 >= -0.003 and dp95 <= 0:
        return "strong_confirmed"
    if d05 > 0 and ci_lo > 0 and d01 >= -0.003 and dp95 <= 0.25:
        return "modest_confirmed"
    if d05 > 0 and ci_lo <= 0:
        return "direction_only_underpowered"
    return "not_confirmed"


def main() -> None:
    refuse_if_consumed()
    out = final_confirm_dir()
    log = artifacts_dir() / "results" / "stage6" / "logs" / "confirm_final.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out / "method_lock.json"
    lock = load_json(lock_path)
    lock_hash = (out / "method_lock.sha256").read_text().strip()
    assert sha256_file(lock_path) == lock_hash
    assert_same_lock_for_resume(lock_hash)
    if not (out / "CONFIRM_EVAL.RUNNING").exists():
        mark_running(method_lock_hash=lock_hash)
    _progress("CONFIRM_EVAL.RUNNING", log)

    meta = _build_confirm_s_manifest(out)
    _progress(f"manifest_s_labelled n={len(meta)} events={meta.event_id.nunique()}", log)

    hist_path = out / "confirm_history_features.parquet"
    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        _progress("history_features_resume", log)
    else:
        hist = _query_confirm_history(meta, out, log)

    man_csv = out / "confirm_s_eval_manifest.csv"
    man_json = out / "confirm_s_eval_manifest.json"
    _launch_candidate_caches(man_csv, man_json, out, log)

    stead = pd.read_parquet(out / "cache" / "stead_top10" / "stead_top10.parquet")
    ida = pd.read_parquet(out / "cache" / "ida_top10" / "ida_top10.parquet")
    union_path = out / "confirm_union.parquet"
    if not union_path.exists():
        _progress("building_union", log)
        union = union_candidates_phaseC(stead, ida, stead_k=5, ida_k=5, max_union=10)
        # attach labels from meta
        meta_ix = meta.set_index("trace_name")
        union["true_s_sample"] = union["trace_name"].astype(str).map(lambda t: float(meta_ix.loc[t, "s_arrival_sample"]) if t in meta_ix.index else np.nan)
        union["sampling_rate_hz"] = union["trace_name"].astype(str).map(lambda t: float(meta_ix.loc[t, "sampling_rate_hz"]) if t in meta_ix.index else 100.0)
        union["event_id"] = union["trace_name"].astype(str).map(lambda t: str(meta_ix.loc[t, "event_id"]) if t in meta_ix.index else "")
        union.to_parquet(union_path, index=False)
    else:
        union = pd.read_parquet(union_path)
    _progress(f"union_rows={len(union)} traces={union.trace_name.nunique()}", log)

    names = meta.trace_name.astype(str).to_numpy()
    true = meta.s_arrival_sample.to_numpy(float)
    sr = meta.sampling_rate_hz.to_numpy(float)
    events = meta.event_id.astype(str).to_numpy()
    global_res = lock["fixed_rescore"]["global_residual"]
    lw = float(lock["fixed_rescore"]["lambdas_s"]["lw"])
    lh = float(lock["fixed_rescore"]["lambdas_s"]["lh"])
    lp = float(lock["fixed_rescore"]["lambdas_s"]["lp"])

    pred_paths = {
        "STEAD_top1": out / "pred_STEAD_top1.npy",
        "IDA_top1": out / "pred_IDA_top1.npy",
        "fixed_rescore_STEAD": out / "pred_fixed_rescore_STEAD.npy",
        "fixed_rescore_UNION": out / "pred_fixed_rescore_UNION.npy",
        "oracle_UNION": out / "pred_oracle_UNION.npy",
    }
    if not pred_paths["STEAD_top1"].exists():
        np.save(pred_paths["STEAD_top1"], _top1_from_cache(stead, names))
        _progress("saved_STEAD_top1", log)
    if not pred_paths["IDA_top1"].exists():
        np.save(pred_paths["IDA_top1"], _top1_from_cache(ida, names))
        _progress("saved_IDA_top1", log)
    if not pred_paths["oracle_UNION"].exists():
        np.save(pred_paths["oracle_UNION"], _oracle(union, meta))
        _progress("saved_oracle", log)
    if not pred_paths["fixed_rescore_UNION"].exists():
        np.save(pred_paths["fixed_rescore_UNION"], _fixed_rescore(union, meta, hist, global_res, lw, lh, lp))
        _progress("saved_fixed_UNION", log)
    if not pred_paths["fixed_rescore_STEAD"].exists():
        from earthquake.stage6.phaseB import topk_from_cache

        s5 = topk_from_cache(stead, 5).rename(columns={"candidate_probability": "stead_probability"})
        s5["ida_probability"] = np.nan
        # reuse fixed rescore path expecting stead/ida prob cols
        s5u = s5.copy()
        if "stead_probability" not in s5u.columns:
            s5u["stead_probability"] = s5u.get("candidate_probability", np.nan)
        s5u["ida_probability"] = np.nan
        s5u["candidate_index"] = s5u.get("candidate_rank", 0)
        np.save(pred_paths["fixed_rescore_STEAD"], _fixed_rescore(s5u, meta, hist, global_res, lw, lh, lp))
        _progress("saved_fixed_STEAD", log)

    # Assemble predictions table (metrics computed but NOT printed)
    preds = {
        k: np.load(v) for k, v in pred_paths.items()
    }
    rows = []
    for i, tn in enumerate(names):
        rows.append(
            {
                "trace_name": tn,
                "event_id": events[i],
                "true_s_sample": true[i],
                "sampling_rate_hz": sr[i],
                "STEAD_top1": preds["STEAD_top1"][i],
                "IDA_top1": preds["IDA_top1"][i],
                "fixed_rescore_STEAD": preds["fixed_rescore_STEAD"][i],
                "fixed_rescore_UNION": preds["fixed_rescore_UNION"][i],
                "oracle_UNION": preds["oracle_UNION"][i],
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_path = out / "confirm_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    metrics = {k: comprehensive_pick_metrics(preds[k], true, sr) for k in preds}
    # enrich coverage fields
    for k, m in metrics.items():
        m["n_events"] = int(meta.event_id.nunique())
        m["n_stations"] = int(meta.station.nunique()) if "station" in meta.columns else None
        m["n_traces"] = int(len(meta))
    save_json(metrics, out / "confirm_method_metrics.json")

    # comparison table
    cmp_rows = []
    for k, m in metrics.items():
        cmp_rows.append({"method": k, **{kk: m[kk] for kk in ["f1@0.1", "f1@0.5", "precision@0.5", "recall@0.5", "miss_rate", "wrong_peak_rate", "detected_ae_p95", "prediction_coverage"]}})
    pd.DataFrame(cmp_rows).to_csv(out / "confirm_comparison_table.csv", index=False)

    boot = {
        "fixed_UNION_vs_STEAD_top1": _bootstrap(preds["fixed_rescore_UNION"], preds["STEAD_top1"], true, sr, events),
        "fixed_UNION_vs_fixed_STEAD": _bootstrap(preds["fixed_rescore_UNION"], preds["fixed_rescore_STEAD"], true, sr, events),
        "fixed_STEAD_vs_STEAD_top1": _bootstrap(preds["fixed_rescore_STEAD"], preds["STEAD_top1"], true, sr, events),
    }
    save_json(boot, out / "confirm_bootstrap.json")

    # complementarity quick
    u = union.copy()
    both = float(u.groupby("trace_name")["both_support"].any().mean()) if "both_support" in u.columns else None
    save_json(
        {
            "dual_source_trace_rate": both,
            "mean_candidates": float(u.groupby("trace_name").size().mean()),
            "n_union_rows": int(len(u)),
        },
        out / "confirm_candidate_complementarity.json",
    )

    status = _decide(metrics["fixed_rescore_UNION"], metrics["STEAD_top1"], boot["fixed_UNION_vs_STEAD_top1"])
    verdict = {
        "recommended_main_method": "fixed_rescore_UNION",
        "confirm_status": status,
        "catalog_assisted": True,
        "blind_picker": False,
        "sota_claim_allowed": False,
        "blind_picker_claim_allowed": False,
        "multistation_may_start": False,
        "ranker_recommended": False,
        "confirm_consumed": True,
        "n_confirm_events_total": 2700,
        "n_confirm_traces_total": 74753,
        "n_eval_s_labelled": int(len(meta)),
        "n_eval_events": int(meta.event_id.nunique()),
        "method_lock_sha256": lock_hash,
        "noise_confirm_unavailable": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(verdict, out / "confirm_final_verdict.json")

    # cache manifest
    save_json(
        {
            "stead_cache": str(out / "cache/stead_top10/stead_top10.parquet"),
            "ida_cache": str(out / "cache/ida_top10/ida_top10.parquet"),
            "stead_sha256": sha256_file(out / "cache/stead_top10/stead_top10.parquet"),
            "ida_sha256": sha256_file(out / "cache/ida_top10/ida_top10.parquet"),
            "union_sha256": sha256_file(union_path),
        },
        out / "confirm_candidate_cache_manifest.json",
    )

    mark_consumed(
        method_lock_hash=lock_hash,
        predictions_path=pred_path,
        metrics_path=out / "confirm_method_metrics.json",
        bootstrap_path=out / "confirm_bootstrap.json",
    )

    # Report (metrics now allowed)
    fu, st, fs = metrics["fixed_rescore_UNION"], metrics["STEAD_top1"], metrics["fixed_rescore_STEAD"]
    md = f"""# Stage 6 Final Confirmatory Report

**Status:** `{status}`  
**Main method:** `fixed_rescore_UNION`  
**Confirm:** CONSUMED  
**SOTA claim allowed:** false  
**Multistation:** false  

## Population

- confirm events/traces (all): 2700 / 74753
- S-labelled eval: {len(meta)} traces / {meta.event_id.nunique()} events

## Core metrics

| Method | F1@0.1 | F1@0.5 | P95 | miss | coverage |
|--|--:|--:|--:|--:|--:|
| STEAD top-1 | {st['f1@0.1']:.4f} | {st['f1@0.5']:.4f} | {st['detected_ae_p95']:.3f} | {st['miss_rate']:.4f} | {st['prediction_coverage']:.3f} |
| fixed_rescore_STEAD | {fs['f1@0.1']:.4f} | {fs['f1@0.5']:.4f} | {fs['detected_ae_p95']:.3f} | {fs['miss_rate']:.4f} | {fs['prediction_coverage']:.3f} |
| fixed_rescore_UNION | {fu['f1@0.1']:.4f} | {fu['f1@0.5']:.4f} | {fu['detected_ae_p95']:.3f} | {fu['miss_rate']:.4f} | {fu['prediction_coverage']:.3f} |
| UNION oracle (ceiling) | {metrics['oracle_UNION']['f1@0.1']:.4f} | {metrics['oracle_UNION']['f1@0.5']:.4f} | {metrics['oracle_UNION']['detected_ae_p95']:.3f} | — | — |

## Bootstrap (fixed UNION − STEAD)

ΔF1@0.5 mean={boot['fixed_UNION_vs_STEAD_top1']['f1@0.5']['mean_delta']:+.4f} CI={boot['fixed_UNION_vs_STEAD_top1']['f1@0.5']['ci95']}

## Claims

Catalog-assisted S-phase candidate re-picking/refinement only. Not blind picker. Not SOTA.
"""
    (ROOT / "reports/stage6/stage6_final_confirmatory_report.md").write_text(md)
    paper = ROOT / "paper" / "candidate_edits"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "stage6_final_confirm_update.md").write_text(
        f"# Candidate paper edits (NOT applied)\n\nConfirm status `{status}`.\nMain method fixed_rescore_UNION.\nDo not claim blind SOTA.\n"
    )
    _progress(f"CONFIRM.CONSUMED status={status}", log)
    # final public print
    print(json.dumps({"confirm_status": status, "consumed": True}, indent=2))


if __name__ == "__main__":
    main()
