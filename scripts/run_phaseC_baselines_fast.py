#!/usr/bin/env python
"""Fast Phase C non-learning baselines on full stage6_dev."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.gating.cache_io import attach_expected_s
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.phaseB import oracle_metrics_for_set, topk_from_cache
from earthquake.utils import ensure_dir


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_baselines")
        raise SystemExit("method_lock")
    except RuntimeError:
        pass


def pick_metrics(pred, true, sr) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.2, 0.5))
    b = report_pick_timing_bundle(pred, true, sr)
    return {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.2": float(m.get("f1@0.2s", np.nan)),
        "f1@0.5": float(m["f1@0.5s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(b["miss_rate"]),
        "no_pick_rate": float(b["no_pick_rate"]),
        "wrong_peak_rate": float(b["wrong_peak_rate"]),
        "detected_ae_median": float(b["detected_ae_median"]),
        "detected_ae_mae": float(b["detected_ae_mae"]),
        "detected_ae_p95": float(b["detected_ae_p95"]),
    }


def main() -> None:
    _guard()
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC")
    pred_dir = ensure_dir(out / "baseline_preds")
    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    global_res = hist_man["global_residual"]
    hist = pd.read_parquet(
        artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"
    )
    hist = hist[hist["subset"] == "stage6_dev"].drop_duplicates("trace_name").set_index("trace_name")
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    stead = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseB" / "stead_top10" / "stead_top10.parquet")
    ida = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseB" / "ida_top10" / "ida_top10.parquet")
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")

    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    names = meta["trace_name"].astype(str).to_numpy()

    # precompute expected S for all meta rows once
    print("precompute history expected S...", flush=True)
    exp_s = {}
    sigma = {}
    hist_ok = {}
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        tn = str(row.trace_name)
        merged = row.copy()
        if tn in hist.index:
            h = hist.loc[tn]
            for c in h.index:
                merged[c] = h[c]
        e = attach_expected_s(
            merged,
            global_res=global_res,
            shrink_k=50.0,
            min_history=5,
            mad_disable_s=1.0,
            min_sigma_s=0.05,
            max_sigma_s=1.0,
        )
        exp_s[tn] = float(e["expected_s_sample"])
        sigma[tn] = float(e["history_sigma_samples"])
        hist_ok[tn] = bool(e["gate_history_available"])

    def top1(cache: pd.DataFrame) -> np.ndarray:
        g = cache.sort_values("candidate_rank").groupby("trace_name").first()
        col = "top1_s_sample" if "top1_s_sample" in g.columns else "candidate_sample"
        return g[col].reindex(names).to_numpy(float)

    def prob_heur(u: pd.DataFrame) -> np.ndarray:
        uu = u.copy()
        uu["_p"] = np.fmax(uu["stead_probability"].fillna(-1), uu["ida_probability"].fillna(-1))
        best = uu.sort_values(["trace_name", "_p"], ascending=[True, False]).groupby("trace_name").first()
        return best["candidate_sample"].reindex(names).to_numpy(float)

    def fixed(cands: pd.DataFrame, *, prob_from: str) -> np.ndarray:
        by = {str(tn): g for tn, g in cands.groupby(cands["trace_name"].astype(str), sort=False)}
        preds = []
        for tn in tqdm(names, desc=f"fixed:{prob_from}"):
            g = by.get(tn)
            if g is None or len(g) == 0:
                preds.append(np.nan)
                continue
            lst = []
            for _, r in g.iterrows():
                if prob_from == "stead_cand":
                    p = float(r["candidate_probability"])
                else:
                    ps = float(r["stead_probability"]) if pd.notna(r.get("stead_probability")) else -1.0
                    pi = float(r["ida_probability"]) if pd.notna(r.get("ida_probability")) else -1.0
                    p = max(ps, pi)
                samp = float(r["candidate_sample"] if "candidate_sample" in r else r.get("candidate_sample"))
                if "candidate_index" in r:
                    samp = float(r["candidate_sample"])
                    rank = int(r["candidate_index"])
                else:
                    rank = int(r["candidate_rank"])
                lst.append(
                    PeakCandidate(
                        sample_index=int(samp),
                        absolute_utc=None,
                        peak_probability=max(p, 1e-6),
                        prominence=max(p, 1e-6),
                        peak_width=float("nan"),
                        local_entropy=0.0,
                        rank=rank,
                        fallback_peak=False,
                        phase="S",
                    )
                )
            best, _ = rescore_phase_candidates(
                lst,
                expected_sample=exp_s[tn],
                sigma_samples=sigma[tn],
                lambda_wave=0.5,
                lambda_history=2.0,
                lambda_prominence=0.0,
                history_available=hist_ok[tn],
            )
            preds.append(float(best.sample_index) if best is not None else np.nan)
        return np.asarray(preds, float)

    results = {}
    print("STEAD/IDA top1", flush=True)
    p_stead = top1(stead)
    p_ida = top1(ida)
    results["STEAD_top1"] = pick_metrics(p_stead, true, sr)
    results["IDA_top1"] = pick_metrics(p_ida, true, sr)
    np.save(pred_dir / "STEAD_top1.npy", p_stead)
    np.save(pred_dir / "IDA_top1.npy", p_ida)

    print("prob heuristic", flush=True)
    p_heur = prob_heur(union)
    results["prob_heuristic_UNION"] = pick_metrics(p_heur, true, sr)
    np.save(pred_dir / "prob_heuristic_UNION.npy", p_heur)

    print("fixed STEAD", flush=True)
    p_fs = fixed(topk_from_cache(stead, 5), prob_from="stead_cand")
    results["fixed_rescore_STEAD"] = pick_metrics(p_fs, true, sr)
    np.save(pred_dir / "fixed_rescore_STEAD.npy", p_fs)

    print("fixed UNION", flush=True)
    p_fu = fixed(union, prob_from="union")
    results["fixed_rescore_UNION"] = pick_metrics(p_fu, true, sr)
    np.save(pred_dir / "fixed_rescore_UNION.npy", p_fu)

    ora = oracle_metrics_for_set(union, meta)
    results["oracle_UNION"] = {
        "f1@0.1": ora["oracle_f1@0.1"],
        "f1@0.5": ora["oracle_f1@0.5"],
        "detected_ae_p95": ora["closest_candidate_p95"],
        "miss_rate": ora["candidate_miss_rate"],
    }
    save_json(results, out / "phaseC_baselines.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
