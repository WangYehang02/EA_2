#!/usr/bin/env python
"""Non-learning baselines + full-dev ranker evaluation for Phase C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.gating.cache_io import attach_expected_s, fixed_rescore_s
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned
from earthquake.stage6.phaseB import oracle_metrics_for_set, topk_from_cache
from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.features import feature_names
from earthquake.stage6.ranker.models import build_ranker
from earthquake.utils import ensure_dir


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_eval")
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


def baseline_top1(cache: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    g = cache.sort_values("candidate_rank").groupby("trace_name").first()
    if "top1_s_sample" in g.columns:
        s = g["top1_s_sample"]
    else:
        s = g["candidate_sample"]
    return s.reindex(meta["trace_name"].astype(str)).to_numpy(dtype=float)


def baseline_prob_heuristic(union: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """Pre-registered: max available of (stead_prob, ida_prob); no weight search."""
    u = union.copy()
    u["_p"] = np.fmax(
        u["stead_probability"].fillna(-1).to_numpy(),
        u["ida_probability"].fillna(-1).to_numpy(),
    )
    best = u.sort_values(["trace_name", "_p"], ascending=[True, False]).groupby("trace_name").first()
    return best["candidate_sample"].reindex(meta["trace_name"].astype(str)).to_numpy(dtype=float)


def baseline_fixed_rescore(
    cand_df: pd.DataFrame,
    meta: pd.DataFrame,
    hist: pd.DataFrame,
    global_res: dict,
    *,
    lw: float,
    lh: float,
    lp: float,
    sample_col: str = "candidate_sample",
    prob_col: str = "stead_probability",
) -> np.ndarray:
    hist_map = {str(r.trace_name): r for _, r in hist.iterrows()}
    by = {str(tn): g for tn, g in cand_df.groupby(cand_df["trace_name"].astype(str), sort=False)}
    preds = []
    for _, row in meta.iterrows():
        tn = str(row["trace_name"])
        g = by.get(tn)
        h = hist_map.get(tn)
        if g is None or len(g) == 0:
            preds.append(np.nan)
            continue
        merged = row.copy()
        if h is not None:
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
            p = float(r[prob_col]) if prob_col in r and pd.notna(r[prob_col]) else float("nan")
            if not np.isfinite(p):
                # union: take max available
                ps = float(r["stead_probability"]) if pd.notna(r.get("stead_probability")) else -1
                pi = float(r["ida_probability"]) if pd.notna(r.get("ida_probability")) else -1
                p = max(ps, pi)
            cands.append(
                PeakCandidate(
                    sample_index=int(r[sample_col]),
                    absolute_utc=None,
                    peak_probability=max(p, 1e-6),
                    prominence=max(p, 1e-6),
                    peak_width=float("nan"),
                    local_entropy=0.0,
                    rank=int(r.get("candidate_index", r.get("candidate_rank", 0))),
                    fallback_peak=False,
                    phase="S",
                )
            )
        pred = fixed_rescore_s(
            cands,
            expected_s=float(exp["expected_s_sample"]),
            sigma_samples=float(exp["history_sigma_samples"]),
            history_available=bool(exp["gate_history_available"]),
            lw=lw,
            lh=lh,
            lp=lp,
        )
        preds.append(pred)
    return np.asarray(preds, dtype=float)


@torch.no_grad()
def eval_ranker_ckpt(ckpt_path: Path, feat_df: pd.DataFrame, device: str) -> pd.DataFrame:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = blob["feature_names"]
    model = build_ranker(blob["variant"], len(names))
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    ds = TraceCandidateDataset(feat_df, names, max_k=10, hard_boost=False, seed=0)
    ds.indices = list(range(len(ds.groups)))
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_traces)
    rows = []
    max_k = 10
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)
        pred_idx = logits.argmax(dim=-1).cpu().numpy()
        samples = batch["samples"].numpy()
        for i, tn in enumerate(batch["trace_name"]):
            pi = int(pred_idx[i])
            pred = np.nan if pi >= max_k or (not bool(mask[i, pi])) else float(samples[i, pi])
            rows.append(
                {
                    "trace_name": tn,
                    "event_id": batch["event_id"][i],
                    "pred_s_sample": pred,
                    "true_s_sample": float(batch["true_s"][i]),
                    "sampling_rate_hz": float(batch["sr"][i]),
                    "selected_index": pi,
                    "none_of_k": bool(pi >= max_k),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--ranker-ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _guard()
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC")
    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    global_res = hist_man["global_residual"]
    hist = pd.read_parquet(artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet")
    hist = hist[hist["subset"] == "stage6_dev"].drop_duplicates("trace_name")
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    stead = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseB" / "stead_top10" / "stead_top10.parquet")
    ida = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseB" / "ida_top10" / "ida_top10.parquet")
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")

    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)

    results = {}
    results["STEAD_top1"] = pick_metrics(baseline_top1(stead, meta), true, sr)
    results["IDA_top1"] = pick_metrics(baseline_top1(ida, meta), true, sr)
    # fixed rescore on STEAD K5
    s5 = topk_from_cache(stead, 5)
    results["fixed_rescore_STEAD"] = pick_metrics(
        baseline_fixed_rescore(s5, meta, hist, global_res, lw=0.5, lh=2.0, lp=0.0, prob_col="candidate_probability"),
        true,
        sr,
    )
    results["fixed_rescore_UNION"] = pick_metrics(
        baseline_fixed_rescore(union, meta, hist, global_res, lw=0.5, lh=2.0, lp=0.0),
        true,
        sr,
    )
    results["prob_heuristic_UNION"] = pick_metrics(baseline_prob_heuristic(union, meta), true, sr)
    ora = oracle_metrics_for_set(union, meta.rename(columns={"s_arrival_sample": "s_arrival_sample"}))
    results["oracle_UNION"] = {
        "f1@0.1": ora["oracle_f1@0.1"],
        "f1@0.5": ora["oracle_f1@0.5"],
        "detected_ae_p95": ora["closest_candidate_p95"],
        "miss_rate": ora["candidate_miss_rate"],
    }
    save_json(results, out / "phaseC_baselines.json")

    if args.ranker_ckpt:
        feat = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R2.parquet")
        # if R1 ckpt, load R1 features
        blob = torch.load(args.ranker_ckpt, map_location="cpu", weights_only=False)
        if blob["variant"] == "R1":
            feat = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R1.parquet")
        device = args.device if torch.cuda.is_available() else "cpu"
        preds = eval_ranker_ckpt(Path(args.ranker_ckpt), feat, device)
        # Keep raw parquet for provenance; metrics MUST be keyed (never row-order vs meta).
        preds.to_parquet(out / "ranker_dev_predictions.parquet", index=False)
        aligned = keyed_align_predictions(meta, preds)
        met = metrics_from_aligned(aligned)
        met["none_of_k_rate"] = float(aligned["none_of_k"].mean()) if "none_of_k" in aligned.columns else float((~np.isfinite(aligned["pred_s_sample"])).mean())
        met["alignment"] = "keyed_trace_name"
        save_json(met, out / "ranker_dev_metrics_keyed.json")
        # Do not overwrite historical overnight ranker_dev_metrics.json (provenance).
        print(json.dumps(met, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
