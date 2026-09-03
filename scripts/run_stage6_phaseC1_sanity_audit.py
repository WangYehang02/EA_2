#!/usr/bin/env python
"""Stage 6 Phase C.1 — Candidate Ranker Sanity Audit.

Read-only: no training, no confirm unseal, does not overwrite phaseC_final_verdict.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, full_stage6_paths, load_full_event_ids, load_full_trace_names
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage6.phaseC1.infer import align_preds_to_meta, predict_ranker_variants
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics, correct_at
from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.models import build_ranker
from earthquake.utils import ensure_dir


def _guard_confirm_sealed() -> dict:
    seal = full_stage6_paths()["confirm_seal"]
    assert seal.exists(), "CONFIRM_SEALED missing"
    payload = json.loads(seal.read_text())
    assert payload.get("status") == "SEALED"
    try:
        assert_full_confirm_access_allowed(purpose="phaseC1_audit")
        raise SystemExit("method_lock unexpectedly present — abort")
    except RuntimeError:
        pass
    # ensure confirm lists not loaded into audit population
    return {"confirm_seal_path": str(seal), "status": "SEALED", "payload": payload}


def _sha_list(xs: list[str]) -> str:
    blob = "\n".join(xs).encode()
    return hashlib.sha256(blob).hexdigest()


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def cohort_provenance(meta: pd.DataFrame, out: Path) -> dict:
    events = sorted(meta["event_id"].astype(str).unique().tolist())
    traces = meta["trace_name"].astype(str).tolist()
    traces_sorted = sorted(traces)
    confirm_ev = set(load_full_event_ids("stage6_internal_confirm"))
    confirm_tr = set(load_full_trace_names("stage6_internal_confirm"))
    train_ev = set(load_full_event_ids("stage6_picker_train")) | set(load_full_event_ids("stage6_ranker_train"))
    train_tr = set(load_full_trace_names("stage6_picker_train")) | set(load_full_trace_names("stage6_ranker_train"))

    paths = {
        "phaseB_eval_manifest_csv": artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv",
        "phaseB_eval_manifest_json": artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.json",
        "stead_cache": artifacts_dir() / "cache" / "stage6" / "phaseB" / "stead_top10" / "stead_top10.parquet",
        "ida_cache": artifacts_dir() / "cache" / "stage6" / "phaseB" / "ida_top10" / "ida_top10.parquet",
        "dev_union": artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet",
        "dev_features_R2": artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R2.parquet",
        "dev_features_R1": artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R1.parquet",
        "ranker_preds": artifacts_dir() / "results" / "stage6" / "phaseC" / "ranker_dev_predictions.parquet",
    }
    path_hashes = {k: _file_sha(v) for k, v in paths.items()}

    # candidate cache coverage
    stead = pd.read_parquet(paths["stead_cache"], columns=["trace_name"])
    ida = pd.read_parquet(paths["ida_cache"], columns=["trace_name"])
    union = pd.read_parquet(paths["dev_union"], columns=["trace_name"])
    feat = pd.read_parquet(paths["dev_features_R2"], columns=["trace_name"])
    set_meta = set(traces)
    set_stead = set(stead.trace_name.astype(str))
    set_ida = set(ida.trace_name.astype(str))
    set_union = set(union.trace_name.astype(str))
    set_feat = set(feat.trace_name.astype(str))

    dup = int(meta["trace_name"].duplicated().sum())
    overlap_confirm_ev = sorted(set(events) & confirm_ev)[:5]
    overlap_confirm_tr = sorted(set_meta & confirm_tr)[:5]
    overlap_train_tr = sorted(set_meta & train_tr)[:5]

    doc = {
        "audit_population": "phaseB_full_s_labelled_dev",
        "n_events": int(meta["event_id"].nunique()),
        "n_traces": int(len(meta)),
        "n_s_labelled_traces": int(meta["s_arrival_sample"].notna().sum()) if "s_arrival_sample" in meta.columns else int(len(meta)),
        "event_id_set_sha256": _sha_list(events),
        "trace_name_set_sha256": _sha_list(traces_sorted),
        "trace_name_order_sha256": _sha_list(traces),
        "duplicate_trace_rows": dup,
        "path_hashes": {k: v for k, v in path_hashes.items()},
        "coverage": {
            "meta_minus_stead": len(set_meta - set_stead),
            "meta_minus_ida": len(set_meta - set_ida),
            "meta_minus_union": len(set_meta - set_union),
            "meta_minus_feat": len(set_meta - set_feat),
            "stead_minus_meta": len(set_stead - set_meta),
            "ida_minus_meta": len(set_ida - set_meta),
            "union_minus_meta": len(set_union - set_meta),
            "feat_minus_meta": len(set_feat - set_meta),
            "union_n_rows": int(len(union)),
            "feat_n_rows": int(len(feat)),
            "join_row_inflation_union_vs_traces": int(len(union) - len(set_union)),
        },
        "contamination_checks": {
            "confirm_event_overlap_n": len(set(events) & confirm_ev),
            "confirm_trace_overlap_n": len(set_meta & confirm_tr),
            "confirm_event_examples": overlap_confirm_ev,
            "confirm_trace_examples": overlap_confirm_tr,
            "picker_or_ranker_train_trace_overlap_n": len(set_meta & train_tr),
            "train_trace_examples": overlap_train_tr,
            "phaseA_1024_head_note": "Phase A annotate used head(1024) of stage6_dev; Phase C.1 uses full S-labelled 87293",
            "pilot_10k_excluded": True,
        },
        "expected": {"n_events": 5341, "n_traces": 87293},
        "matches_expected": int(meta["event_id"].nunique()) == 5341 and len(meta) == 87293,
    }
    save_json(doc, out / "cohort_provenance.json")
    return doc


@torch.no_grad()
def eval_all_rankers(device: str, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict]]:
    root = artifacts_dir() / "models" / "stage6" / "ranker"
    feat_r1 = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R1.parquet")
    feat_r2 = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R2.parquet")
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    names_meta = meta["trace_name"].astype(str).tolist()

    batch_cache: dict[str, list] = {}

    def get_batches(variant: str, names: list[str]):
        key = variant  # R1/R2 feature schemas are fixed per variant
        if key not in batch_cache:
            feat = feat_r1 if variant == "R1" else feat_r2
            ds = TraceCandidateDataset(feat, names, max_k=10, hard_boost=False, seed=0)
            ds.indices = list(range(len(ds.groups)))
            loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_traces)
            batch_cache[key] = [batch for batch in loader]
        return batch_cache[key]

    rows = []
    pred_map: dict[str, np.ndarray] = {}
    detail: dict[str, dict] = {}

    for d in sorted(root.iterdir()):
        ck = d / "best.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        variant = blob["variant"]
        names = list(blob["feature_names"])
        model = build_ranker(variant, len(names))
        model.load_state_dict(blob["model"])
        model.to(device).eval()
        batches = get_batches(variant, names)
        pred_by_tn = {}
        none_by_tn = {}
        for batch in batches:
            logits = model(batch["x"].to(device), batch["mask"].to(device))
            idx = logits.argmax(-1).cpu().numpy()
            samp = batch["samples"].numpy()
            mask = batch["mask"].numpy()
            for i, tn in enumerate(batch["trace_name"]):
                pi = int(idx[i])
                none = pi >= 10
                pred = np.nan if none or not bool(mask[i, pi]) else float(samp[i, pi])
                pred_by_tn[str(tn)] = pred
                none_by_tn[str(tn)] = none
        pred = np.asarray([pred_by_tn.get(t, np.nan) for t in names_meta], float)
        none = np.asarray([none_by_tn.get(t, True) for t in names_meta], bool)
        metrics = comprehensive_pick_metrics(pred, true, sr, none_mask=none)
        key = d.name
        pred_map[key] = pred
        detail[key] = {
            "ckpt": str(ck),
            "variant": variant,
            "seed": blob.get("seed"),
            "beta": blob.get("beta"),
            "gamma": blob.get("gamma"),
            "selection_metric_s_f1@0.5": blob.get("metrics", {}).get("s_f1@0.5"),
            "selection_metric_scope_note": "training used ~12k random event-light subsample of dev when |dev|>12000; not full 87293",
            "n_params": blob.get("n_params"),
            **metrics,
        }
        rows.append({"model": key, **detail[key]})
        print({"eval": key, "f1@0.5": metrics["f1@0.5"], "none_rate": metrics["none_of_k_rate"], "p95": metrics["detected_ae_p95"]}, flush=True)

    return pd.DataFrame(rows), pred_map, detail


def load_baseline_preds(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    bp = artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds"
    out = {}
    for name in ["STEAD_top1", "IDA_top1", "fixed_rescore_STEAD", "fixed_rescore_UNION", "prob_heuristic_UNION"]:
        p = bp / f"{name}.npy"
        if p.exists():
            arr = np.load(p)
            assert len(arr) == len(meta), (name, len(arr), len(meta))
            out[name] = arr.astype(float)
    # oracle from union
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    # closest candidate per trace
    preds = []
    by = {str(tn): g for tn, g in union.groupby(union.trace_name.astype(str), sort=False)}
    for i, tn in enumerate(meta.trace_name.astype(str)):
        g = by.get(tn)
        t, s = true[i], sr[i]
        if g is None or len(g) == 0 or not np.isfinite(t):
            preds.append(np.nan)
            continue
        samp = g["candidate_sample"].to_numpy(float)
        ae = np.abs((samp - t) / s)
        preds.append(float(samp[int(np.argmin(ae))]))
    out["oracle_UNION"] = np.asarray(preds, float)
    return out



def _fast_metric_bundle(pred: np.ndarray, true: np.ndarray, sr: np.ndarray, none_mask: np.ndarray | None) -> dict[str, float]:
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    sr = np.asarray(sr, float)
    has_true = np.isfinite(true)
    has_pred = np.isfinite(pred)
    if none_mask is None:
        none_mask = ~has_pred
    else:
        none_mask = np.asarray(none_mask, bool)
    err = np.full(len(true), np.nan)
    both = has_true & has_pred
    err[both] = np.abs((pred[both] - true[both]) / sr[both])
    out: dict[str, float] = {}
    for w, fkey, pkey, rkey in [
        (0.1, "f1@0.1", None, None),
        (0.5, "f1@0.5", "precision@0.5", "recall@0.5"),
    ]:
        matched = both & (err <= w)
        tp = int(matched.sum())
        fp = int((has_pred & ~matched).sum())
        fn = int((has_true & ~matched).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[fkey] = 2 * prec * rec / max(prec + rec, 1e-12)
        if pkey:
            out[pkey] = prec
            out[rkey] = rec
    out["miss_rate"] = float((has_true & ~has_pred).sum() / max(int(has_true.sum()), 1))
    out["none_of_k_rate"] = float(none_mask.mean())
    out["wrong_peak_rate"] = float(((both) & (err > 0.5)).sum() / max(int(has_true.sum()), 1))
    det = err[np.isfinite(err)]
    out["detected_ae_p95"] = float(np.percentile(det, 95)) if det.size else float("nan")
    return out


def event_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    event_ids: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 20260817,
    none_a: np.ndarray | None = None,
    none_b: np.ndarray | None = None,
) -> dict:
    """Paired event-level bootstrap of metric deltas: a - b."""
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    idx_by: dict[str, list[int]] = {}
    for i, e in enumerate(event_ids.astype(str)):
        idx_by.setdefault(e, []).append(i)
    event_lists = [np.asarray(idx_by[e], dtype=int) for e in events]
    if none_a is None:
        none_a = ~np.isfinite(pred_a)
    if none_b is None:
        none_b = ~np.isfinite(pred_b)
    none_a = np.asarray(none_a, bool)
    none_b = np.asarray(none_b, bool)

    base_a = _fast_metric_bundle(pred_a, true, sr, none_a)
    base_b = _fast_metric_bundle(pred_b, true, sr, none_b)
    keys = [
        "f1@0.5", "f1@0.1", "precision@0.5", "recall@0.5",
        "miss_rate", "none_of_k_rate", "wrong_peak_rate", "detected_ae_p95",
    ]
    deltas = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        draw = rng.integers(0, len(event_lists), size=len(event_lists))
        ix = np.concatenate([event_lists[j] for j in draw])
        ma = _fast_metric_bundle(pred_a[ix], true[ix], sr[ix], none_a[ix])
        mb = _fast_metric_bundle(pred_b[ix], true[ix], sr[ix], none_b[ix])
        for k in keys:
            deltas[k][b] = float(ma[k] - mb[k])

    out = {
        "n_boot": n_boot,
        "seed": seed,
        "n_events": int(len(events)),
        "n_traces": int(len(true)),
        "definition": "delta = method_a - baseline_b (positive means a better for F1/precision/recall; for miss/none/wrong/p95 positive means a worse)",
        "point_delta": {k: float(base_a[k] - base_b[k]) for k in keys},
        "metrics_a": {k: base_a[k] for k in keys},
        "metrics_b": {k: base_b[k] for k in keys},
    }
    for k in keys:
        arr = deltas[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {
            "mean_delta": float(arr.mean()),
            "ci95": [float(lo), float(hi)],
            "ci_direction": (
                "a_better_stable"
                if (k.startswith("f1") or k.startswith("precision") or k.startswith("recall")) and lo > 0
                else "a_worse_stable"
                if (k.startswith("f1") or k.startswith("precision") or k.startswith("recall")) and hi < 0
                else "a_lower_stable"
                if k in {"miss_rate", "none_of_k_rate", "wrong_peak_rate", "detected_ae_p95"} and hi < 0
                else "a_higher_stable"
                if k in {"miss_rate", "none_of_k_rate", "wrong_peak_rate", "detected_ae_p95"} and lo > 0
                else "inconclusive"
            ),
        }
    return out


def none_of_k_audit(
    r3: pd.DataFrame,
    fixed: np.ndarray,
    meta: pd.DataFrame,
    feat: pd.DataFrame,
    union: pd.DataFrame,
    out: Path,
) -> dict:
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    names = meta["trace_name"].astype(str).to_numpy()
    pred_r = align_preds_to_meta(r3, meta, "pred_reported")
    none = align_preds_to_meta(r3, meta, "none_of_k") > 0.5
    # rebuild none from selected_index for safety
    none = r3.set_index("trace_name")["none_of_k"].reindex(names).fillna(True).to_numpy(bool)

    # oracle correct peak exists within 0.5
    oracle_ok = np.zeros(len(meta), bool)
    by = {str(tn): g for tn, g in union.groupby(union.trace_name.astype(str), sort=False)}
    for i, tn in enumerate(names):
        g = by.get(str(tn))
        if g is None or not np.isfinite(true[i]):
            continue
        ae = np.abs((g["candidate_sample"].to_numpy(float) - true[i]) / sr[i])
        oracle_ok[i] = bool((ae <= 0.5).any())

    fixed_ok = correct_at(fixed, true, sr, 0.5)
    r3_ok = correct_at(pred_r, true, sr, 0.5)

    # transitions
    both_correct = fixed_ok & r3_ok
    fixed_only = fixed_ok & ~r3_ok & ~none
    r3_only = ~fixed_ok & r3_ok
    both_wrong = ~fixed_ok & ~r3_ok & ~none
    fixed_correct_r3_none = fixed_ok & none
    fixed_wrong_r3_none = ~fixed_ok & none
    both_none = none & ~np.isfinite(fixed)  # fixed should always finite
    # also treat fixed always predicts
    labels = np.array(["other"] * len(meta), dtype=object)
    labels[both_correct] = "both_correct"
    labels[fixed_only] = "fixed_only_correct"
    labels[r3_only] = "r3_only_correct"
    labels[both_wrong] = "both_wrong"
    labels[fixed_correct_r3_none] = "fixed_correct_r3_none"
    labels[fixed_wrong_r3_none] = "fixed_wrong_r3_none"

    trans = pd.DataFrame(
        {
            "trace_name": names,
            "event_id": meta["event_id"].astype(str).to_numpy(),
            "transition": labels,
            "none_of_k": none,
            "oracle_has_correct_peak_0.5": oracle_ok,
            "fixed_correct_0.5": fixed_ok,
            "r3_correct_0.5": r3_ok,
            "fixed_ae": np.where(np.isfinite(fixed), np.abs((fixed - true) / sr), np.nan),
            "r3_ae": np.where(np.isfinite(pred_r), np.abs((pred_r - true) / sr), np.nan),
        }
    )
    # attach station from meta if present
    if "station" in meta.columns:
        trans["station"] = meta["station"].astype(str).to_numpy()
    elif "station_id" in meta.columns:
        trans["station"] = meta["station_id"].astype(str).to_numpy()
    else:
        # derive from trace_name EVENT.NET.STA..CH
        trans["station"] = [tn.split(".")[2] if tn.count(".") >= 2 else tn for tn in names]

    # feature-level strata from first candidate row
    tr = feat.sort_values("candidate_index").groupby("trace_name", sort=False).first()
    hist_av = tr["history_available"].reindex(names).fillna(0).to_numpy(float) > 0.5
    fallback = tr["fallback_level_norm"].reindex(names).fillna(0).to_numpy(float)
    both_sup = tr["both_support"].reindex(names).fillna(0).to_numpy(float) > 0.5
    disagree = tr["model_time_disagreement_s"].reindex(names).to_numpy(float)
    # max candidate prob
    maxp = feat.assign(_p=np.fmax(feat["prob_stead"].fillna(-1), feat["prob_ida"].fillna(-1))).groupby("trace_name")["_p"].max()
    maxp = maxp.reindex(names).to_numpy(float)
    p_bin = np.where(maxp >= 0.7, "high", np.where(maxp >= 0.3, "medium", "low"))

    trans["history_available"] = hist_av
    trans["exact_history"] = hist_av & (fallback < 0.34)
    trans["fallback_history"] = hist_av & (fallback >= 0.34)
    trans["dual_source"] = both_sup
    trans["prob_bin"] = p_bin
    trans["has_model_disagreement"] = np.isfinite(disagree) & (disagree > 0.05)

    trans.to_csv(out / "fixed_to_ranker_transitions.csv", index=False)

    # P95 contribution: remove fixed large-error samples that R3 abstained
    fixed_ae = trans["fixed_ae"].to_numpy(float)
    r3_detected = np.isfinite(pred_r)
    # counterfactual: if R3 used fixed on none, what P95?
    cf = pred_r.copy()
    cf[none] = fixed[none]
    p95_r3 = float(np.percentile(np.abs((pred_r[r3_detected] - true[r3_detected]) / sr[r3_detected]), 95))
    p95_fixed = float(np.percentile(np.abs((fixed - true) / sr), 95))
    p95_cf = float(np.percentile(np.abs((cf - true) / sr), 95))
    # among R3 abstentions, how many were fixed AE > fixed P95 / >10s / >30s
    abs_none = none
    big = {
        "n_none": int(none.sum()),
        "none_with_fixed_ae_gt_p95": int((abs_none & (fixed_ae > p95_fixed)).sum()),
        "none_with_fixed_ae_gt_10s": int((abs_none & (fixed_ae > 10)).sum()),
        "none_with_fixed_ae_gt_30s": int((abs_none & (fixed_ae > 30)).sum()),
        "none_with_fixed_correct_0.5": int(fixed_correct_r3_none.sum()),
        "none_with_fixed_wrong_0.5": int(fixed_wrong_r3_none.sum()),
        "p95_r3": p95_r3,
        "p95_fixed": p95_fixed,
        "p95_r3_none_fallback_fixed": p95_cf,
        "p95_improvement_r3_vs_fixed": p95_r3 - p95_fixed,
        "p95_improvement_explained_by_abstention": (p95_r3 - p95_cf),
        "note": "If p95_r3_none_fallback_fixed ≈ p95_fixed, abstention drives most P95 gain",
    }

    # conditional metrics when R3 outputs a real candidate
    mask_pick = ~none
    metrics_when_pick = comprehensive_pick_metrics(pred_r[mask_pick], true[mask_pick], sr[mask_pick], none_mask=np.zeros(mask_pick.sum(), bool))

    # concentration
    ev_loss = trans[trans.transition == "fixed_correct_r3_none"].groupby("event_id").size().sort_values(ascending=False)
    st_loss = trans[trans.transition == "fixed_correct_r3_none"].groupby("station").size().sort_values(ascending=False)

    def rate(mask):
        return {
            "n": int(mask.sum()),
            "none_rate": float(none[mask].mean()) if mask.any() else float("nan"),
            "fixed_correct_r3_none_rate": float((fixed_correct_r3_none & mask).sum() / max(mask.sum(), 1)),
        }

    doc = {
        "overall_none_of_k_rate": float(none.mean()),
        "none_rate_when_oracle_has_correct_peak": float(none[oracle_ok].mean()) if oracle_ok.any() else None,
        "none_rate_when_oracle_no_correct_peak": float(none[~oracle_ok].mean()) if (~oracle_ok).any() else None,
        "n_fixed_correct_r3_none": int(fixed_correct_r3_none.sum()),
        "n_fixed_wrong_r3_none": int(fixed_wrong_r3_none.sum()),
        "n_r3_none_counted_as_fn_at_0.5": int((none & np.isfinite(true)).sum()),  # all none on labeled are FN for recall
        "metrics_when_r3_outputs_candidate": metrics_when_pick,
        "transition_counts": {k: int((labels == k).sum()) for k in sorted(set(labels))},
        "p95_abstention_decomposition": big,
        "strata": {
            "history_available": rate(hist_av),
            "history_unavailable": rate(~hist_av),
            "exact_history": rate(trans["exact_history"].to_numpy(bool)),
            "fallback_history": rate(trans["fallback_history"].to_numpy(bool)),
            "dual_source": rate(both_sup),
            "single_source": rate(~both_sup),
            "prob_high": rate(p_bin == "high"),
            "prob_medium": rate(p_bin == "medium"),
            "prob_low": rate(p_bin == "low"),
            "model_disagreement": rate(trans["has_model_disagreement"].to_numpy(bool)),
        },
        "concentration": {
            "top10_events_share_of_fixed_correct_r3_none": float(ev_loss.head(10).sum() / max(int(fixed_correct_r3_none.sum()), 1)),
            "top10_stations_share_of_fixed_correct_r3_none": float(st_loss.head(10).sum() / max(int(fixed_correct_r3_none.sum()), 1)),
            "top10_events": ev_loss.head(10).astype(int).to_dict(),
            "top10_stations": st_loss.head(10).astype(int).to_dict(),
        },
    }
    save_json(doc, out / "none_of_k_audit.json")
    return doc


def implementation_audit() -> dict:
    """Static + light dynamic checks; no training."""
    findings = []
    # mask fill
    findings.append(
        {
            "check": "padding_mask_fill",
            "file": "src/earthquake/stage6/ranker/models.py",
            "status": "ok",
            "evidence": "scores.masked_fill(~mask, neg) before cat with none logit; padded slots get -inf-like score",
        }
    )
    findings.append(
        {
            "check": "none_class_index",
            "file": "src/earthquake/stage6/ranker/dataset.py + models.py",
            "status": "ok",
            "evidence": "target=max_k(10) for none; logits shape [B,K+1]; eval uses pi>=max_k as none",
        }
    )
    findings.append(
        {
            "check": "candidate_sort_order",
            "file": "dataset.py",
            "status": "ok",
            "evidence": "groupby trace_name then sort_values(candidate_index); positive_index aligned to candidate_index",
        }
    )
    findings.append(
        {
            "check": "gamma_none_emphasis",
            "file": "models.py:listwise_loss",
            "status": "design_risk_not_bug",
            "evidence": "loss = ce + beta*pairwise + gamma*none_ce; gamma=1.0 doubles none-target CE pressure; hard_boost also upsamples ~25% none",
        }
    )
    findings.append(
        {
            "check": "checkpoint_selection_population",
            "file": "scripts/train_stage6_candidate_ranker.py",
            "status": "evaluation_inconsistency",
            "evidence": "When |dev|>12000, epoch selection uses rng.choice(..., size=12000) subset; overnight compares those subset metrics across seeds; full-dev only for final reported ckpt",
            "severity": "reported_best may differ from max full-dev F1 among existing ckpts",
        }
    )
    findings.append(
        {
            "check": "marginal_pass_via_p95",
            "file": "scripts/finalize_stage6_phaseC.py",
            "status": "gate_design",
            "evidence": "marginal_pass if (0.005<=d05<0.01 OR dp95<=-0.15) and direction_stable — P95 drop from abstention can trigger marginal_pass despite F1 regression",
        }
    )
    findings.append(
        {
            "check": "feature_norm_fit_train_only",
            "file": "features / dataset build",
            "status": "ok_assumed",
            "evidence": "features are raw / clipped z; no StandardScaler fit on dev labels; history from picker_train only",
        }
    )
    findings.append(
        {
            "check": "confirm_sealed",
            "status": "ok",
            "evidence": "assert_full_confirm_access_allowed raises without method_lock",
        }
    )
    # dynamic: padded never selected
    model = build_ranker("R3", n_features=len(__import__("earthquake.stage6.ranker.schema", fromlist=["R2_FEATURE_NAMES"]).R2_FEATURE_NAMES))
    model.eval()
    B, K, F = 4, 10, len(__import__("earthquake.stage6.ranker.schema", fromlist=["R2_FEATURE_NAMES"]).R2_FEATURE_NAMES)
    x = torch.randn(B, K, F)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[:, :3] = True
    with torch.no_grad():
        logits = model(x, mask)
        # force none logit very low
        logits2 = logits.clone()
        logits2[:, -1] = -1e9
        idx = logits2.argmax(-1)
    assert bool((idx < 3).all()), "padded candidates selected under forced non-none"
    findings.append({"check": "dynamic_padded_not_selected", "status": "ok", "evidence": "argmax among candidates with none suppressed stays in valid mask"})

    return {
        "bugs_blocking": [],
        "evaluation_inconsistencies": [f for f in findings if f.get("status") == "evaluation_inconsistency"],
        "design_risks": [f for f in findings if f.get("status") in {"design_risk_not_bug", "gate_design"}],
        "checks": findings,
        "implementation_or_evaluation_bug": False,
        "note": "12k-subset checkpoint selection is an evaluation inconsistency that can mis-rank seeds, but full-dev metrics recomputed here; no inference mask/index bug found that invalidates reported full-dev numbers.",
    }


def write_report(out: Path, verdict: dict, cohort: dict, none_doc: dict, sel: dict, boot: dict, impl: dict) -> None:
    md = f"""# Stage 6 Phase C.1 — Candidate Ranker Sanity Audit

**Confirm seal (start/end):** SEALED  
**Verdict:** `{verdict["verdict"]}`  
**Recommended Stage 6 main method:** `{verdict["recommended_stage6_main_method"]}`  
**Ranker role:** `{verdict["ranker_role"]}`  
**Multistation may start:** `{verdict["multistation_may_start"]}`  
**Confirm may unseal:** `{verdict["confirm_may_unseal"]}`

## Cohort

- events={cohort["n_events"]}, traces={cohort["n_traces"]}, matches_expected={cohort["matches_expected"]}
- event_set_sha256=`{cohort["event_id_set_sha256"]}`
- trace_set_sha256=`{cohort["trace_name_set_sha256"]}`
- confirm overlap events/traces: {cohort["contamination_checks"]["confirm_event_overlap_n"]} / {cohort["contamination_checks"]["confirm_trace_overlap_n"]}

## Reported R3 vs fixed_rescore_UNION

See `phaseC1_final_verdict.json` for full numbers.

## Selection audit

- reported_best_ranker: `{sel["reported_best_ranker"]}`
- max_dev_f1_existing_ranker: `{sel["max_dev_f1_existing_ranker"]}`
- selection used ~12k-subset metrics during training/overnight seed pick: **yes**

## none_of_k / P95

- overall none rate: {none_doc["overall_none_of_k_rate"]:.4f}
- none | oracle has correct peak: {none_doc["none_rate_when_oracle_has_correct_peak"]:.4f}
- none | oracle no correct peak: {none_doc["none_rate_when_oracle_no_correct_peak"]:.4f}
- fixed_correct→r3_none: {none_doc["n_fixed_correct_r3_none"]}
- fixed_wrong→r3_none: {none_doc["n_fixed_wrong_r3_none"]}
- P95 abstention note: {none_doc["p95_abstention_decomposition"]["note"]}
- p95_r3={none_doc["p95_abstention_decomposition"]["p95_r3"]:.3f}, p95_fixed={none_doc["p95_abstention_decomposition"]["p95_fixed"]:.3f}, p95_none_fallback_fixed={none_doc["p95_abstention_decomposition"]["p95_r3_none_fallback_fixed"]:.3f}

## Implementation

- blocking bugs: {impl.get("bugs_blocking")}
- evaluation inconsistencies: {len(impl.get("evaluation_inconsistencies", []))}
- original overnight `marginal_pass` reinterpreted as: `{verdict.get("marginal_pass_reinterpretation")}`

## Artifacts

All under `artifacts/results/stage6/phaseC1/`. Original `phaseC_final_verdict.json` untouched.
"""
    (ROOT / "reports/stage6/phaseC1_ranker_sanity_audit.md").write_text(md)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC1")
    seal_start = _guard_confirm_sealed()

    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    # prefer s_arrival_sample
    if "s_arrival_sample" not in meta.columns and "true_s_sample" in meta.columns:
        meta = meta.rename(columns={"true_s_sample": "s_arrival_sample"})
    cohort = cohort_provenance(meta, out)
    assert cohort["contamination_checks"]["confirm_event_overlap_n"] == 0
    assert cohort["contamination_checks"]["confirm_trace_overlap_n"] == 0

    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    event_ids = meta["event_id"].astype(str).to_numpy()

    # baselines
    baselines = load_baseline_preds(meta)
    base_rows = []
    for name, pred in baselines.items():
        none = ~np.isfinite(pred)
        m = comprehensive_pick_metrics(pred, true, sr, none_mask=none)
        base_rows.append({"model": name, **m})
        print({"baseline": name, "f1@0.5": m["f1@0.5"], "p95": m["detected_ae_p95"], "cov": m["prediction_coverage"]}, flush=True)

    # all rankers
    ranker_df, pred_map, detail = eval_all_rankers(device, meta)
    all_df = pd.concat([pd.DataFrame(base_rows), ranker_df], ignore_index=True)
    all_df.to_csv(out / "all_existing_ranker_metrics.csv", index=False)

    reported_name = "R3_seed2026_b0.2_g1.0"
    reported_ckpt = artifacts_dir() / "models" / "stage6" / "ranker" / reported_name / "best.pt"
    assert reported_ckpt.exists()
    max_row = ranker_df.loc[ranker_df["f1@0.5"].idxmax()]
    sel = {
        "reported_best_ranker": reported_name,
        "reported_selection_metric_in_ckpt": detail[reported_name]["selection_metric_s_f1@0.5"],
        "reported_full_dev_f1@0.5": float(detail[reported_name]["f1@0.5"]),
        "max_dev_f1_existing_ranker": str(max_row["model"]),
        "max_dev_f1@0.5": float(max_row["f1@0.5"]),
        "selection_objective": "overnight compared training-time s_f1@0.5 stored in best.pt (12k-subset eval); final full-dev only for winner",
        "overnight_beta_gamma_for_R3": "fixed to beta=0.2 gamma=1.0 after R2 structure pick (R2 grid used F1@0.5 on subset)",
        "mismatch": str(max_row["model"]) != reported_name,
        "mismatch_reason": (
            "Training/overnight selection used ~12k-subset F1; full-dev recomputation can reorder models. "
            "Do not rebrand max-F1 as pre-registered best."
            if str(max_row["model"]) != reported_name
            else "reported best also has max full-dev F1 among existing rankers"
        ),
    }
    save_json(sel, out / "ranker_selection_audit.json")

    # R3 variants
    feat = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R2.parquet")
    union = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    fixed = baselines["fixed_rescore_UNION"]
    r3_var = predict_ranker_variants(
        reported_ckpt,
        feat,
        device=device,
        fixed_fallback=fixed,
        meta_trace_names=meta["trace_name"].astype(str).tolist(),
    )
    r3_var.to_parquet(out / "r3_diagnostic_predictions.parquet", index=False)

    diag = {}
    for col, label in [
        ("pred_reported", "R3_reported"),
        ("pred_forced_choice", "R3_forced_choice"),
        ("pred_none_fallback_fixed", "R3_none_fallback_fixed"),
    ]:
        pred = align_preds_to_meta(r3_var, meta, col)
        none = ~np.isfinite(pred) if col != "pred_reported" else align_preds_to_meta(r3_var, meta, "none_of_k") > 0.5
        if col == "pred_reported":
            none = r3_var.set_index("trace_name")["none_of_k"].reindex(meta.trace_name.astype(str)).fillna(True).to_numpy(bool)
        elif col == "pred_forced_choice":
            none = ~np.isfinite(pred)
        else:
            none = ~np.isfinite(pred)
        diag[label] = comprehensive_pick_metrics(pred, true, sr, none_mask=none)
        diag[label]["role"] = "posthoc_diagnostic" if label != "R3_reported" else "reported"
    # also metrics for max-F1 ranker
    max_pred = pred_map[str(max_row["model"])]
    diag["max_f1_existing_ranker"] = comprehensive_pick_metrics(max_pred, true, sr, none_mask=~np.isfinite(max_pred))
    diag["max_f1_existing_ranker"]["model"] = str(max_row["model"])
    diag["fixed_rescore_UNION"] = comprehensive_pick_metrics(fixed, true, sr, none_mask=~np.isfinite(fixed))
    save_json(diag, out / "diagnostic_variants_metrics.json")

    none_doc = none_of_k_audit(r3_var, fixed, meta, feat, union, out)

    # bootstrap
    boot = {}
    pairs = {
        "R3_reported_vs_fixed": (
            align_preds_to_meta(r3_var, meta, "pred_reported"),
            fixed,
            r3_var.set_index("trace_name")["none_of_k"].reindex(meta.trace_name.astype(str)).fillna(True).to_numpy(bool),
            ~np.isfinite(fixed),
        ),
        "max_f1_ranker_vs_fixed": (max_pred, fixed, ~np.isfinite(max_pred), ~np.isfinite(fixed)),
        "R3_forced_choice_vs_fixed": (
            align_preds_to_meta(r3_var, meta, "pred_forced_choice"),
            fixed,
            ~np.isfinite(align_preds_to_meta(r3_var, meta, "pred_forced_choice")),
            ~np.isfinite(fixed),
        ),
        "R3_none_fallback_fixed_vs_fixed": (
            align_preds_to_meta(r3_var, meta, "pred_none_fallback_fixed"),
            fixed,
            ~np.isfinite(align_preds_to_meta(r3_var, meta, "pred_none_fallback_fixed")),
            ~np.isfinite(fixed),
        ),
    }
    for name, (pa, pb, na, nb) in pairs.items():
        print({"bootstrap": name}, flush=True)
        boot[name] = event_bootstrap(pa, pb, true, sr, event_ids, none_a=na, none_b=nb)
    save_json(boot, out / "bootstrap_vs_fixed.json")

    impl = implementation_audit()
    save_json(impl, out / "implementation_audit.json")

    # final verdict rules
    r3m = diag["R3_reported"]
    fxm = diag["fixed_rescore_UNION"]
    d05 = r3m["f1@0.5"] - fxm["f1@0.5"]
    d01 = r3m["f1@0.1"] - fxm["f1@0.1"]
    dp95 = r3m["detected_ae_p95"] - fxm["detected_ae_p95"]
    ci = boot["R3_reported_vs_fixed"]["f1@0.5"]["ci95"]
    ci_lo = ci[0]

    posthoc_pass = False
    for lab in ["R3_forced_choice", "R3_none_fallback_fixed"]:
        dm = diag[lab]
        dd05 = dm["f1@0.5"] - fxm["f1@0.5"]
        dd01 = dm["f1@0.1"] - fxm["f1@0.1"]
        ddp95 = dm["detected_ae_p95"] - fxm["detected_ae_p95"]
        bkey = f"{lab}_vs_fixed"
        blo = boot[bkey]["f1@0.5"]["ci95"][0]
        if dd05 >= 0.01 and blo > 0 and dd01 >= -0.003 and ddp95 <= 0:
            posthoc_pass = True

    if impl.get("implementation_or_evaluation_bug"):
        verdict_name = "implementation_or_evaluation_bug"
        main_method = "investigate_before_method_lock"
        ranker_role = "invalid_until_fix"
        multi = False
        unseal = False
    elif posthoc_pass:
        verdict_name = "posthoc_rescue_candidate_requires_method_lock"
        main_method = "fixed_rescore_UNION"
        ranker_role = "posthoc_diagnostic_only"
        multi = False
        unseal = False
    else:
        # strong-pass check on reported R3
        strong = (
            d05 >= 0.01
            and ci_lo > 0
            and d01 >= -0.003
            and dp95 <= 0
            and r3m["recall@0.5"] >= fxm["recall@0.5"] - 0.01  # no clear selective recall collapse
        )
        if strong:
            verdict_name = "strong_pass"  # unexpected
            main_method = reported_name
            ranker_role = "primary"
            multi = False  # still require user for multistation
            unseal = False
        else:
            verdict_name = "ranker_failed_predeclared_gate"
            main_method = "fixed_rescore_UNION"
            ranker_role = "negative_ablation_or_selective_quality_filter"
            multi = False
            unseal = False

    # seed consistency among R3 seeds on full-dev
    r3_scores = [float(detail[k]["f1@0.5"]) for k in detail if k.startswith("R3_")]
    seed_consistent = (max(r3_scores) - min(r3_scores)) <= 0.01 if len(r3_scores) >= 2 else None

    final = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "confirm_seal_start": seal_start["status"],
        "eval_scope": "phaseB_full_s_labelled_dev",
        "eval_events": cohort["n_events"],
        "eval_s_traces": cohort["n_traces"],
        "event_set_sha256": cohort["event_id_set_sha256"],
        "trace_set_sha256": cohort["trace_name_set_sha256"],
        "reported_best_ranker": reported_name,
        "max_dev_f1_existing_ranker": sel["max_dev_f1_existing_ranker"],
        "fixed_rescore_UNION": {k: fxm[k] for k in ["precision@0.5", "recall@0.5", "f1@0.1", "f1@0.5", "prediction_coverage", "none_of_k_rate", "miss_rate", "matched_count_0.5", "detected_ae_p95", "detected_ae_p95_denominator"]},
        "R3_reported": {k: r3m[k] for k in ["precision@0.5", "recall@0.5", "f1@0.1", "f1@0.5", "prediction_coverage", "none_of_k_rate", "miss_rate", "matched_count_0.5", "detected_ae_p95", "detected_ae_p95_denominator"]},
        "delta_r3_minus_fixed": {"f1@0.5": d05, "f1@0.1": d01, "detected_ae_p95": dp95, "recall@0.5": r3m["recall@0.5"] - fxm["recall@0.5"], "none_rate": r3m["none_of_k_rate"] - fxm["none_of_k_rate"]},
        "p95_improvement_mainly_from_abstention": abs(none_doc["p95_abstention_decomposition"]["p95_r3_none_fallback_fixed"] - fxm["detected_ae_p95"]) < 0.25,
        "bootstrap_R3_vs_fixed_f1_05_ci": ci,
        "diagnostic_variants": {k: {kk: diag[k][kk] for kk in ["f1@0.1", "f1@0.5", "none_of_k_rate", "miss_rate", "detected_ae_p95", "prediction_coverage"]} for k in ["R3_forced_choice", "R3_none_fallback_fixed"]},
        "seed_consistent_full_dev_R3": seed_consistent,
        "implementation_or_evaluation_bug": bool(impl.get("implementation_or_evaluation_bug")),
        "verdict": verdict_name,
        "marginal_pass_reinterpretation": "timing_only_tradeoff_with_recall_or_f1_regression",
        "recommended_stage6_main_method": main_method,
        "ranker_role": ranker_role,
        "multistation_may_start": multi,
        "confirm_may_unseal": unseal,
        "confirm_remains_sealed": True,
        "original_phaseC_verdict_untouched": True,
        "original_overnight_verdict": "marginal_pass",
    }

    seal_end = _guard_confirm_sealed()
    final["confirm_seal_end"] = seal_end["status"]
    save_json(final, out / "phaseC1_final_verdict.json")
    write_report(out, final, cohort, none_doc, sel, boot, impl)
    (out / "PHASEC1.DONE").write_text("DONE\n")
    print(json.dumps({k: final[k] for k in ["verdict", "recommended_stage6_main_method", "R3_reported", "fixed_rescore_UNION", "p95_improvement_mainly_from_abstention", "max_dev_f1_existing_ranker"]}, indent=2))


if __name__ == "__main__":
    main()
