"""Stage 6 Phase B: candidate oracle, union, complementarity, event bootstrap (no training)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from earthquake.fusion.peak_candidates import extract_candidates
from earthquake.metrics import match_picks, report_pick_timing_bundle


CAND_EXTRACT_CFG = {
    "k": 10,
    "min_distance": 50,
    "min_prominence": 0.05,
    "min_probability": 0.1,
    "phase": "S",
}
UNION_DEDUP_S = 0.05
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260815


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_ida_best_checkpoint(
    ckpt_path: Path,
    *,
    stead_weight: str = "stead",
    smoke_rows: pd.DataFrame | None = None,
    waveform_reader=None,
    device: str = "cpu",
    expected_epoch: int = 14,
) -> dict[str, Any]:
    """Validate ID-A best.pt; never loads last.pt."""
    import torch
    import seisbench.models as sbm

    from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
    from earthquake.stage6.bn_policy import set_train_bn_eval

    if ckpt_path.name != "best.pt":
        return {"ok": False, "reason": f"refusing non-best checkpoint name: {ckpt_path.name}"}
    if "last.pt" in str(ckpt_path):
        return {"ok": False, "reason": "refusing last.pt"}

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = int(blob.get("epoch", -999))
    state = blob["model"]
    n_nan = 0
    n_inf = 0
    n_params = 0
    for v in state.values():
        if not torch.is_floating_point(v):
            continue
        n_params += int(v.numel())
        n_nan += int(torch.isnan(v).sum().item())
        n_inf += int(torch.isinf(v).sum().item())

    label_order = str(blob.get("label_order", "PSN"))
    digest = sha256_file(ckpt_path)

    ok = (
        epoch == expected_epoch
        and n_nan == 0
        and n_inf == 0
        and label_order == "PSN"
        and n_params > 0
    )
    out: dict[str, Any] = {
        "ok": bool(ok),
        "path": str(ckpt_path),
        "sha256": digest,
        "epoch": epoch,
        "expected_epoch": expected_epoch,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_float_params": n_params,
        "label_order": label_order,
        "loaded_last_pt": False,
        "bn_policy": "eval_frozen_running_stats",
    }
    if epoch != expected_epoch:
        out["reason"] = f"epoch={epoch} != expected {expected_epoch}"
        out["ok"] = False
        return out
    if n_nan or n_inf:
        out["reason"] = "nonfinite parameters"
        out["ok"] = False
        return out

    # BN consistency: running stats should match STEAD (frozen during finetune)
    stead = sbm.PhaseNet.from_pretrained(stead_weight)
    ida = sbm.PhaseNet.from_pretrained(stead_weight)
    ida.load_state_dict(state, strict=True)
    set_train_bn_eval(ida)
    ida.eval()
    bn_max_diff = 0.0
    n_bn = 0
    for (n1, m1), (n2, m2) in zip(stead.named_modules(), ida.named_modules()):
        if isinstance(m1, torch.nn.modules.batchnorm._BatchNorm):
            n_bn += 1
            d = float(torch.max(torch.abs(m1.running_mean - m2.running_mean)).item())
            d = max(d, float(torch.max(torch.abs(m1.running_var - m2.running_var)).item()))
            bn_max_diff = max(bn_max_diff, d)
    out["n_bn_layers"] = n_bn
    out["bn_running_max_abs_diff_vs_stead"] = bn_max_diff
    # allow tiny float noise; large drift means BN was updated
    if bn_max_diff > 1e-3:
        out["ok"] = False
        out["reason"] = f"BN running stats drifted vs STEAD: {bn_max_diff}"
        return out

    if smoke_rows is not None and waveform_reader is not None and len(smoke_rows):
        ref = SeisBenchPhaseNetReference(weight=stead_weight, device=device)
        ref.model = ida.to(device)
        ref.model.eval()
        n_fail = 0
        n_nan_pred = 0
        with torch.no_grad():
            for _, row in smoke_rows.iterrows():
                try:
                    wave = waveform_reader.read_waveform(str(row.trace_name))
                    pred = ref.predict_row(wave, row, remap_to_waveform=True)
                    s = np.asarray(pred["s_proba_on_waveform"], dtype=np.float64)
                    p = np.asarray(pred["p_proba_on_waveform"], dtype=np.float64)
                    if (not np.isfinite(s).all()) or (not np.isfinite(p).all()):
                        n_nan_pred += 1
                except Exception:  # noqa: BLE001
                    n_fail += 1
        out["smoke_n"] = int(len(smoke_rows))
        out["smoke_fail"] = n_fail
        out["smoke_nan_pred"] = n_nan_pred
        out["utc_remap"] = True
        if n_fail or n_nan_pred:
            out["ok"] = False
            out["reason"] = f"smoke inference fail={n_fail} nan_pred={n_nan_pred}"
            return out

    if not ok:
        out["reason"] = out.get("reason", "validation failed")
    return out


def build_s_labelled_eval_manifest(
    events: pd.DataFrame,
    *,
    split_trace_names: list[str],
    split_event_ids: list[str],
    seed: int = 20260815,
    max_traces: int | None = None,
    max_traces_per_event: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fixed Stage6_dev S-labelled eval list. No model preds; event-aware optional subsample."""
    allowed_tr = set(split_trace_names)
    allowed_ev = set(str(e) for e in split_event_ids)
    df = events[events["trace_name"].astype(str).isin(allowed_tr)].copy()
    df = df[df["event_id"].astype(str).isin(allowed_ev)]
    s = pd.to_numeric(df["s_arrival_sample"], errors="coerce")
    df = df.loc[s.notna()].copy()
    df["trace_name"] = df["trace_name"].astype(str)
    df["event_id"] = df["event_id"].astype(str)

    rng = np.random.default_rng(seed)
    if max_traces_per_event is not None:
        parts = []
        for eid, g in df.groupby("event_id", sort=True):
            if len(g) <= max_traces_per_event:
                parts.append(g)
            else:
                idx = rng.choice(len(g), size=max_traces_per_event, replace=False)
                parts.append(g.iloc[np.sort(idx)])
        df = pd.concat(parts, ignore_index=True)

    if max_traces is not None and len(df) > max_traces:
        # event-unit sample: shuffle events, take all traces until budget
        eids = df["event_id"].drop_duplicates().tolist()
        rng.shuffle(eids)
        keep = []
        n = 0
        for eid in eids:
            g = df[df["event_id"] == eid]
            if n >= max_traces:
                break
            keep.append(g)
            n += len(g)
        df = pd.concat(keep, ignore_index=True)

    # stable order: by origin/event then trace
    if "origin_time" in df.columns:
        df = df.sort_values(["origin_time", "event_id", "trace_name"]).reset_index(drop=True)
    else:
        df = df.sort_values(["event_id", "trace_name"]).reset_index(drop=True)

    summary = {
        "n_events": int(df["event_id"].nunique()),
        "n_traces": int(len(df)),
        "n_s_labelled_traces": int(len(df)),
        "n_stations": int(df["station"].nunique()) if "station" in df.columns else None,
        "time_start": str(df["origin_time"].min()) if "origin_time" in df.columns else None,
        "time_end": str(df["origin_time"].max()) if "origin_time" in df.columns else None,
        "traces_per_event": df.groupby("event_id").size().describe().to_dict(),
        "seed": seed,
        "max_traces": max_traces,
        "max_traces_per_event": max_traces_per_event,
        "sampling": "all_s_labelled" if max_traces is None else "event_unit_cap",
    }
    if "distance_km" in df.columns:
        summary["distance_km"] = df["distance_km"].describe().to_dict()
    if "source_depth_km" in df.columns:
        summary["source_depth_km"] = df["source_depth_km"].describe().to_dict()
    if "channel_prefix" in df.columns:
        summary["channel_prefix_counts"] = df["channel_prefix"].astype(str).value_counts().to_dict()
    return df, summary


def candidates_from_s_proba(
    s_proba: np.ndarray,
    *,
    sampling_rate: float,
    waveform_starttime,
    k: int = 10,
    source: str,
) -> list[dict[str, Any]]:
    cfg = dict(CAND_EXTRACT_CFG)
    cfg["k"] = int(k)
    cands = extract_candidates(
        np.asarray(s_proba, dtype=float),
        phase="S",
        k=int(k),
        min_distance=int(cfg["min_distance"]),
        min_prominence=float(cfg["min_prominence"]),
        min_probability=float(cfg["min_probability"]),
        sampling_rate=float(sampling_rate),
        waveform_starttime=waveform_starttime,
    )
    rows = []
    for c in cands:
        rows.append(
            {
                "candidate_sample": int(c.sample_index),
                "candidate_time_utc": c.absolute_utc,
                "candidate_probability": float(c.peak_probability),
                "candidate_rank": int(c.rank),
                "candidate_source": source,
                "fallback_peak": bool(c.fallback_peak),
            }
        )
    return rows


def topk_from_cache(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df[df["candidate_rank"] < int(k)].copy()
    return out


def union_candidates(
    stead: pd.DataFrame,
    ida: pd.DataFrame,
    *,
    stead_k: int = 5,
    ida_k: int = 5,
    max_union: int = 10,
    dedup_s: float = UNION_DEDUP_S,
    sampling_rate_hz: float = 100.0,
) -> pd.DataFrame:
    """Union STEAD top-stead_k and ID-A top-ida_k with 0.05s dedup; cap at max_union.

    Does not average uncalibrated probabilities. Representative time prefers higher STEAD
    probability when both support; else the available source's time.
    """
    s = topk_from_cache(stead, stead_k)
    i = topk_from_cache(ida, ida_k)
    rows_out = []
    all_names = sorted(set(s["trace_name"].astype(str)).union(set(i["trace_name"].astype(str))))
    s_groups = {str(tn): g for tn, g in s.groupby(s["trace_name"].astype(str), sort=False)}
    i_groups = {str(tn): g for tn, g in i.groupby(i["trace_name"].astype(str), sort=False)}
    for tn in all_names:
        sg = s_groups.get(tn)
        ig = i_groups.get(tn)
        base = sg if sg is not None else ig
        assert base is not None
        sr = float(base["sampling_rate_hz"].iloc[0]) if "sampling_rate_hz" in base.columns else sampling_rate_hz
        eid = str(base["event_id"].iloc[0]) if "event_id" in base.columns else ""
        true_s = float(base["true_s_sample"].iloc[0]) if "true_s_sample" in base.columns else np.nan

        pool: list[dict[str, Any]] = []
        if sg is not None:
            for _, r in sg.iterrows():
                pool.append(
                    {
                        "sample": float(r["candidate_sample"]),
                        "prob_stead": float(r["candidate_probability"]),
                        "prob_ida": np.nan,
                        "rank_stead": int(r["candidate_rank"]),
                        "rank_ida": -1,
                        "from_stead": True,
                        "from_ida": False,
                        "time_utc": r.get("candidate_time_utc"),
                    }
                )
        if ig is not None:
            for _, r in ig.iterrows():
                samp = float(r["candidate_sample"])
                matched = None
                for p in pool:
                    if abs(p["sample"] - samp) / sr <= dedup_s:
                        matched = p
                        break
                if matched is None:
                    pool.append(
                        {
                            "sample": samp,
                            "prob_stead": np.nan,
                            "prob_ida": float(r["candidate_probability"]),
                            "rank_stead": -1,
                            "rank_ida": int(r["candidate_rank"]),
                            "from_stead": False,
                            "from_ida": True,
                            "time_utc": r.get("candidate_time_utc"),
                        }
                    )
                else:
                    matched["from_ida"] = True
                    matched["prob_ida"] = float(r["candidate_probability"])
                    matched["rank_ida"] = int(r["candidate_rank"])

        def sort_key(p: dict) -> tuple:
            joint = 1 if (p["from_stead"] and p["from_ida"]) else 0
            probs = [x for x in (p["prob_stead"], p["prob_ida"]) if np.isfinite(x)]
            mx = max(probs) if probs else -1.0
            rs = p["rank_stead"] if p["rank_stead"] >= 0 else 99
            return (-joint, -mx, rs)

        pool_sorted = sorted(pool, key=sort_key)[:max_union]
        for rank, p in enumerate(pool_sorted):
            rows_out.append(
                {
                    "trace_name": tn,
                    "event_id": eid,
                    "true_s_sample": true_s,
                    "sampling_rate_hz": sr,
                    "candidate_rank": rank,
                    "candidate_sample": p["sample"],
                    "candidate_time_utc": p["time_utc"],
                    "prob_stead": p["prob_stead"],
                    "prob_ida": p["prob_ida"],
                    "rank_stead": p["rank_stead"],
                    "rank_ida": p["rank_ida"],
                    "both_support": bool(p["from_stead"] and p["from_ida"]),
                    "source_mask": (
                        "stead+ida"
                        if p["from_stead"] and p["from_ida"]
                        else ("stead" if p["from_stead"] else "ida")
                    ),
                    "candidate_source": "union",
                    "n_union": len(pool_sorted),
                }
            )
    return pd.DataFrame(rows_out)


def oracle_pick_closest(cands: np.ndarray, true_sample: float, sr: float) -> float:
    c = np.asarray(cands, dtype=float)
    c = c[np.isfinite(c)]
    if (not np.isfinite(true_sample)) or c.size == 0 or (not np.isfinite(sr)) or sr <= 0:
        return float("nan")
    ae = np.abs((c - true_sample) / sr)
    return float(c[int(np.argmin(ae))])


def oracle_metrics_for_set(
    cand_df: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    sample_col: str = "candidate_sample",
) -> dict[str, Any]:
    """Oracle = closest candidate to human S; no new times outside candidate set."""
    meta = meta.reset_index(drop=True)
    by_trace: dict[str, np.ndarray] = {}
    for tn, g in cand_df.groupby(cand_df["trace_name"].astype(str), sort=False):
        by_trace[str(tn)] = g[sample_col].to_numpy(dtype=float)
    preds = []
    trues = []
    srs = []
    ranks_hit = []
    n_miss = 0
    n_cands = []
    closest_ae = []
    for _, row in meta.iterrows():
        tn = str(row["trace_name"])
        true = float(row["s_arrival_sample"]) if "s_arrival_sample" in row else float(row.get("true_s_sample", np.nan))
        sr = float(row.get("sampling_rate_hz", 100.0))
        samples = by_trace.get(tn)
        n_cands.append(0 if samples is None else int(samples.size))
        if samples is None or samples.size == 0:
            preds.append(np.nan)
            n_miss += 1
            closest_ae.append(np.nan)
            ranks_hit.append(np.nan)
        else:
            pred = oracle_pick_closest(samples, true, sr)
            preds.append(pred)
            if np.isfinite(pred) and np.isfinite(true):
                ae = abs(pred - true) / sr
                closest_ae.append(ae)
                ae_all = np.abs((samples - true) / sr)
                ranks_hit.append(int(np.argmin(ae_all)))
            else:
                closest_ae.append(np.nan)
                ranks_hit.append(np.nan)
                n_miss += 1
        trues.append(true)
        srs.append(sr)

    pred_a = np.asarray(preds, dtype=float)
    true_a = np.asarray(trues, dtype=float)
    sr_a = np.asarray(srs, dtype=float)
    m = match_picks(pred_a, true_a, sr_a, windows_s=(0.1, 0.2, 0.5))
    labeled = int(np.isfinite(true_a).sum())
    recalls = {}
    for w in (0.1, 0.2, 0.5):
        hit = 0
        for p, t, sr in zip(pred_a, true_a, sr_a):
            if not np.isfinite(t):
                continue
            if np.isfinite(p) and abs(p - t) / sr <= w:
                hit += 1
        recalls[w] = hit / max(labeled, 1)

    ae = np.asarray(closest_ae, dtype=float)
    ae_ok = ae[np.isfinite(ae)]
    rank_arr = np.asarray(ranks_hit, dtype=float)
    rank_ok = rank_arr[np.isfinite(rank_arr)].astype(int)
    rank_hist = {str(i): int((rank_ok == i).sum()) for i in range(0, 11)} if rank_ok.size else {}

    # Clarify F1 vs hit-rate when almost all have predictions
    note = (
        "Oracle F1 uses match_picks on closest-candidate predictions. "
        "With near-zero candidate misses, oracle F1@tol ≈ oracle recall@tol ≈ candidate hit-rate@tol "
        "(wrong peaks count as FP+FN)."
    )
    return {
        "n_traces": int(len(meta)),
        "n_labeled": labeled,
        "candidate_miss_rate": float(n_miss / max(labeled, 1)),
        "mean_n_candidates": float(np.mean(n_cands)) if n_cands else 0.0,
        "oracle_recall@0.1": float(recalls[0.1]),
        "oracle_recall@0.2": float(recalls[0.2]),
        "oracle_recall@0.5": float(recalls[0.5]),
        "oracle_f1@0.1": float(m["f1@0.1s"]),
        "oracle_f1@0.2": float(m["f1@0.2s"]),
        "oracle_f1@0.5": float(m["f1@0.5s"]),
        "closest_candidate_median_ae": float(np.median(ae_ok)) if ae_ok.size else float("nan"),
        "closest_candidate_p95": float(np.percentile(ae_ok, 95)) if ae_ok.size else float("nan"),
        "correct_candidate_rank_hist": rank_hist,
        "oracle_f1_vs_hitrate_note": note,
        "oracle_predictions": pred_a,
        "true_samples": true_a,
        "sampling_rates": sr_a,
        "event_ids": meta["event_id"].astype(str).to_numpy(),
    }


def complementarity_table(
    stead_cands: pd.DataFrame,
    ida_cands: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    k: int,
    tol_s: float,
) -> pd.DataFrame:
    """Four mutually exclusive classes at a tolerance."""
    rows = []
    s = topk_from_cache(stead_cands, k)
    i = topk_from_cache(ida_cands, k)
    s_map = {str(tn): g["candidate_sample"].to_numpy(dtype=float) for tn, g in s.groupby(s["trace_name"].astype(str), sort=False)}
    i_map = {str(tn): g["candidate_sample"].to_numpy(dtype=float) for tn, g in i.groupby(i["trace_name"].astype(str), sort=False)}
    for _, row in meta.iterrows():
        tn = str(row["trace_name"])
        true = float(row["s_arrival_sample"])
        sr = float(row["sampling_rate_hz"])
        ss = s_map.get(tn, np.array([], dtype=float))
        ii = i_map.get(tn, np.array([], dtype=float))
        stead_ok = bool(ss.size and np.min(np.abs((ss - true) / sr)) <= tol_s)
        ida_ok = bool(ii.size and np.min(np.abs((ii - true) / sr)) <= tol_s)
        if stead_ok and ida_ok:
            cls = "both_correct"
        elif stead_ok and not ida_ok:
            cls = "stead_only_correct"
        elif ida_ok and not stead_ok:
            cls = "ida_only_correct"
        else:
            cls = "neither_correct"
        rows.append(
            {
                "trace_name": tn,
                "event_id": str(row["event_id"]),
                "tol_s": tol_s,
                "k": k,
                "class": cls,
                "stead_correct": stead_ok,
                "ida_correct": ida_ok,
                "station": row.get("station"),
                "distance_km": row.get("distance_km"),
                "source_depth_km": row.get("source_depth_km"),
                "channel_prefix": row.get("channel_prefix"),
            }
        )
    return pd.DataFrame(rows)


def event_level_bootstrap_delta(
    event_ids: np.ndarray,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired event-level bootstrap of mean(trace metric) difference B-A.

    metric_* are per-trace scores (e.g. 1/0 hit); aggregated as mean within event then mean over events.
    """
    event_ids = np.asarray(event_ids).astype(str)
    metric_a = np.asarray(metric_a, dtype=float)
    metric_b = np.asarray(metric_b, dtype=float)
    uniq = np.unique(event_ids)
    # pre-aggregate per event
    a_ev = []
    b_ev = []
    n_tr = []
    for e in uniq:
        m = event_ids == e
        a_ev.append(np.nanmean(metric_a[m]))
        b_ev.append(np.nanmean(metric_b[m]))
        n_tr.append(int(m.sum()))
    a_ev = np.asarray(a_ev)
    b_ev = np.asarray(b_ev)
    delta0 = float(np.nanmean(b_ev - a_ev))
    rng = np.random.default_rng(seed)
    boots = np.empty(reps, dtype=float)
    n = len(uniq)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        boots[r] = float(np.nanmean(b_ev[idx] - a_ev[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_events": int(n),
        "n_traces": int(len(event_ids)),
        "mean_delta": delta0,
        "ci95": [float(lo), float(hi)],
        "direction": "b_better" if delta0 > 0 else ("a_better" if delta0 < 0 else "tie"),
        "reps": reps,
        "seed": seed,
        "mean_traces_per_event": float(np.mean(n_tr)),
    }


def per_trace_hit(pred: np.ndarray, true: np.ndarray, sr: np.ndarray, tol: float) -> np.ndarray:
    out = np.zeros(len(pred), dtype=float)
    for i in range(len(pred)):
        if not np.isfinite(true[i]):
            out[i] = np.nan
        elif np.isfinite(pred[i]) and abs(pred[i] - true[i]) / sr[i] <= tol:
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def top1_diagnosis(pred_s: np.ndarray, true_s: np.ndarray, sr: np.ndarray) -> dict[str, Any]:
    m = match_picks(pred_s, true_s, sr, windows_s=(0.1, 0.2, 0.5))
    bundle = report_pick_timing_bundle(pred_s, true_s, sr)
    ae = np.abs((pred_s - true_s) / sr)
    both = np.isfinite(pred_s) & np.isfinite(true_s)
    ae_b = ae[both]
    return {
        **{k: m[k] for k in m if k.startswith(("f1@", "precision@", "recall@", "miss"))},
        "detected_ae_median": bundle["detected_ae_median"],
        "detected_ae_mae": bundle["detected_ae_mae"],
        "detected_ae_p95": bundle["detected_ae_p95"],
        "wrong_peak_rate": bundle["wrong_peak_rate"],
        "no_pick_rate": bundle["no_pick_rate"],
        "n_ae_gt_10s": int((ae_b > 10).sum()) if ae_b.size else 0,
        "n_ae_gt_30s": int((ae_b > 30).sum()) if ae_b.size else 0,
        "p95_naming_warning": bundle["p95_naming_warning"],
    }
