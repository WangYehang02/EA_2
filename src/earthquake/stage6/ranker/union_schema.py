"""Phase-C union with STEAD-preferred representative time + dual timestamps."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from earthquake.stage6.phaseB import UNION_DEDUP_S, topk_from_cache


def irreversible_hash(text: str, salt: str = "stage6_phaseC_v1") -> str:
    return hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()[:16]


def union_candidates_phaseC(
    stead: pd.DataFrame,
    ida: pd.DataFrame,
    *,
    stead_k: int = 5,
    ida_k: int = 5,
    max_union: int = 10,
    dedup_s: float = UNION_DEDUP_S,
    sampling_rate_hz: float = 100.0,
) -> pd.DataFrame:
    """Union STEAD∪ID-A with Phase-C representative-time rules.

    When both support a peak: keep STEAD sample/time; store ida_time and disagreement.
    Never averages probabilities; never uses human S.
    """
    s = topk_from_cache(stead, stead_k)
    i = topk_from_cache(ida, ida_k)
    rows_out: list[dict[str, Any]] = []
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
        top1_s_stead = float(base["top1_s_sample"].iloc[0]) if "top1_s_sample" in base.columns and sg is not None else np.nan
        top1_p_stead = float(base["top1_p_sample"].iloc[0]) if "top1_p_sample" in base.columns and sg is not None else np.nan
        top1_p_prob_stead = float(base["top1_p_prob"].iloc[0]) if "top1_p_prob" in base.columns and sg is not None else np.nan
        if ig is not None and "top1_p_sample" in ig.columns:
            top1_p_ida = float(ig["top1_p_sample"].iloc[0])
            top1_p_prob_ida = float(ig["top1_p_prob"].iloc[0]) if "top1_p_prob" in ig.columns else np.nan
        else:
            top1_p_ida = np.nan
            top1_p_prob_ida = np.nan

        pool: list[dict[str, Any]] = []
        if sg is not None:
            for _, r in sg.iterrows():
                pool.append(
                    {
                        "sample": float(r["candidate_sample"]),
                        "stead_time_utc": r.get("candidate_time_utc"),
                        "ida_time_utc": None,
                        "prob_stead": float(r["candidate_probability"]),
                        "prob_ida": np.nan,
                        "rank_stead": int(r["candidate_rank"]),
                        "rank_ida": -1,
                        "from_stead": True,
                        "from_ida": False,
                        "stead_sample": float(r["candidate_sample"]),
                        "ida_sample": np.nan,
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
                            "stead_time_utc": None,
                            "ida_time_utc": r.get("candidate_time_utc"),
                            "prob_stead": np.nan,
                            "prob_ida": float(r["candidate_probability"]),
                            "rank_stead": -1,
                            "rank_ida": int(r["candidate_rank"]),
                            "from_stead": False,
                            "from_ida": True,
                            "stead_sample": np.nan,
                            "ida_sample": samp,
                        }
                    )
                else:
                    # both support: keep STEAD representative sample/time
                    matched["from_ida"] = True
                    matched["prob_ida"] = float(r["candidate_probability"])
                    matched["rank_ida"] = int(r["candidate_rank"])
                    matched["ida_sample"] = samp
                    matched["ida_time_utc"] = r.get("candidate_time_utc")

        def sort_key(p: dict) -> tuple:
            joint = 1 if (p["from_stead"] and p["from_ida"]) else 0
            probs = [x for x in (p["prob_stead"], p["prob_ida"]) if np.isfinite(x)]
            mx = max(probs) if probs else -1.0
            rs = p["rank_stead"] if p["rank_stead"] >= 0 else 99
            return (-joint, -mx, rs)

        pool_sorted = sorted(pool, key=sort_key)[:max_union]
        # top1/top2 margins within available probs (max of available sources)
        probs_all = []
        for p in pool_sorted:
            vals = [x for x in (p["prob_stead"], p["prob_ida"]) if np.isfinite(x)]
            probs_all.append(max(vals) if vals else 0.0)
        margin = 0.0
        if len(probs_all) >= 2:
            sp = sorted(probs_all, reverse=True)
            margin = float(sp[0] - sp[1])

        n_union = len(pool_sorted)
        for rank, p in enumerate(pool_sorted):
            dis = np.nan
            if np.isfinite(p["stead_sample"]) and np.isfinite(p["ida_sample"]):
                dis = abs(float(p["stead_sample"]) - float(p["ida_sample"])) / sr
            rows_out.append(
                {
                    "trace_name": tn,
                    "event_id": eid,
                    "event_id_hash": irreversible_hash(eid),
                    "trace_key_hash": irreversible_hash(tn),
                    "true_s_sample": true_s,
                    "sampling_rate_hz": sr,
                    "candidate_index": rank,
                    "candidate_sample": p["sample"],
                    "candidate_time_utc": p["stead_time_utc"] if p["from_stead"] else p["ida_time_utc"],
                    "stead_time": p["stead_time_utc"],
                    "ida_time": p["ida_time_utc"],
                    "stead_sample": p["stead_sample"],
                    "ida_sample": p["ida_sample"],
                    "stead_probability": p["prob_stead"],
                    "ida_probability": p["prob_ida"],
                    "stead_rank": p["rank_stead"],
                    "ida_rank": p["rank_ida"],
                    "both_support": bool(p["from_stead"] and p["from_ida"]),
                    "candidate_source_mask": (
                        "stead+ida"
                        if p["from_stead"] and p["from_ida"]
                        else ("stead" if p["from_stead"] else "ida")
                    ),
                    "model_time_disagreement": dis,
                    "candidate_count": n_union,
                    "prob_margin_top12": margin,
                    "top1_s_stead": top1_s_stead,
                    "top1_p_stead": top1_p_stead,
                    "top1_p_prob_stead": top1_p_prob_stead,
                    "top1_p_ida": top1_p_ida,
                    "top1_p_prob_ida": top1_p_prob_ida,
                    "candidate_source": "union",
                }
            )
    return pd.DataFrame(rows_out)
