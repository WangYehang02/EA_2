"""Labels and analysis classes for Phase C ranker (labels never enter features)."""

from __future__ import annotations

import numpy as np
import pandas as pd


POS_TOL_S = 0.5


def label_union_candidates(union: pd.DataFrame, *, tol_s: float = POS_TOL_S) -> pd.DataFrame:
    """Mark positive candidate (closest within tol) or none_of_k per trace."""
    rows = []
    for tn, g in union.groupby(union["trace_name"].astype(str), sort=False):
        g = g.sort_values("candidate_index").reset_index(drop=True)
        true = float(g["true_s_sample"].iloc[0])
        sr = float(g["sampling_rate_hz"].iloc[0])
        if not np.isfinite(true):
            # unlabeled — skip for training
            for _, r in g.iterrows():
                d = dict(r)
                d["is_positive"] = False
                d["label_none_of_k"] = True
                d["closest_ae_s"] = np.nan
                d["positive_index"] = -1
                rows.append(d)
            continue
        ae = np.abs((g["candidate_sample"].to_numpy(dtype=float) - true) / sr)
        # tie-break: min AE then smaller candidate_index
        order = np.lexsort((g["candidate_index"].to_numpy(), ae))
        best_i = int(order[0])
        best_ae = float(ae[best_i])
        pos_idx = int(g.iloc[best_i]["candidate_index"]) if best_ae <= tol_s else -1
        none = pos_idx < 0
        for j, (_, r) in enumerate(g.iterrows()):
            d = dict(r)
            d["is_positive"] = (int(r["candidate_index"]) == pos_idx) and (not none)
            d["label_none_of_k"] = bool(none)
            d["closest_ae_s"] = best_ae
            d["positive_index"] = pos_idx
            rows.append(d)
    return pd.DataFrame(rows)


def attach_analysis_classes(labeled: pd.DataFrame, stead_top1: pd.Series | None = None) -> pd.DataFrame:
    """Per-trace analysis classes (not model inputs)."""
    out_rows = []
    for tn, g in labeled.groupby("trace_name", sort=False):
        true = float(g["true_s_sample"].iloc[0])
        sr = float(g["sampling_rate_hz"].iloc[0])
        none = bool(g["label_none_of_k"].iloc[0])
        n_cand = int(g["candidate_count"].iloc[0])
        both = bool((g["both_support"]).any())
        # recoverable by source
        ae = np.abs((g["candidate_sample"].to_numpy(float) - true) / sr) if np.isfinite(true) else None
        stead_ok = False
        ida_ok = False
        if ae is not None:
            for i, (_, r) in enumerate(g.iterrows()):
                if ae[i] <= 0.5:
                    if r["candidate_source_mask"] in ("stead", "stead+ida"):
                        stead_ok = True
                    if r["candidate_source_mask"] in ("ida", "stead+ida"):
                        ida_ok = True
        top1 = np.nan
        if stead_top1 is not None and tn in stead_top1.index:
            top1 = float(stead_top1.loc[tn])
        elif "top1_s_stead" in g.columns:
            top1 = float(g["top1_s_stead"].iloc[0])
        top1_ok = bool(np.isfinite(top1) and np.isfinite(true) and abs(top1 - true) / sr <= 0.5)
        classes = []
        if none:
            classes.append("none_of_k")
        else:
            if top1_ok:
                classes.append("easy_top1_correct")
            else:
                classes.append("recoverable_stead_wrong")
            if ida_ok and not stead_ok:
                classes.append("ida_only_recoverable")
            elif stead_ok and not ida_ok:
                classes.append("stead_only_recoverable")
            elif stead_ok and ida_ok:
                classes.append("both_models_recoverable")
        if n_cand >= 2:
            classes.append("multi_peak")
        for _, r in g.iterrows():
            d = dict(r)
            d["analysis_classes"] = "|".join(classes)
            out_rows.append(d)
    return pd.DataFrame(out_rows)
