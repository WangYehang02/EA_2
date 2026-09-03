"""Feature matrix for Phase C ranker — leakage-safe allow-list only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.gating.cache_io import attach_expected_s
from earthquake.stage6.ranker.schema import R1_FEATURE_NAMES, R2_FEATURE_NAMES, assert_no_forbidden_features

MAD_FLOOR = 0.05
Z_CLIP = 10.0


def _nan0(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def build_candidate_feature_row(
    row: pd.Series,
    hist_row: pd.Series | None,
    *,
    global_res: dict,
    shrink_k: float = 50.0,
    variant: str = "R2",
) -> dict[str, float]:
    sr = float(row.get("sampling_rate_hz", 100.0))
    ps = float(row["stead_probability"]) if pd.notna(row.get("stead_probability")) else np.nan
    pi = float(row["ida_probability"]) if pd.notna(row.get("ida_probability")) else np.nan
    rs = int(row.get("stead_rank", -1))
    ri = int(row.get("ida_rank", -1))
    feat = {
        "prob_stead": _nan0(ps),
        "prob_stead_miss": 0.0 if np.isfinite(ps) else 1.0,
        "prob_ida": _nan0(pi),
        "prob_ida_miss": 0.0 if np.isfinite(pi) else 1.0,
        "rank_stead_norm": (rs / 4.0) if rs >= 0 else 0.0,
        "rank_ida_norm": (ri / 4.0) if ri >= 0 else 0.0,
        "from_stead": 1.0 if "stead" in str(row.get("candidate_source_mask", "")) else 0.0,
        "from_ida": 1.0 if "ida" in str(row.get("candidate_source_mask", "")) else 0.0,
        "both_support": 1.0 if bool(row.get("both_support", False)) else 0.0,
        "union_rank_norm": float(row.get("candidate_index", 0)) / 9.0,
        "prob_margin_top12": _nan0(float(row.get("prob_margin_top12", 0.0))),
        "model_time_disagreement_s": _nan0(float(row.get("model_time_disagreement", np.nan))),
    }
    if variant == "R1":
        assert_no_forbidden_features(list(feat.keys()))
        return feat

    # history
    if hist_row is None:
        hist_row = pd.Series(dtype=float)
    # attach expected S via residual prior helpers
    merged = pd.concat([row, hist_row], axis=0)
    # avoid duplicate index issues
    merged = merged[~merged.index.duplicated(keep="last")]
    try:
        exp = attach_expected_s(
            merged,
            global_res=global_res,
            shrink_k=shrink_k,
            min_history=5,
            mad_disable_s=1.0,
            min_sigma_s=0.05,
            max_sigma_s=1.0,
        )
    except Exception:
        exp = {
            "expected_s_sample": np.nan,
            "history_sigma_s": np.nan,
            "gate_history_available": 0.0,
            "shrinkage_weight_pred": 0.0,
        }

    hist_s = float(exp["expected_s_sample"])
    cand = float(row["candidate_sample"])
    mad = float(hist_row.get("residual_s_mad", np.nan)) if hist_row is not None else np.nan
    if not np.isfinite(mad):
        mad = float(hist_row.get("tau_s_mad", np.nan)) if hist_row is not None else np.nan
    mad_floor = max(MAD_FLOOR, mad if np.isfinite(mad) else MAD_FLOOR)
    resid = (cand - hist_s) / sr if np.isfinite(hist_s) else np.nan
    z = resid / mad_floor if np.isfinite(resid) else np.nan
    z_clip = float(np.clip(z, -Z_CLIP, Z_CLIP)) if np.isfinite(z) else 0.0
    hc = float(hist_row.get("history_count", 0) or 0)
    fb = float(hist_row.get("fallback_level", -1) if pd.notna(hist_row.get("fallback_level", np.nan)) else -1)

    pred_p = float(row.get("top1_p_stead", np.nan))
    pred_p_prob = float(row.get("top1_p_prob_stead", np.nan))
    pred_p_ok = np.isfinite(pred_p)
    cand_sp = (cand - pred_p) / sr if pred_p_ok else np.nan
    hist_sp = (hist_s - pred_p) / sr if (pred_p_ok and np.isfinite(hist_s)) else np.nan

    feat.update(
        {
            "hist_pred_s_sample": _nan0(hist_s) / 12000.0,  # coarse scale
            "cand_minus_hist_s": _nan0(resid),
            "cand_minus_hist_abs": abs(_nan0(resid)),
            "cand_minus_hist_z": _nan0(z),
            "cand_minus_hist_z_clipped": z_clip,
            "log1p_history_count": float(np.log1p(hc)),
            "history_mad_s": float(mad_floor),
            "fallback_level_norm": (fb / 4.0) if fb >= 0 else 1.0,
            "history_available": float(exp.get("gate_history_available", 0.0)),
            "base_tau_s": _nan0(float(hist_row.get("base_tau_s", np.nan))) / 100.0,
            "shrinkage_weight": _nan0(float(exp.get("shrinkage_weight_pred", 0.0))),
            "pred_p_sample_stead": _nan0(pred_p) / 12000.0,
            "pred_p_prob_stead": _nan0(pred_p_prob),
            "pred_p_available": 1.0 if pred_p_ok else 0.0,
            "cand_s_minus_pred_p": _nan0(cand_sp),
            "hist_s_minus_pred_p": _nan0(hist_sp),
            "cand_sp_minus_hist_sp": _nan0(cand_sp - hist_sp) if (np.isfinite(cand_sp) and np.isfinite(hist_sp)) else 0.0,
            "source_depth_km": _nan0(float(row.get("source_depth_km", np.nan))) / 50.0,
            "distance_km": _nan0(float(row.get("distance_km", np.nan))) / 100.0,
            "hyp_distance_km": _nan0(float(row.get("hyp_distance_km", np.nan))) / 100.0,
            "azimuth_deg": _nan0(float(row.get("azimuth_deg", np.nan))) / 360.0,
            "station_elevation_m": _nan0(float(row.get("station_elevation_m", np.nan))) / 1000.0,
        }
    )
    names = R1_FEATURE_NAMES if variant == "R1" else R2_FEATURE_NAMES
    out = {k: float(feat[k]) for k in names}
    assert_no_forbidden_features(list(out.keys()))
    return out


def assemble_feature_table(
    union: pd.DataFrame,
    hist: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    global_res: dict,
    variant: str = "R2",
) -> pd.DataFrame:
    hist_map = {str(r.trace_name): r for _, r in hist.iterrows()}
    meta_map = {str(r.trace_name): r for _, r in meta.iterrows()}
    rows = []
    for _, r in union.iterrows():
        tn = str(r["trace_name"])
        m = meta_map.get(tn)
        if m is not None:
            for col in ("source_depth_km", "distance_km", "hyp_distance_km", "azimuth_deg", "station_elevation_m"):
                if col in m.index:
                    r = r.copy()
                    r[col] = m[col]
        feat = build_candidate_feature_row(r, hist_map.get(tn), global_res=global_res, variant=variant)
        feat["trace_name"] = tn
        feat["event_id"] = str(r["event_id"])
        feat["candidate_index"] = int(r["candidate_index"])
        feat["candidate_sample"] = float(r["candidate_sample"])
        feat["is_positive"] = bool(r.get("is_positive", False))
        feat["label_none_of_k"] = bool(r.get("label_none_of_k", False))
        feat["positive_index"] = int(r.get("positive_index", -1))
        feat["candidate_count"] = int(r.get("candidate_count", 0))
        feat["analysis_classes"] = str(r.get("analysis_classes", ""))
        feat["sampling_rate_hz"] = float(r.get("sampling_rate_hz", 100.0))
        feat["true_s_sample"] = float(r.get("true_s_sample", np.nan))
        rows.append(feat)
    return pd.DataFrame(rows)


def feature_names(variant: str) -> list[str]:
    return list(R1_FEATURE_NAMES if variant == "R1" else R2_FEATURE_NAMES)
