"""Frozen Phase C candidate / feature schema (must match Phase B extraction)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from earthquake.stage6.phaseB import CAND_EXTRACT_CFG, UNION_DEDUP_S

CANDIDATE_SCHEMA_VERSION = "stage6_phaseC_union_stead5_ida5_v1"

CANDIDATE_SCHEMA: dict[str, Any] = {
    "version": CANDIDATE_SCHEMA_VERSION,
    "source": "UNION_STEAD5_IDA5",
    "stead_k": 5,
    "ida_k": 5,
    "max_union": 10,
    "dedup_s": UNION_DEDUP_S,
    "representative_time_rule": (
        "If both support within dedup_s, keep STEAD sample/time as representative; "
        "store ida_time and model_time_disagreement; never average probs or use human S."
    ),
    "peak_extract": dict(CAND_EXTRACT_CFG),
    "forbid_human_s_for_representative_time": True,
    "forbid_prob_weighted_average_time": True,
}

FORBIDDEN_FEATURE_SUBSTRINGS = (
    "true_s",
    "true_p",
    "s_arrival",
    "p_arrival",
    "human_",
    "manual_",
    "uncertainty",
    "label_snr",
    "event_id",
    "trace_name",
    "station_id",
    "source_id",
    "network",
    "station",
    "location",
    "channel_prefix",
    "confirm",
    "oracle_ae",
    "closest_ae",
    "label_class",
)

# Explicit allow-list used by feature builder (order = model input order for R2)
R1_FEATURE_NAMES = [
    "prob_stead",
    "prob_stead_miss",
    "prob_ida",
    "prob_ida_miss",
    "rank_stead_norm",
    "rank_ida_norm",
    "from_stead",
    "from_ida",
    "both_support",
    "union_rank_norm",
    "prob_margin_top12",
    "model_time_disagreement_s",
]

R2_EXTRA_FEATURE_NAMES = [
    "hist_pred_s_sample",
    "cand_minus_hist_s",
    "cand_minus_hist_abs",
    "cand_minus_hist_z",
    "cand_minus_hist_z_clipped",
    "log1p_history_count",
    "history_mad_s",
    "fallback_level_norm",
    "history_available",
    "base_tau_s",
    "shrinkage_weight",
    "pred_p_sample_stead",
    "pred_p_prob_stead",
    "pred_p_available",
    "cand_s_minus_pred_p",
    "hist_s_minus_pred_p",
    "cand_sp_minus_hist_sp",
    "source_depth_km",
    "distance_km",
    "hyp_distance_km",
    "azimuth_deg",
    "station_elevation_m",
]

R2_FEATURE_NAMES = R1_FEATURE_NAMES + R2_EXTRA_FEATURE_NAMES


def schema_sha256() -> str:
    blob = json.dumps(CANDIDATE_SCHEMA, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def forbidden_feature_names(names: list[str]) -> list[str]:
    bad = []
    for n in names:
        low = n.lower()
        if any(s in low for s in FORBIDDEN_FEATURE_SUBSTRINGS):
            # allow pred_p_* and history fields that mention samples but not human arrivals
            if low.startswith("pred_p_") or low.startswith("hist_") or low.startswith("cand_"):
                continue
            if low in {"source_depth_km", "distance_km", "hyp_distance_km", "azimuth_deg", "station_elevation_m"}:
                continue
            bad.append(n)
    return bad


def assert_no_forbidden_features(names: list[str]) -> None:
    bad = forbidden_feature_names(names)
    if bad:
        raise RuntimeError(f"Forbidden ranker features: {bad}")
