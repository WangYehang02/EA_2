"""v4 Stage-6 full-dev stop gate: frozen protocol, A/B/C, grouping, verdict.

Postmortem evaluation of frozen v4 checkpoints. Not PILOT.PASSED.
Never trains. Never reads confirm waveforms or confirm metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earthquake.stage10.crop_v2 import LONG_PS_S
from earthquake.stage6.phaseB import BOOTSTRAP_REPS, UNION_DEDUP_S, oracle_metrics_for_set, oracle_pick_closest
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics, correct_at

ROOT = Path(__file__).resolve().parents[3]
V4 = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v4_seed42_fp32")
MARK = "FULLDEV.POSTMORTEM_STOP_GATE"
N_BOOT = 5000
BOOT_SEED = 42
DELTA_F1 = 0.01
P95_IMPROVE_S = 0.15
NOISE_N = 4000
HEIGHT = 0.2

# Pre-registered grouping. Locked before DKPN full-dev numbers exist.
# P–S: crop_v2.LONG_PS_S = 30 s (long P–S kept in v2/v4 training).
# Distance: artifacts/results/stage6/missing_s_label_audit.json bins.
# SNR: round dB bins locked here; not chosen from DKPN full-dev.
# Station/channel/network: all levels present in the frozen Phase-B manifest.
PS_INTERVAL_BINS = (
    ("ps_lt_10s", 0.0, 10.0),
    ("ps_10_to_30s", 10.0, float(LONG_PS_S)),
    ("ps_ge_30s", float(LONG_PS_S), float("inf")),
)
DISTANCE_BINS_KM = (
    ("dist_0_50", 0.0, 50.0),
    ("dist_50_100", 50.0, 100.0),
    ("dist_100_200", 100.0, 200.0),
    ("dist_200_400", 200.0, 400.0),
    ("dist_ge_400", 400.0, float("inf")),
)
SNR_BINS_DB = (
    ("snr_lt_0", float("-inf"), 0.0),
    ("snr_0_10", 0.0, 10.0),
    ("snr_10_20", 10.0, 20.0),
    ("snr_ge_20", 20.0, float("inf")),
)

WAVEFORM_ONLY_DEV_F1_LOCKED = {
    "STEAD_top1": 0.8402621057816778,
    "IDA_top1": 0.8389676148144753,
    "PhaseNet-ETHZ": 0.8355245795446365,
    "PhaseNet-SCEDC": 0.7583606941838649,
    "SegPhase-100Hz": 0.6786043735417007,
}
STRONGEST_WAVEFORM_ONLY = "STEAD_top1"

VERDICTS = (
    "keep_as_primary_picker_candidate",
    "keep_as_union_candidate_only",
    "rejected_candidate_source",
)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def scalar_metrics(m: dict) -> dict:
    skip = {"correct@0.5", "correct@0.1", "event_id"}
    out = {}
    for k, v in m.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            continue
        out[k] = jsonable(v)
    return out


def metrics_pack(pred: np.ndarray, true: np.ndarray, sr: np.ndarray, n_peaks: np.ndarray | None = None) -> dict:
    m = comprehensive_pick_metrics(pred, true, sr)
    out = scalar_metrics(m)
    out["picks_per_trace"] = float(np.mean(n_peaks)) if n_peaks is not None else float(out.get("prediction_coverage", np.nan))
    out["frac_no_s_peak"] = float(np.mean(~np.isfinite(pred)))
    if n_peaks is not None:
        out["frac_no_s_peak"] = float(np.mean(np.asarray(n_peaks) == 0))
        out["picks_per_trace"] = float(np.mean(n_peaks))
    return out


def event_bootstrap_f1(
    event_ids: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Event-unit bootstrap of F1@0.5(b) − F1@0.5(a)."""
    rng = np.random.default_rng(int(seed))
    eids = np.asarray(event_ids).astype(str)
    events = np.unique(eids)
    idx: dict[str, np.ndarray] = {}
    for i, e in enumerate(eids):
        idx.setdefault(e, []).append(i)
    idx = {e: np.asarray(v, dtype=np.int64) for e, v in idx.items()}
    deltas = np.empty(int(n_boot), dtype=np.float64)
    ev = np.asarray(events)
    for i in range(int(n_boot)):
        draw = rng.choice(ev, size=len(ev), replace=True)
        ia = np.concatenate([idx[e] for e in draw])
        fa = comprehensive_pick_metrics(pred_a[ia], true[ia], sr[ia])["f1@0.5"]
        fb = comprehensive_pick_metrics(pred_b[ia], true[ia], sr[ia])["f1@0.5"]
        deltas[i] = float(fb - fa)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "unit": "event",
        "mean_delta": float(deltas.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "ci95_lo_gt_0": bool(lo > 0),
    }


def stop_gates(
    *,
    dkpn: dict,
    baseline: dict,
    union_oracle: dict,
    union_plus_dkpn_oracle: dict,
    boot_dkpn_minus_baseline: dict,
    boot_union_plus_minus_union: dict,
) -> dict:
    f1_d = float(dkpn["f1@0.5"])
    f1_s = float(baseline["f1@0.5"])
    miss_d = float(dkpn["miss_rate"])
    miss_s = float(baseline["miss_rate"])
    p95_d = float(dkpn["detected_ae_p95"])
    p95_s = float(baseline["detected_ae_p95"])
    f1_u = float(union_oracle["f1@0.5"] if "f1@0.5" in union_oracle else union_oracle.get("oracle_f1@0.5"))
    f1_ud = float(
        union_plus_dkpn_oracle["f1@0.5"]
        if "f1@0.5" in union_plus_dkpn_oracle
        else union_plus_dkpn_oracle.get("oracle_f1@0.5")
    )
    a = bool(f1_d - f1_s >= DELTA_F1)
    b = bool(f1_d >= f1_s - 1e-12 and (p95_s - p95_d) >= P95_IMPROVE_S and miss_d <= miss_s + 1e-12)
    c_delta = bool(f1_ud - f1_u >= DELTA_F1)
    c_ci = bool(float(boot_union_plus_minus_union.get("ci95_lo", -1)) > 0)
    c = bool(c_delta and c_ci)
    gates = {
        "A_vs_waveform_only_delta_f1_0.5_ge_0.01": a,
        "B_f1_not_down_and_p95_ge_0.15s_and_miss_not_worse": b,
        "C_union_oracle_delta_f1_0.5_ge_0.01_and_ci_lo_gt_0": c,
        "C_delta": c_delta,
        "C_ci_lo_gt_0": c_ci,
    }
    passed = bool(a or b or c)
    if a or b:
        verdict = "keep_as_primary_picker_candidate"
    elif c:
        verdict = "keep_as_union_candidate_only"
    else:
        verdict = "rejected_candidate_source"
    return {
        "gates": gates,
        "passed": passed,
        "verdict": verdict,
        "delta_f1_vs_waveform_only": float(f1_d - f1_s),
        "delta_p95_vs_waveform_only": float(p95_s - p95_d),
        "delta_miss_vs_waveform_only": float(miss_d - miss_s),
        "delta_union_oracle_f1": float(f1_ud - f1_u),
        "boot_dkpn_minus_baseline": boot_dkpn_minus_baseline,
        "boot_union_plus_minus_union": boot_union_plus_minus_union,
        "strongest_waveform_only": STRONGEST_WAVEFORM_ONLY,
        "margin_f1": DELTA_F1,
        "p95_improve_s": P95_IMPROVE_S,
        "did_not_lower_threshold_because_pilot_looked_good": True,
    }


def assign_bin(value: float, bins: tuple[tuple[str, float, float], ...]) -> str:
    v = float(value)
    if not np.isfinite(v):
        return "unknown"
    for name, lo, hi in bins:
        if v >= lo and v < hi:
            return name
    last = bins[-1]
    if np.isfinite(value) and value >= last[1]:
        return last[0]
    return "unknown"


def grouping_table(meta: pd.DataFrame, pred: np.ndarray, n_peaks: np.ndarray | None = None) -> dict:
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    ps = (meta["s_arrival_sample"] - meta["p_arrival_sample"]) / meta["sampling_rate_hz"]
    g = meta.copy()
    g["_pred"] = pred
    g["_ps"] = ps.to_numpy(float)
    g["ps_bin"] = [assign_bin(v, PS_INTERVAL_BINS) for v in g["_ps"]]
    g["dist_bin"] = [assign_bin(v, DISTANCE_BINS_KM) for v in g["distance_km"].to_numpy(float)]
    g["snr_bin"] = [assign_bin(v, SNR_BINS_DB) for v in g["snr_db"].to_numpy(float)]
    g["station"] = g["station"].astype(str)
    g["channel_prefix"] = g["channel_prefix"].astype(str)
    g["network"] = g["network"].astype(str)
    if n_peaks is not None:
        g["_n_peaks"] = n_peaks

    def _one(mask: np.ndarray) -> dict:
        if int(mask.sum()) == 0:
            return {"n": 0}
        pk = g["_n_peaks"].to_numpy()[mask] if "_n_peaks" in g.columns else None
        m = metrics_pack(g["_pred"].to_numpy(float)[mask], true[mask], sr[mask], pk)
        m["n"] = int(mask.sum())
        m["n_events"] = int(g.loc[mask, "event_id"].nunique())
        return m

    out: dict[str, Any] = {"locked_bins": {
        "ps_interval_s": [list(x) for x in PS_INTERVAL_BINS],
        "distance_km": [list(x) for x in DISTANCE_BINS_KM],
        "snr_db": [list(x) for x in SNR_BINS_DB],
        "station": "all_stations_in_frozen_manifest_full_csv_no_posthoc_subset",
        "channel_prefix": "all_values_in_frozen_manifest",
        "network": "all_values_in_frozen_manifest",
        "no_result_selected_groups": True,
    }}
    for col, key in (("ps_bin", "ps_interval"), ("dist_bin", "distance"), ("snr_bin", "snr"), ("channel_prefix", "channel_prefix"), ("network", "network")):
        rows = {}
        for name, sub in g.groupby(col, sort=True):
            rows[str(name)] = _one(g.index.isin(sub.index))
        out[key] = rows
    st_rows = []
    for name, sub in g.groupby("station", sort=True):
        m = _one(g.index.isin(sub.index))
        m["station"] = str(name)
        st_rows.append(m)
    out["station_n"] = int(g["station"].nunique())
    out["station_preview_n"] = min(5, len(st_rows))
    return out, pd.DataFrame(st_rows)


def per_event_table(meta: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    ok05 = correct_at(pred, true, sr, 0.5)
    ok01 = correct_at(pred, true, sr, 0.1)
    df = pd.DataFrame(
        {
            "event_id": meta["event_id"].astype(str).to_numpy(),
            "trace_name": meta["trace_name"].astype(str).to_numpy(),
            "pred_s_sample": pred,
            "true_s_sample": true,
            "correct@0.5": ok05,
            "correct@0.1": ok01,
        }
    )
    rows = []
    for eid, g in df.groupby("event_id", sort=True):
        rows.append(
            {
                "event_id": eid,
                "n_traces": int(len(g)),
                "recall@0.5": float(g["correct@0.5"].mean()),
                "recall@0.1": float(g["correct@0.1"].mean()),
                "frac_no_pick": float((~np.isfinite(g["pred_s_sample"].to_numpy(float))).mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_union_dkpn5(
    union: pd.DataFrame,
    dkpn_k5: pd.DataFrame,
    *,
    dedup_s: float = UNION_DEDUP_S,
) -> pd.DataFrame:
    """Keep UNION_STEAD5_IDA5 candidates; add unmatched DKPN top-5 peaks (0.05 s dedup)."""
    rows = []
    u_map = {str(tn): g for tn, g in union.groupby(union["trace_name"].astype(str), sort=False)}
    d_map = {str(tn): g for tn, g in dkpn_k5.groupby(dkpn_k5["trace_name"].astype(str), sort=False)}
    names = sorted(set(u_map) | set(d_map))
    for tn in names:
        ug = u_map.get(tn)
        dg = d_map.get(tn)
        base = ug if ug is not None else dg
        sr = float(base["sampling_rate_hz"].iloc[0]) if "sampling_rate_hz" in base.columns else 100.0
        eid = str(base["event_id"].iloc[0]) if "event_id" in base.columns else ""
        existing = []
        if ug is not None:
            for _, r in ug.iterrows():
                existing.append(float(r["candidate_sample"]))
                rows.append(
                    {
                        "trace_name": tn,
                        "event_id": eid,
                        "sampling_rate_hz": sr,
                        "candidate_sample": float(r["candidate_sample"]),
                        "source": "union",
                    }
                )
        if dg is not None:
            for _, r in dg.iterrows():
                samp = float(r["candidate_sample"])
                if existing and min(abs(samp - e) / sr for e in existing) <= dedup_s:
                    continue
                existing.append(samp)
                rows.append(
                    {
                        "trace_name": tn,
                        "event_id": eid,
                        "sampling_rate_hz": sr,
                        "candidate_sample": samp,
                        "source": "dkpn5",
                    }
                )
    return pd.DataFrame(rows)


def dkpn_k5_frame(names: np.ndarray, events: np.ndarray, sr: np.ndarray, peaks_list: list) -> pd.DataFrame:
    rows = []
    for tn, eid, srate, peaks in zip(names, events, sr, peaks_list):
        for i, samp in enumerate(peaks):
            rows.append(
                {
                    "trace_name": str(tn),
                    "event_id": str(eid),
                    "sampling_rate_hz": float(srate),
                    "candidate_sample": float(samp),
                    "candidate_rank": int(i),
                }
            )
    return pd.DataFrame(rows)


def oracle_pred_from_candidates(cand: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    o = oracle_metrics_for_set(cand, meta, sample_col="candidate_sample")
    return o["oracle_predictions"], o
