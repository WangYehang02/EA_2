"""Event-wise travel-time residual / moveout soft rescoring (experimental).

Catalog-assisted. Final picks always come from the existing UNION candidate set.
Does not modify locked Stage-6 confirm artifacts.
Travel-time baseline must be train-only (reuse Stage-6 picker_train MLP / history base_tau).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s

Mode = Literal["absolute_s", "sp"]
Agg = Literal["median", "weighted_median"]


@dataclass(frozen=True)
class MoveoutRescoreConfig:
    mode: Mode = "absolute_s"
    aggregator: Agg = "median"
    sigma_r_s: float = 0.5
    lambda_moveout: float = 0.25
    min_neighbors: int = 1
    use_mad_gate: bool = False
    sigma_mad_s: float = 1.0
    eps: float = 1e-8


@dataclass
class MoveoutEventPack:
    trace_names: list[str]
    distance_km: np.ndarray
    sr: np.ndarray
    cand_sample: list[np.ndarray]
    cand_score: list[np.ndarray]
    cand_prob: list[np.ndarray]
    # per-candidate residuals (seconds): tau_S - T_S_hat  OR  SP - SP_hat
    cand_resid: list[np.ndarray]
    baseline_idx: np.ndarray
    baseline_sample: np.ndarray
    # theoretical predictions (station-level)
    t_hat: np.ndarray  # T_S_hat or DeltaSP_hat


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * cw[-1]
    return float(v[int(np.searchsorted(cw, cutoff))])


def build_moveout_packs(
    table: pd.DataFrame,
    *,
    mode: Mode = "absolute_s",
) -> dict[str, MoveoutEventPack]:
    """Build event packs. Requires columns:
    candidate_sample, fixed_score, cand_prob, pred_p_sample, sampling_rate_hz,
    origin_time, trace_start_time, baseline_selected, base_tau_s, base_delta_sp,
    distance_km, event_id, trace_name
    """
    df = table.copy()
    df["tau_s"] = absolute_travel_time_s(
        origin_time=df["origin_time"],
        trace_start_time=df["trace_start_time"],
        sample=df["candidate_sample"].to_numpy(float),
        sampling_rate_hz=df["sampling_rate_hz"].to_numpy(float),
    )
    df["_sp"] = (df["candidate_sample"].to_numpy(float) - df["pred_p_sample"].to_numpy(float)) / df[
        "sampling_rate_hz"
    ].to_numpy(float)
    if mode == "absolute_s":
        df["_resid"] = df["tau_s"] - df["base_tau_s"].to_numpy(float)
        t_col = "base_tau_s"
    else:
        df["_resid"] = df["_sp"] - df["base_delta_sp"].to_numpy(float)
        t_col = "base_delta_sp"

    packs: dict[str, MoveoutEventPack] = {}
    for eid, g in df.groupby("event_id", sort=False):
        tnames = g.groupby("trace_name", sort=False).size().index.astype(str).tolist()
        distance, sr, that = [], [], []
        cs_l, sc_l, cp_l, resid_l = [], [], [], []
        bidx, bsam = [], []
        for tn in tnames:
            gg = g[g.trace_name.astype(str) == tn].sort_values("candidate_index")
            distance.append(float(gg.distance_km.iloc[0]))
            sr.append(float(gg.sampling_rate_hz.iloc[0]))
            that.append(float(gg[t_col].iloc[0]))
            cs_l.append(gg.candidate_sample.to_numpy(float))
            sc_l.append(gg.fixed_score.to_numpy(float))
            cp_l.append(gg.cand_prob.to_numpy(float))
            resid_l.append(gg._resid.to_numpy(float))
            bs = gg.baseline_selected.to_numpy(bool)
            bi = int(np.argmax(bs)) if bs.any() else int(np.nanargmax(gg.fixed_score.to_numpy(float)))
            bidx.append(bi)
            bsam.append(float(cs_l[-1][bi]))
        packs[str(eid)] = MoveoutEventPack(
            trace_names=tnames,
            distance_km=np.asarray(distance, float),
            sr=np.asarray(sr, float),
            cand_sample=cs_l,
            cand_score=sc_l,
            cand_prob=cp_l,
            cand_resid=resid_l,
            baseline_idx=np.asarray(bidx, int),
            baseline_sample=np.asarray(bsam, float),
            t_hat=np.asarray(that, float),
        )
    return packs


def _event_center(
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    aggregator: Agg,
) -> tuple[float, float]:
    """Return (center, MAD)."""
    ok = np.isfinite(residuals)
    if not ok.any():
        return float("nan"), float("nan")
    r = residuals[ok]
    w = weights[ok]
    if aggregator == "weighted_median":
        med = _weighted_median(r, w)
    else:
        med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) if r.size else float("nan")
    return med, mad


def _rescore_pack(
    pack: MoveoutEventPack,
    cfg: MoveoutRescoreConfig,
    *,
    neighbor_resid_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = len(pack.trace_names)
    new_idx = pack.baseline_idx.copy()
    new_sample = pack.baseline_sample.copy()
    n_neighbors = np.zeros(n, dtype=np.int32)
    changed = np.zeros(n, dtype=bool)
    center = np.full(n, np.nan)
    mad_arr = np.full(n, np.nan)
    r_sel = np.full(n, np.nan)
    r_orig = np.full(n, np.nan)

    # neighbor reference residuals from selected prediction (or override)
    sel_resid = np.full(n, np.nan)
    sel_w = np.ones(n)
    for i in range(n):
        bi = int(pack.baseline_idx[i])
        if len(pack.cand_resid[i]) == 0:
            continue
        if neighbor_resid_override is not None and np.isfinite(neighbor_resid_override[i]):
            sel_resid[i] = float(neighbor_resid_override[i])
        else:
            sel_resid[i] = float(pack.cand_resid[i][bi])
        q = float(pack.cand_prob[i][bi]) if len(pack.cand_prob[i]) else 1.0
        sel_w[i] = q if np.isfinite(q) and q > 0 else 1.0

    sigma = max(cfg.sigma_r_s, 1e-6)
    for i in range(n):
        cs = pack.cand_sample[i]
        k = len(cs)
        if k == 0:
            continue
        # leave-one-out neighbors
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        usable = np.where(mask & np.isfinite(sel_resid))[0]
        n_neighbors[i] = int(len(usable))
        if len(usable) < int(cfg.min_neighbors):
            continue
        med, mad = _event_center(sel_resid[usable], sel_w[usable], aggregator=cfg.aggregator)
        if not np.isfinite(med):
            continue
        center[i] = med
        mad_arr[i] = mad
        mad_factor = 1.0
        if cfg.use_mad_gate and np.isfinite(mad):
            mad_factor = float(np.exp(-0.5 * (mad / max(cfg.sigma_mad_s, 1e-6)) ** 2))

        resid_i = pack.cand_resid[i]
        base = pack.cand_score[i]
        r_score = np.zeros(k)
        for c in range(k):
            if not np.isfinite(resid_i[c]):
                r_score[c] = 0.0
                continue
            delta = abs(float(resid_i[c]) - med)
            r_score[c] = float(np.exp(-0.5 * (delta / sigma) ** 2) * mad_factor)
        final = base + cfg.lambda_moveout * r_score
        jbest = int(np.nanargmax(final))
        new_idx[i] = jbest
        new_sample[i] = float(cs[jbest])
        r_sel[i] = float(r_score[jbest])
        r_orig[i] = float(r_score[int(pack.baseline_idx[i])])
        changed[i] = jbest != int(pack.baseline_idx[i])

    return {
        "selected_index": new_idx,
        "selected_sample": new_sample,
        "n_neighbors": n_neighbors,
        "event_center": center,
        "event_mad": mad_arr,
        "moveout_score_selected": r_sel,
        "moveout_score_original_top1": r_orig,
        "changed_candidate": changed,
    }


def apply_moveout_rescore_packs(
    packs: dict[str, MoveoutEventPack],
    cfg: MoveoutRescoreConfig,
    *,
    neighbor_resid_by_trace: dict[str, float] | None = None,
    shuffle_event_geometry: bool = False,
    shuffle_neighbor_resid: bool = False,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply moveout soft rescoring.

    neighbor_resid_by_trace: LEAKAGE/oracle only — override neighbor residuals
    (e.g. catalog S). Target candidates are never rewritten.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    items = list(packs.items())

    for eid, pack in items:
        p = pack
        if shuffle_event_geometry:
            # destroy distance ordering / residual geometry by permuting residuals among stations
            p = MoveoutEventPack(
                trace_names=pack.trace_names,
                distance_km=pack.distance_km.copy(),
                sr=pack.sr,
                cand_sample=pack.cand_sample,
                cand_score=pack.cand_score,
                cand_prob=pack.cand_prob,
                cand_resid=[r.copy() for r in pack.cand_resid],
                baseline_idx=pack.baseline_idx,
                baseline_sample=pack.baseline_sample,
                t_hat=pack.t_hat.copy(),
            )
            # shuffle station residual vectors' selected refs via permuting t_hat and baseline resid mapping
            order = rng.permutation(len(p.trace_names))
            p.t_hat = p.t_hat[order]
            # Also permute cand_resid lists to break event coherence
            p.cand_resid = [p.cand_resid[j] for j in order]
            # keep samples/scores aligned with original stations — only residual refs shuffled:
            # simpler diagnostic: shuffle selected residuals only via override below

        override = None
        if neighbor_resid_by_trace is not None:
            override = np.array(
                [float(neighbor_resid_by_trace.get(tn, np.nan)) for tn in pack.trace_names],
                dtype=np.float64,
            )
        elif shuffle_neighbor_resid or shuffle_event_geometry:
            override = np.array(
                [
                    float(pack.cand_resid[i][pack.baseline_idx[i]]) if len(pack.cand_resid[i]) else np.nan
                    for i in range(len(pack.trace_names))
                ],
                dtype=np.float64,
            )
            rng.shuffle(override)

        res = _rescore_pack(pack if not shuffle_event_geometry else pack, cfg, neighbor_resid_override=override)
        # Note: for shuffle_event_geometry we still score original pack with shuffled override
        for i, tn in enumerate(pack.trace_names):
            rows.append(
                {
                    "trace_name": tn,
                    "event_id": eid,
                    "baseline_sample": float(pack.baseline_sample[i]),
                    "new_sample": float(res["selected_sample"][i]),
                    "baseline_cand_index": int(pack.baseline_idx[i]),
                    "new_cand_index": int(res["selected_index"][i]),
                    "n_neighbors": int(res["n_neighbors"][i]),
                    "event_center": float(res["event_center"][i])
                    if np.isfinite(res["event_center"][i])
                    else np.nan,
                    "event_mad": float(res["event_mad"][i]) if np.isfinite(res["event_mad"][i]) else np.nan,
                    "moveout_score_selected": float(res["moveout_score_selected"][i])
                    if np.isfinite(res["moveout_score_selected"][i])
                    else np.nan,
                    "moveout_score_original_top1": float(res["moveout_score_original_top1"][i])
                    if np.isfinite(res["moveout_score_original_top1"][i])
                    else np.nan,
                    "changed_candidate": bool(res["changed_candidate"][i]),
                    "distance_km": float(pack.distance_km[i]),
                }
            )
    return pd.DataFrame(rows)


def per_candidate_moveout_delta(
    pack: MoveoutEventPack,
    station_i: int,
    *,
    aggregator: Agg = "median",
    neighbor_resid_override: np.ndarray | None = None,
    min_neighbors: int = 1,
) -> np.ndarray:
    """Leave-one-out |r_i(c) - m_E^{-i}| for each candidate of station i."""
    n = len(pack.trace_names)
    sel = np.full(n, np.nan)
    w = np.ones(n)
    for j in range(n):
        bi = int(pack.baseline_idx[j])
        if len(pack.cand_resid[j]) == 0:
            continue
        if neighbor_resid_override is not None and np.isfinite(neighbor_resid_override[j]):
            sel[j] = float(neighbor_resid_override[j])
        else:
            sel[j] = float(pack.cand_resid[j][bi])
        q = float(pack.cand_prob[j][bi]) if len(pack.cand_prob[j]) else 1.0
        w[j] = q if np.isfinite(q) and q > 0 else 1.0
    mask = np.ones(n, dtype=bool)
    mask[station_i] = False
    usable = np.where(mask & np.isfinite(sel))[0]
    out = np.full(len(pack.cand_resid[station_i]), np.nan)
    if len(usable) < min_neighbors:
        return out
    med, _ = _event_center(sel[usable], w[usable], aggregator=aggregator)
    if not np.isfinite(med):
        return out
    resid = pack.cand_resid[station_i]
    for c in range(len(resid)):
        if np.isfinite(resid[c]):
            out[c] = abs(float(resid[c]) - med)
    return out
