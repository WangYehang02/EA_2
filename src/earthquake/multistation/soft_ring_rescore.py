"""Candidate-level soft same-ring multi-station rescoring (experimental).

Final picks always come from the existing UNION candidate set.
Does not modify locked Stage-6 confirm artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Mode = Literal["absolute_s", "sp", "hybrid"]
NeighborCandMode = Literal["top1", "all_candidates"]


@dataclass(frozen=True)
class SoftRingRescoreConfig:
    mode: Mode = "sp"
    neighbor_candidate_mode: NeighborCandMode = "top1"
    sigma_distance_km: float = 4.0
    sigma_s_s: float = 1.0
    sigma_sp_s: float = 0.5
    lambda_s: float = 0.0
    lambda_sp: float = 0.25
    max_distance_diff_km: float | None = 10.0
    use_neighbor_confidence: bool = True
    eps: float = 1e-8


def fixed_candidate_scores(
    *,
    cand_sample: np.ndarray,
    cand_prob: np.ndarray,
    expected_s_sample: np.ndarray,
    history_sigma_samples: np.ndarray,
    history_available: np.ndarray,
    lw: float = 0.5,
    lh: float = 2.0,
    lp: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    p = np.asarray(cand_prob, dtype=np.float64)
    p = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
    samp = np.asarray(cand_sample, dtype=np.float64)
    exp = np.asarray(expected_s_sample, dtype=np.float64)
    sig = np.asarray(history_sigma_samples, dtype=np.float64)
    hav = np.asarray(history_available, dtype=bool)
    hist = np.zeros_like(samp, dtype=np.float64)
    ok = hav & np.isfinite(samp) & np.isfinite(exp) & np.isfinite(sig) & (sig > 1e-6)
    hist[ok] = np.exp(-0.5 * ((samp[ok] - exp[ok]) / sig[ok]) ** 2)
    score = lw * np.log(p + eps) + np.where(hav, lh * np.log(hist + eps), 0.0)
    _ = lp
    return score


def absolute_travel_time_s(
    *,
    origin_time,
    trace_start_time,
    sample: np.ndarray,
    sampling_rate_hz: np.ndarray | float,
) -> np.ndarray:
    origin = pd.to_datetime(origin_time, utc=True, errors="coerce")
    start = pd.to_datetime(trace_start_time, utc=True, errors="coerce")
    offset = (start - origin).dt.total_seconds().to_numpy(dtype=np.float64)
    sr = np.asarray(sampling_rate_hz, dtype=np.float64)
    if sr.ndim == 0:
        sr = np.full(len(sample), float(sr))
    samp = np.asarray(sample, dtype=np.float64)
    out = np.full(len(samp), np.nan, dtype=np.float64)
    ok = np.isfinite(offset) & np.isfinite(samp) & np.isfinite(sr) & (sr > 0)
    out[ok] = offset[ok] + samp[ok] / sr[ok]
    return out


@dataclass
class EventPack:
    trace_names: list[str]
    distance_km: np.ndarray
    pred_p: np.ndarray
    sr: np.ndarray
    # ragged candidate arrays stored as list
    cand_sample: list[np.ndarray]
    cand_prob: list[np.ndarray]
    cand_score: list[np.ndarray]
    cand_tau: list[np.ndarray]
    cand_sp: list[np.ndarray]
    baseline_idx: np.ndarray
    baseline_sample: np.ndarray


def build_event_packs(table: pd.DataFrame) -> dict[str, EventPack]:
    """One-time index: event_id -> EventPack."""
    need_cols = [
        "trace_name",
        "event_id",
        "distance_km",
        "candidate_index",
        "candidate_sample",
        "cand_prob",
        "fixed_score",
        "pred_p_sample",
        "sampling_rate_hz",
        "baseline_selected",
        "origin_time",
        "trace_start_time",
    ]
    df = table[need_cols].copy()
    df["tau_s"] = absolute_travel_time_s(
        origin_time=df["origin_time"],
        trace_start_time=df["trace_start_time"],
        sample=df["candidate_sample"].to_numpy(float),
        sampling_rate_hz=df["sampling_rate_hz"].to_numpy(float),
    )
    df["_sp"] = (df["candidate_sample"].to_numpy(float) - df["pred_p_sample"].to_numpy(float)) / df[
        "sampling_rate_hz"
    ].to_numpy(float)

    packs: dict[str, EventPack] = {}
    for eid, g in df.groupby("event_id", sort=False):
        tnames = g.groupby("trace_name", sort=False).size().index.astype(str).tolist()
        distance, pred_p, sr = [], [], []
        cs_l, cp_l, sc_l, tau_l, sp_l = [], [], [], [], []
        bidx, bsam = [], []
        for tn in tnames:
            gg = g[g.trace_name.astype(str) == tn].sort_values("candidate_index")
            distance.append(float(gg.distance_km.iloc[0]))
            pred_p.append(float(gg.pred_p_sample.iloc[0]))
            sr.append(float(gg.sampling_rate_hz.iloc[0]))
            cs_l.append(gg.candidate_sample.to_numpy(float))
            cp_l.append(gg.cand_prob.to_numpy(float))
            sc_l.append(gg.fixed_score.to_numpy(float))
            tau_l.append(gg.tau_s.to_numpy(float))
            sp_l.append(gg._sp.to_numpy(float))
            bs = gg.baseline_selected.to_numpy(bool)
            if bs.any():
                bi = int(np.argmax(bs))
            else:
                bi = int(np.nanargmax(gg.fixed_score.to_numpy(float)))
            bidx.append(bi)
            bsam.append(float(cs_l[-1][bi]))
        packs[str(eid)] = EventPack(
            trace_names=tnames,
            distance_km=np.asarray(distance, float),
            pred_p=np.asarray(pred_p, float),
            sr=np.asarray(sr, float),
            cand_sample=cs_l,
            cand_prob=cp_l,
            cand_score=sc_l,
            cand_tau=tau_l,
            cand_sp=sp_l,
            baseline_idx=np.asarray(bidx, int),
            baseline_sample=np.asarray(bsam, float),
        )
    return packs


def _rescore_pack(
    pack: EventPack,
    cfg: SoftRingRescoreConfig,
    *,
    sp_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = len(pack.trace_names)
    new_idx = pack.baseline_idx.copy()
    new_sample = pack.baseline_sample.copy()
    n_neighbors = np.zeros(n, dtype=np.int32)
    eff_w = np.zeros(n, dtype=np.float64)
    ring_sel = np.full(n, np.nan)
    ring_orig = np.full(n, np.nan)
    changed = np.zeros(n, dtype=bool)

    # selected neighbor refs
    sel_sp = np.full(n, np.nan)
    sel_tau = np.full(n, np.nan)
    sel_q = np.ones(n)
    for i in range(n):
        bi = int(pack.baseline_idx[i])
        if len(pack.cand_sample[i]) == 0:
            continue
        if sp_override is not None and np.isfinite(sp_override[i]):
            sel_sp[i] = float(sp_override[i])
        else:
            sel_sp[i] = float(pack.cand_sp[i][bi])
        sel_tau[i] = float(pack.cand_tau[i][bi])
        q = float(pack.cand_prob[i][bi])
        sel_q[i] = q if (cfg.use_neighbor_confidence and np.isfinite(q)) else 1.0

    sigma_d = max(cfg.sigma_distance_km, 1e-6)
    sigma_s = max(cfg.sigma_s_s, 1e-6)
    sigma_sp = max(cfg.sigma_sp_s, 1e-6)
    use_s = cfg.mode in ("absolute_s", "hybrid") and cfg.lambda_s != 0
    use_sp = cfg.mode in ("sp", "hybrid") and cfg.lambda_sp != 0
    # If mode absolute_s with lambda_s>0, use_s True; etc.
    if cfg.mode == "absolute_s":
        use_s, use_sp = True, False
    elif cfg.mode == "sp":
        use_s, use_sp = False, True
    else:
        use_s, use_sp = True, True

    for i in range(n):
        cs = pack.cand_sample[i]
        k = len(cs)
        if k == 0:
            continue
        dd = np.abs(pack.distance_km - pack.distance_km[i])
        neigh = np.where(np.arange(n) != i)[0]
        if cfg.max_distance_diff_km is not None:
            neigh = neigh[dd[neigh] <= float(cfg.max_distance_diff_km)]
        if cfg.neighbor_candidate_mode == "top1":
            usable = neigh[np.isfinite(sel_sp[neigh])]
        else:
            usable = np.asarray([j for j in neigh if len(pack.cand_sp[j]) > 0], dtype=np.int64)
        n_neighbors[i] = int(len(usable))
        if len(usable) == 0:
            continue

        wd = np.exp(-0.5 * (dd[usable] / sigma_d) ** 2)
        if cfg.neighbor_candidate_mode == "top1":
            den = float(np.sum(wd * sel_q[usable]))
        else:
            den = float(np.sum(wd))
        if den <= 0:
            continue

        r_s = np.zeros(k)
        r_sp = np.zeros(k)
        sp_i = pack.cand_sp[i]
        tau_i = pack.cand_tau[i]

        for c in range(k):
            num_s = 0.0
            num_sp = 0.0
            if cfg.neighbor_candidate_mode == "top1":
                if use_sp and np.isfinite(sp_i[c]):
                    dlt = sp_i[c] - sel_sp[usable]
                    wt = np.exp(-0.5 * (dlt / sigma_sp) ** 2)
                    num_sp = float(np.sum(wd * sel_q[usable] * wt))
                if use_s and np.isfinite(tau_i[c]):
                    dlt = tau_i[c] - sel_tau[usable]
                    wt = np.exp(-0.5 * (dlt / sigma_s) ** 2)
                    ok = np.isfinite(sel_tau[usable])
                    num_s = float(np.sum((wd * sel_q[usable] * wt)[ok]))
            else:
                for t, j in enumerate(usable):
                    wdj = float(wd[t])
                    qj = pack.cand_prob[j]
                    qj = np.where(np.isfinite(qj), qj, 1.0)
                    if not cfg.use_neighbor_confidence:
                        qj = np.ones_like(qj)
                    if use_sp and np.isfinite(sp_i[c]):
                        wt = np.exp(-0.5 * ((sp_i[c] - pack.cand_sp[j]) / sigma_sp) ** 2)
                        num_sp += wdj * float(np.max(qj * wt))
                    if use_s and np.isfinite(tau_i[c]):
                        tt = pack.cand_tau[j]
                        wt = np.exp(-0.5 * ((tau_i[c] - tt) / sigma_s) ** 2)
                        wt = np.where(np.isfinite(tt), wt, 0.0)
                        num_sp_term = float(np.max(qj * wt))
                        num_s += wdj * num_sp_term
            r_s[c] = num_s / den
            r_sp[c] = num_sp / den

        base = pack.cand_score[i]
        if cfg.mode == "absolute_s":
            final = base + cfg.lambda_s * r_s
            r_used = r_s
        elif cfg.mode == "sp":
            final = base + cfg.lambda_sp * r_sp
            r_used = r_sp
        else:
            final = base + cfg.lambda_s * r_s + cfg.lambda_sp * r_sp
            r_used = cfg.lambda_s * r_s + cfg.lambda_sp * r_sp

        jbest = int(np.nanargmax(final))
        new_idx[i] = jbest
        new_sample[i] = float(cs[jbest])
        ring_sel[i] = float(r_used[jbest])
        ring_orig[i] = float(r_used[int(pack.baseline_idx[i])])
        eff_w[i] = float(np.mean(wd))
        changed[i] = jbest != int(pack.baseline_idx[i])

    return {
        "selected_index": new_idx,
        "selected_sample": new_sample,
        "n_neighbors": n_neighbors,
        "effective_neighbor_weight": eff_w,
        "ring_score_selected": ring_sel,
        "ring_score_original_top1": ring_orig,
        "changed_candidate": changed,
    }


def apply_soft_ring_rescore_packs(
    packs: dict[str, EventPack],
    cfg: SoftRingRescoreConfig,
    *,
    shuffle_event_id: bool = False,
    shuffle_neighbor_sp: bool = False,
    neighbor_sp_by_trace: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply soft rescoring across event packs.

    neighbor_sp_by_trace: optional trace_name -> ΔSP (s) used ONLY as neighbor
    reference (sel_sp). Target candidates are never rewritten. For oracle/leakage
    diagnostics when values come from catalog S + predicted P.
    """
    rng = rng or np.random.default_rng(0)

    rows = []
    items = list(packs.items())
    if shuffle_event_id:
        # Break event structure: assign each station to a random other event pack by
        # concatenating all stations then re-splitting with same sizes — expensive.
        # Faster diagnostic: within each pack, randomly permute distances among stations.
        for eid, pack in items:
            p = EventPack(
                trace_names=pack.trace_names,
                distance_km=pack.distance_km.copy(),
                pred_p=pack.pred_p,
                sr=pack.sr,
                cand_sample=pack.cand_sample,
                cand_prob=pack.cand_prob,
                cand_score=pack.cand_score,
                cand_tau=pack.cand_tau,
                cand_sp=pack.cand_sp,
                baseline_idx=pack.baseline_idx,
                baseline_sample=pack.baseline_sample,
            )
            rng.shuffle(p.distance_km)  # destroy distance geometry
            res = _rescore_pack(p, cfg)
            for i, tn in enumerate(p.trace_names):
                rows.append(
                    {
                        "trace_name": tn,
                        "event_id": eid,
                        "baseline_sample": float(p.baseline_sample[i]),
                        "new_sample": float(res["selected_sample"][i]),
                        "baseline_cand_index": int(p.baseline_idx[i]),
                        "new_cand_index": int(res["selected_index"][i]),
                        "n_neighbors": int(res["n_neighbors"][i]),
                        "effective_neighbor_weight": float(res["effective_neighbor_weight"][i]),
                        "ring_score_selected": float(res["ring_score_selected"][i])
                        if np.isfinite(res["ring_score_selected"][i])
                        else np.nan,
                        "ring_score_original_top1": float(res["ring_score_original_top1"][i])
                        if np.isfinite(res["ring_score_original_top1"][i])
                        else np.nan,
                        "changed_candidate": bool(res["changed_candidate"][i]),
                        "distance_km": float(pack.distance_km[i]),
                    }
                )
        return pd.DataFrame(rows)

    for eid, pack in items:
        sp_override = None
        if neighbor_sp_by_trace is not None:
            sp_override = np.array(
                [float(neighbor_sp_by_trace.get(tn, np.nan)) for tn in pack.trace_names],
                dtype=np.float64,
            )
        elif shuffle_neighbor_sp:
            sp_override = np.array(
                [float(pack.cand_sp[i][pack.baseline_idx[i]]) if len(pack.cand_sp[i]) else np.nan for i in range(len(pack.trace_names))],
                dtype=np.float64,
            )
            rng.shuffle(sp_override)
        res = _rescore_pack(pack, cfg, sp_override=sp_override)
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
                    "effective_neighbor_weight": float(res["effective_neighbor_weight"][i]),
                    "ring_score_selected": float(res["ring_score_selected"][i])
                    if np.isfinite(res["ring_score_selected"][i])
                    else np.nan,
                    "ring_score_original_top1": float(res["ring_score_original_top1"][i])
                    if np.isfinite(res["ring_score_original_top1"][i])
                    else np.nan,
                    "changed_candidate": bool(res["changed_candidate"][i]),
                    "distance_km": float(pack.distance_km[i]),
                }
            )
    return pd.DataFrame(rows)


# Back-compat wrapper used by older call sites / tests
def apply_soft_ring_rescore(table: pd.DataFrame, cfg: SoftRingRescoreConfig, **kwargs) -> pd.DataFrame:
    packs = build_event_packs(table)
    return apply_soft_ring_rescore_packs(packs, cfg, **kwargs)
