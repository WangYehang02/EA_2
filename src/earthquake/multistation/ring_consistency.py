"""Same-ring S-arrival consistency gate (experimental multi-station denoising).

For each station prediction, look at other stations of the same event with
similar epicentral distance. If too few neighbors show an S pick/peak near the
same absolute travel time, treat the prediction as noise (abstain).

This module does not modify locked Stage-6 confirm artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


NeighborMode = Literal["pred", "catalog"]
RuleMode = Literal["min_support", "max_missing_frac", "majority"]


@dataclass(frozen=True)
class RingGateConfig:
    ring_km: float = 5.0
    time_window_s: float = 1.0
    min_neighbors: int = 2
    """Apply the gate only when at least this many same-ring neighbors exist."""

    rule: RuleMode = "min_support"
    min_support: int = 2
    """For rule=min_support: keep if >= this many neighbors support the pick."""

    max_missing_frac: float = 0.1
    """For rule=max_missing_frac: abstain if missing fraction > this (user draft)."""

    neighbor_mode: NeighborMode = "pred"
    """pred=neighbor model picks; catalog=neighbor labeled S times (semi-oracle)."""

    no_neighbor_policy: Literal["keep", "abstain"] = "keep"


def absolute_travel_times(
    *,
    origin_time: pd.Series,
    trace_start_time: pd.Series,
    sample: np.ndarray,
    sampling_rate_hz: np.ndarray | float,
) -> np.ndarray:
    """UTC-origin absolute travel time (s) for picks given as sample indices."""
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


def apply_ring_consistency_gate(
    meta: pd.DataFrame,
    pred_samples: np.ndarray,
    *,
    cfg: RingGateConfig,
    catalog_s_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return gated predictions and per-trace diagnostics.

    Parameters
    ----------
    meta:
        Must contain event_id, distance_km, origin_time, trace_start_time,
        sampling_rate_hz. Row order must match pred_samples.
    pred_samples:
        Model S pick sample index per row (NaN = already missing).
    catalog_s_samples:
        Optional labeled S sample index (defaults to meta['s_arrival_sample']).
    """
    n = len(meta)
    pred = np.asarray(pred_samples, dtype=np.float64).copy()
    if len(pred) != n:
        raise ValueError(f"pred length {len(pred)} != meta rows {n}")

    if catalog_s_samples is None:
        catalog_s_samples = meta["s_arrival_sample"].to_numpy(dtype=np.float64)
    else:
        catalog_s_samples = np.asarray(catalog_s_samples, dtype=np.float64)

    sr = meta["sampling_rate_hz"].to_numpy(dtype=np.float64)
    tau_pred = absolute_travel_times(
        origin_time=meta["origin_time"],
        trace_start_time=meta["trace_start_time"],
        sample=pred,
        sampling_rate_hz=sr,
    )
    tau_cat = absolute_travel_times(
        origin_time=meta["origin_time"],
        trace_start_time=meta["trace_start_time"],
        sample=catalog_s_samples,
        sampling_rate_hz=sr,
    )
    tau_ref = tau_pred if cfg.neighbor_mode == "pred" else tau_cat

    dis = meta["distance_km"].to_numpy(dtype=np.float64)
    event_ids = meta["event_id"].astype(str).to_numpy()

    gated = pred.copy()
    n_neighbors = np.zeros(n, dtype=np.int32)
    n_support = np.zeros(n, dtype=np.int32)
    applied = np.zeros(n, dtype=bool)
    abstained = np.zeros(n, dtype=bool)

    # Group indices by event
    by_event: dict[str, np.ndarray] = {}
    for i, eid in enumerate(event_ids):
        by_event.setdefault(eid, []).append(i)
    for eid, idxs in by_event.items():
        by_event[eid] = np.asarray(idxs, dtype=np.int64)

    for i in range(n):
        if not np.isfinite(pred[i]) or not np.isfinite(tau_pred[i]) or not np.isfinite(dis[i]):
            continue
        idxs = by_event[event_ids[i]]
        if len(idxs) <= 1:
            if cfg.no_neighbor_policy == "abstain":
                gated[i] = np.nan
                abstained[i] = True
            continue

        ddi = np.abs(dis[idxs] - dis[i])
        neigh = idxs[(ddi < cfg.ring_km) & (idxs != i)]
        # Neighbor must have a finite reference time in the chosen mode
        if cfg.neighbor_mode == "pred":
            have = neigh[np.isfinite(tau_pred[neigh])]
        else:
            have = neigh[np.isfinite(tau_cat[neigh])]
        # All geometric neighbors (with or without peak) for missing-fraction rule
        geom = neigh
        n_neighbors[i] = int(len(geom))

        if len(geom) < cfg.min_neighbors:
            if cfg.no_neighbor_policy == "abstain" and len(geom) == 0:
                gated[i] = np.nan
                abstained[i] = True
            continue

        applied[i] = True
        support_mask = np.isfinite(tau_ref[have]) & (np.abs(tau_ref[have] - tau_pred[i]) <= cfg.time_window_s)
        # have is subset of geom that have peaks; support among those with peaks near tau
        # For missing fraction: neighbors without finite tau_ref count as missing,
        # and those with |Δt|>window also count as not supporting.
        if cfg.neighbor_mode == "pred":
            tau_n = tau_pred[geom]
        else:
            tau_n = tau_cat[geom]
        support_mask_geom = np.isfinite(tau_n) & (np.abs(tau_n - tau_pred[i]) <= cfg.time_window_s)
        n_support[i] = int(support_mask_geom.sum())
        n_geom = int(len(geom))
        missing_frac = 1.0 - (n_support[i] / max(n_geom, 1))

        keep = True
        if cfg.rule == "min_support":
            keep = n_support[i] >= cfg.min_support
        elif cfg.rule == "max_missing_frac":
            keep = missing_frac <= cfg.max_missing_frac
        elif cfg.rule == "majority":
            keep = n_support[i] >= max(1, int(np.ceil(0.5 * n_geom)))
        else:
            raise ValueError(f"unknown rule: {cfg.rule}")

        if not keep:
            gated[i] = np.nan
            abstained[i] = True

    return {
        "gated_pred_samples": gated,
        "n_neighbors": n_neighbors,
        "n_support": n_support,
        "gate_applied": applied,
        "abstained": abstained,
        "tau_pred": tau_pred,
        "tau_catalog": tau_cat,
        "config": asdict(cfg),
        "stats": {
            "n_traces": n,
            "n_finite_input": int(np.isfinite(pred).sum()),
            "n_gate_applied": int(applied.sum()),
            "n_abstained": int(abstained.sum()),
            "abstain_rate": float(abstained.mean()),
            "mean_neighbors_when_applied": float(n_neighbors[applied].mean()) if applied.any() else 0.0,
            "mean_support_when_applied": float(n_support[applied].mean()) if applied.any() else 0.0,
        },
    }
