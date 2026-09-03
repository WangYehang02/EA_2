from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from earthquake.history.region import (
    adjacent_depth_bins,
    depth_bin,
    neighbor_regions,
    path_key,
    region_center_latlon,
    source_region,
    station_id,
)
from earthquake.utils import haversine_km, mad


@dataclass
class PathObservation:
    event_id: str
    origin_time: pd.Timestamp
    source_lat: float
    source_lon: float
    source_depth_km: float
    tau_p: float
    tau_s: float
    delta_sp: float


@dataclass
class PathStats:
    history_count: int = 0
    tau_p_median: float = float("nan")
    tau_p_mad: float = float("nan")
    tau_s_median: float = float("nan")
    tau_s_mad: float = float("nan")
    delta_sp_median: float = float("nan")
    delta_sp_mad: float = float("nan")
    last_history_time: pd.Timestamp | None = None
    nearest_historical_source_distance_km: float = float("nan")
    history_available: bool = False
    matched_key: str | None = None
    fallback_level: int = -1


class TemporalHistoryStore:
    """In-memory temporal path history with no-future-leakage update protocol.

    Usage for each event in time order:
      stats = store.query(...)
      ... evaluate using stats ...
      store.update_event(...)
    """

    def __init__(self, grid_size: float = 0.2, min_history: int = 5):
        self.grid_size = float(grid_size)
        self.min_history = int(min_history)
        self._obs: dict[str, list[PathObservation]] = defaultdict(list)

    def _station(self, row: pd.Series) -> str:
        return station_id(
            str(row["network"]),
            str(row["station"]),
            row.get("location", ""),
            str(row["channel_prefix"]),
        )

    def _key_for_row(self, row: pd.Series) -> str:
        region = source_region(float(row["source_latitude"]), float(row["source_longitude"]), self.grid_size)
        dbin = depth_bin(float(row["source_depth_km"]) if pd.notna(row.get("source_depth_km")) else float("nan"))
        return path_key(region, dbin, self._station(row))

    @staticmethod
    def compute_relative_times(row: pd.Series) -> tuple[float, float, float] | None:
        """Return tau_p, tau_s, delta_sp in seconds, or None if unavailable."""
        origin = row["origin_time"]
        if pd.isna(origin):
            return None

        def _arrival_seconds(prefix: str) -> float | None:
            tcol = f"{prefix}_arrival_time"
            scol = f"{prefix}_arrival_sample"
            if tcol in row and pd.notna(row[tcol]):
                return float((pd.Timestamp(row[tcol]) - pd.Timestamp(origin)).total_seconds())
            if scol in row and pd.notna(row[scol]) and pd.notna(row.get("trace_start_time")) and pd.notna(row.get("sampling_rate_hz")):
                # absolute arrival = start + sample/sr ; tau = arrival - origin
                arrival = pd.Timestamp(row["trace_start_time"]) + pd.to_timedelta(
                    float(row[scol]) / float(row["sampling_rate_hz"]), unit="s"
                )
                return float((arrival - pd.Timestamp(origin)).total_seconds())
            # fallback to catalog travel time if present
            tt = row.get(f"path_travel_time_{prefix}_s")
            if pd.notna(tt):
                return float(tt)
            return None

        tau_p = _arrival_seconds("p")
        tau_s = _arrival_seconds("s")
        if tau_p is None or tau_s is None:
            return None
        return float(tau_p), float(tau_s), float(tau_s - tau_p)

    def _stats_from_obs(
        self,
        obs_list: list[PathObservation],
        query_lat: float,
        query_lon: float,
        query_time: pd.Timestamp,
        key: str,
        fallback_level: int,
    ) -> PathStats:
        if not obs_list:
            return PathStats(history_available=False, matched_key=key, fallback_level=fallback_level)
        tau_p = np.array([o.tau_p for o in obs_list], dtype=np.float64)
        tau_s = np.array([o.tau_s for o in obs_list], dtype=np.float64)
        dsp = np.array([o.delta_sp for o in obs_list], dtype=np.float64)
        last_t = max(o.origin_time for o in obs_list)
        dists = [haversine_km(query_lat, query_lon, o.source_lat, o.source_lon) for o in obs_list]
        return PathStats(
            history_count=len(obs_list),
            tau_p_median=float(np.median(tau_p)),
            tau_p_mad=mad(tau_p),
            tau_s_median=float(np.median(tau_s)),
            tau_s_mad=mad(tau_s),
            delta_sp_median=float(np.median(dsp)),
            delta_sp_mad=mad(dsp),
            last_history_time=last_t,
            nearest_historical_source_distance_km=float(np.min(dists)) if dists else float("nan"),
            history_available=len(obs_list) >= self.min_history,
            matched_key=key,
            fallback_level=fallback_level,
        )

    def _gather(self, keys: list[str], before_time: pd.Timestamp) -> list[PathObservation]:
        out: list[PathObservation] = []
        for k in keys:
            for o in self._obs.get(k, []):
                if o.origin_time < before_time:
                    out.append(o)
        return out

    def query_row(self, row: pd.Series) -> PathStats:
        origin = pd.Timestamp(row["origin_time"])
        lat = float(row["source_latitude"])
        lon = float(row["source_longitude"])
        region = source_region(lat, lon, self.grid_size)
        dbin = depth_bin(float(row["source_depth_km"]) if pd.notna(row.get("source_depth_km")) else float("nan"))
        sid = self._station(row)
        primary = path_key(region, dbin, sid)

        # Level 0: exact key
        obs = self._gather([primary], origin)
        stats = self._stats_from_obs(obs, lat, lon, origin, primary, 0)
        if stats.history_count >= self.min_history:
            stats.history_available = True
            return stats

        # Level 1: same region+station, adjacent depth
        keys1 = [path_key(region, d, sid) for d in adjacent_depth_bins(dbin)]
        obs = self._gather([primary] + keys1, origin)
        stats = self._stats_from_obs(obs, lat, lon, origin, primary, 1)
        if stats.history_count >= self.min_history:
            stats.history_available = True
            return stats

        # Level 2: neighbor spatial grids, same station+depth
        keys2 = [path_key(r, dbin, sid) for r in neighbor_regions(region, radius=1)]
        obs = self._gather([primary] + keys1 + keys2, origin)
        stats = self._stats_from_obs(obs, lat, lon, origin, primary, 2)
        if stats.history_count >= self.min_history:
            stats.history_available = True
            return stats

        # Level 3: K nearest source regions for same station (by region center distance)
        # Gather all keys for this station
        station_keys = [k for k in self._obs.keys() if k.endswith(f"|{sid}")]
        scored = []
        for k in station_keys:
            try:
                reg, _, _ = k.split("|", 2)
                r0, r1 = map(int, reg.split(":"))
            except Exception:
                continue
            clat, clon = region_center_latlon((r0, r1), self.grid_size)
            scored.append((haversine_km(lat, lon, clat, clon), k))
        scored.sort(key=lambda x: x[0])
        keys3 = [k for _, k in scored[:8]]
        obs = self._gather(keys3, origin)
        stats = self._stats_from_obs(obs, lat, lon, origin, primary, 3)
        if stats.history_count >= self.min_history:
            stats.history_available = True
            return stats

        # Insufficient history
        stats.history_available = False
        stats.fallback_level = 4
        return stats

    def update_row(self, row: pd.Series) -> None:
        rel = self.compute_relative_times(row)
        if rel is None:
            return
        tau_p, tau_s, delta_sp = rel
        key = self._key_for_row(row)
        obs = PathObservation(
            event_id=str(row["event_id"]),
            origin_time=pd.Timestamp(row["origin_time"]),
            source_lat=float(row["source_latitude"]),
            source_lon=float(row["source_longitude"]),
            source_depth_km=float(row["source_depth_km"]) if pd.notna(row.get("source_depth_km")) else float("nan"),
            tau_p=tau_p,
            tau_s=tau_s,
            delta_sp=delta_sp,
        )
        # Enforce no future contamination relative to existing store ordering by caller.
        self._obs[key].append(obs)

    def update_event(self, event_rows: pd.DataFrame) -> None:
        for _, row in event_rows.iterrows():
            self.update_row(row)

    def n_paths(self) -> int:
        return len(self._obs)

    def path_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._obs.items()}
