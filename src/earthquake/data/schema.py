from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


# Logical field names used throughout the codebase.
LOGICAL_FIELDS = [
    "event_id",
    "origin_time",
    "source_latitude",
    "source_longitude",
    "source_depth_km",
    "source_magnitude",
    "network",
    "station",
    "location",
    "channel_prefix",
    "station_latitude",
    "station_longitude",
    "station_elevation_m",
    "trace_name",
    "trace_start_time",
    "sampling_rate_hz",
    "n_samples",
    "p_arrival_sample",
    "s_arrival_sample",
    "p_arrival_time",
    "s_arrival_time",
    "p_uncertainty_s",
    "s_uncertainty_s",
    "distance_km",
    "hyp_distance_km",
    "azimuth_deg",
    "snr_db",
    "path_travel_time_p_s",
    "path_travel_time_s_s",
]


# Candidate raw column names for automatic schema mapping.
CANDIDATES: dict[str, list[str]] = {
    "event_id": ["source_id", "event_id", "origin_id"],
    "origin_time": ["source_origin_time", "origin_time", "event_time"],
    "source_latitude": ["source_latitude_deg", "source_latitude", "latitude"],
    "source_longitude": ["source_longitude_deg", "source_longitude", "longitude"],
    "source_depth_km": ["source_depth_km", "source_depth", "depth"],
    "source_magnitude": ["source_magnitude", "magnitude", "mag"],
    "network": ["station_network_code", "network_code", "network"],
    "station": ["station_code", "station"],
    "location": ["station_location_code", "location_code", "location"],
    "channel_prefix": ["station_channels", "channel", "channels"],
    "station_latitude": ["station_latitude_deg", "station_latitude"],
    "station_longitude": ["station_longitude_deg", "station_longitude"],
    "station_elevation_m": ["station_elevation_m", "station_elevation"],
    "trace_name": ["trace_name", "trace_id", "waveform_id"],
    "trace_start_time": ["trace_start_time", "start_time"],
    "sampling_rate_hz": ["trace_sampling_rate_hz"],  # may be derived from trace_dt_s
    "n_samples": ["trace_npts", "npts", "n_samples"],
    "p_arrival_sample": ["trace_P_arrival_sample", "p_arrival_sample"],
    "s_arrival_sample": ["trace_S_arrival_sample", "s_arrival_sample"],
    "p_arrival_time": ["trace_P_arrival_time", "p_arrival_time"],
    "s_arrival_time": ["trace_S_arrival_time", "s_arrival_time"],
    "p_uncertainty_s": ["trace_P_uncertainty_s", "p_uncertainty_s"],
    "s_uncertainty_s": ["trace_S_uncertainty_s", "s_uncertainty_s"],
    "distance_km": ["path_ep_distance_km", "distance_km", "epicentral_distance_km"],
    "hyp_distance_km": ["path_hyp_distance_km", "hypocentral_distance_km"],
    "azimuth_deg": ["path_azimuth_deg", "azimuth_deg"],
    "snr_db": ["trace_Z_snr_db", "trace_E_snr_db", "snr_db"],
    "path_travel_time_p_s": ["path_travel_time_P_s"],
    "path_travel_time_s_s": ["path_travel_time_S_s"],
}


@dataclass
class SchemaMapping:
    mapping: dict[str, str | None]
    derived: dict[str, str]
    events_columns: list[str]
    noise_columns: list[str]

    def raw(self, logical: str) -> str | None:
        return self.mapping.get(logical)

    def require(self, logical: str) -> str:
        col = self.mapping.get(logical)
        if not col:
            raise KeyError(f"Logical field '{logical}' is not mapped to a CSV column")
        return col

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_columns(columns: list[str]) -> dict[str, str | None]:
    colset = set(columns)
    out: dict[str, str | None] = {}
    for logical, cands in CANDIDATES.items():
        found = None
        for c in cands:
            if c in colset:
                found = c
                break
        out[logical] = found
    return out
