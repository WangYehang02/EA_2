from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from earthquake.data.schema import SchemaMapping


def find_events_csv(instance_root: Path) -> Path:
    events = instance_root / "events"
    preferred = [
        events / "metadata_Instance_events_v3.csv",
        events / "metadata_Instance_events_v2.csv",
        events / "metadata_Instance_events.csv",
    ]
    for p in preferred:
        if p.exists():
            return p
    matches = sorted(events.glob("metadata_Instance_events*.csv"))
    if not matches:
        raise FileNotFoundError(f"No events metadata CSV under {events}")
    return matches[0]


def find_noise_csv(instance_root: Path) -> Path:
    noise = instance_root / "noise"
    preferred = noise / "metadata_Instance_noise.csv"
    if preferred.exists():
        return preferred
    matches = sorted(noise.glob("metadata_Instance_noise*.csv"))
    if not matches:
        raise FileNotFoundError(f"No noise metadata CSV under {noise}")
    return matches[0]


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def iter_csv_chunks(path: Path, chunksize: int = 100_000, usecols=None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False)


def normalize_events_frame(df: pd.DataFrame, schema: SchemaMapping) -> pd.DataFrame:
    """Rename mapped columns to logical names and derive sampling rate if needed."""
    rename = {v: k for k, v in schema.mapping.items() if v is not None and v in df.columns}
    out = df.rename(columns=rename).copy()

    if "sampling_rate_hz" not in out.columns:
        if "trace_dt_s" in df.columns:
            out["sampling_rate_hz"] = 1.0 / pd.to_numeric(df["trace_dt_s"], errors="coerce")
        elif schema.mapping.get("sampling_rate_hz") is None and "trace_dt_s" in out.columns:
            out["sampling_rate_hz"] = 1.0 / pd.to_numeric(out["trace_dt_s"], errors="coerce")

    for col in [
        "origin_time",
        "trace_start_time",
        "p_arrival_time",
        "s_arrival_time",
    ]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    for col in [
        "source_latitude",
        "source_longitude",
        "source_depth_km",
        "source_magnitude",
        "station_latitude",
        "station_longitude",
        "station_elevation_m",
        "sampling_rate_hz",
        "n_samples",
        "p_arrival_sample",
        "s_arrival_sample",
        "p_uncertainty_s",
        "s_uncertainty_s",
        "distance_km",
        "hyp_distance_km",
        "azimuth_deg",
        "snr_db",
        "path_travel_time_p_s",
        "path_travel_time_s_s",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "location" in out.columns:
        out["location"] = out["location"].fillna("").astype(str)
    if "event_id" in out.columns:
        out["event_id"] = out["event_id"].astype(str)
    if "trace_name" in out.columns:
        out["trace_name"] = out["trace_name"].astype(str)

    return out


def normalize_noise_frame(df: pd.DataFrame, schema: SchemaMapping) -> pd.DataFrame:
    rename = {v: k for k, v in schema.mapping.items() if v is not None and v in df.columns}
    out = df.rename(columns=rename).copy()
    if "sampling_rate_hz" not in out.columns and "trace_dt_s" in df.columns:
        out["sampling_rate_hz"] = 1.0 / pd.to_numeric(df["trace_dt_s"], errors="coerce")
    if "trace_start_time" in out.columns:
        out["trace_start_time"] = pd.to_datetime(out["trace_start_time"], utc=True, errors="coerce")
    if "location" in out.columns:
        out["location"] = out["location"].fillna("").astype(str)
    if "trace_name" in out.columns:
        out["trace_name"] = out["trace_name"].astype(str)
    for col in ["station_latitude", "station_longitude", "station_elevation_m", "sampling_rate_hz", "n_samples"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
