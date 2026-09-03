from __future__ import annotations

import math
from typing import Iterable

import numpy as np

DEPTH_BINS = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, math.inf)]
DEPTH_BIN_LABELS = ["0-5", "5-10", "10-20", "20-40", "40+"]


def depth_bin(depth_km: float) -> int:
    if depth_km is None or (isinstance(depth_km, float) and not np.isfinite(depth_km)):
        return len(DEPTH_BINS) - 1
    d = float(depth_km)
    for i, (lo, hi) in enumerate(DEPTH_BINS):
        if lo <= d < hi:
            return i
    return len(DEPTH_BINS) - 1


def depth_bin_label(idx: int) -> str:
    return DEPTH_BIN_LABELS[int(idx)]


def source_region(lat: float, lon: float, grid_size: float) -> tuple[int, int]:
    return (int(math.floor(float(lat) / grid_size)), int(math.floor(float(lon) / grid_size)))


def station_id(network: str, station: str, location: str | None, channel_prefix: str) -> str:
    loc = "" if location is None or (isinstance(location, float) and np.isnan(location)) else str(location)
    return f"{network}.{station}.{loc}.{channel_prefix}"


def path_key(region: tuple[int, int], depth_idx: int, station: str) -> str:
    return f"{region[0]}:{region[1]}|{depth_idx}|{station}"


def parse_path_key(key: str) -> tuple[tuple[int, int], int, str]:
    region_s, depth_s, station = key.split("|", 2)
    r0, r1 = region_s.split(":")
    return (int(r0), int(r1)), int(depth_s), station


def neighbor_regions(region: tuple[int, int], radius: int = 1) -> list[tuple[int, int]]:
    i, j = region
    out = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            out.append((i + di, j + dj))
    return out


def adjacent_depth_bins(depth_idx: int) -> list[int]:
    out = []
    for d in (depth_idx - 1, depth_idx + 1):
        if 0 <= d < len(DEPTH_BINS):
            out.append(d)
    return out


def region_center_latlon(region: tuple[int, int], grid_size: float) -> tuple[float, float]:
    return ((region[0] + 0.5) * grid_size, (region[1] + 0.5) * grid_size)
