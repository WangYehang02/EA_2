from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.multistation.ring_consistency import RingGateConfig, absolute_travel_times, apply_ring_consistency_gate


def test_absolute_travel_times_basic():
    meta = pd.DataFrame(
        {
            "origin_time": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "trace_start_time": ["2020-01-01T00:00:10Z", "2020-01-01T00:00:10Z"],
            "sampling_rate_hz": [100.0, 100.0],
        }
    )
    tau = absolute_travel_times(
        origin_time=meta["origin_time"],
        trace_start_time=meta["trace_start_time"],
        sample=np.array([500.0, 1000.0]),
        sampling_rate_hz=meta["sampling_rate_hz"],
    )
    assert abs(tau[0] - 15.0) < 1e-6
    assert abs(tau[1] - 20.0) < 1e-6


def test_ring_gate_abstains_without_support():
    # One event, 4 same-ring stations; A is a spurious early pick, B/C/D agree.
    rows = []
    for sta, dis, s_samp, pred in [
        ("A", 30.0, 3000, 1000),  # wrong early
        ("B", 31.0, 3000, 3000),
        ("C", 32.0, 3000, 3000),
        ("D", 33.0, 3000, 3010),
    ]:
        rows.append(
            {
                "trace_name": f"e1.IV.{sta}",
                "event_id": "e1",
                "station": sta,
                "distance_km": dis,
                "origin_time": "2020-01-01T00:00:00Z",
                "trace_start_time": "2020-01-01T00:00:00Z",
                "sampling_rate_hz": 100.0,
                "s_arrival_sample": float(s_samp),
            }
        )
    meta = pd.DataFrame(rows)
    pred = np.array([1000.0, 3000.0, 3000.0, 3010.0])
    cfg = RingGateConfig(
        ring_km=5.0,
        time_window_s=1.0,
        min_neighbors=2,
        rule="min_support",
        min_support=2,
        neighbor_mode="pred",
    )
    out = apply_ring_consistency_gate(meta, pred, cfg=cfg)
    assert out["abstained"][0]
    assert not out["abstained"][1]
    assert not out["abstained"][2]
    assert not out["abstained"][3]
    assert not np.isfinite(out["gated_pred_samples"][0])
