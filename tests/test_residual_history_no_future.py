"""Residual history must not use future events (reuse temporal store protocol)."""

from __future__ import annotations

import pandas as pd

from earthquake.history.residual_prior import path_stats_to_residual_stats
from earthquake.history.temporal_store import TemporalHistoryStore
from earthquake.history.travel_time_baseline import TravelTimeBaseline


def _row(eid, t, lat, lon, depth, net, sta, tau_p, tau_s):
    origin = pd.Timestamp(t, tz="UTC")
    return pd.Series(
        {
            "event_id": eid,
            "origin_time": origin,
            "source_latitude": lat,
            "source_longitude": lon,
            "source_depth_km": depth,
            "network": net,
            "station": sta,
            "location": "",
            "channel_prefix": "HH",
            "trace_start_time": origin - pd.Timedelta(seconds=5),
            "sampling_rate_hz": 100.0,
            "p_arrival_time": origin + pd.Timedelta(seconds=tau_p),
            "s_arrival_time": origin + pd.Timedelta(seconds=tau_s),
            "path_travel_time_p_s": tau_p,
            "path_travel_time_s_s": tau_s,
            "distance_km": 50.0,
        }
    )


def test_residual_history_no_future():
    store = TemporalHistoryStore(grid_size=0.2, min_history=1)
    base = TravelTimeBaseline(
        kind="dist_1d",
        meta={},
        table=pd.DataFrame(
            [{"dist_center": 50.0, "tau_p_median": 8.0, "tau_s_median": 14.0, "delta_sp_median": 6.0, "n": 1}]
        ),
    )
    r1 = _row("1", "2020-01-01", 42.0, 13.0, 10.0, "IV", "A", 8.0, 14.0)
    r2 = _row("2", "2020-01-02", 42.01, 13.01, 10.0, "IV", "A", 9.0, 15.0)

    s_before = store.query_row(r2)
    assert s_before.history_count == 0
    store.update_row(r1)
    s_after = store.query_row(r2)
    assert s_after.history_count == 1
    b = base.predict_row(r2)
    rs = path_stats_to_residual_stats(s_after, b)
    assert rs.history_available
    # residual vs base 8/14 with hist median 8/14 from r1
    assert abs(rs.residual_p_median - (8.0 - b["base_tau_p"])) < 1e-6
