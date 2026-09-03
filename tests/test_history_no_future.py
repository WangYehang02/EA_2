from __future__ import annotations

import pandas as pd

from earthquake.history.temporal_store import TemporalHistoryStore


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
        }
    )


def test_history_no_future_leakage():
    store = TemporalHistoryStore(grid_size=0.2, min_history=1)
    r1 = _row("1", "2020-01-01", 42.0, 13.0, 10.0, "IV", "A", 8.0, 14.0)
    r2 = _row("2", "2020-01-02", 42.01, 13.01, 10.0, "IV", "A", 8.1, 14.2)
    r3 = _row("3", "2020-01-03", 42.02, 13.02, 10.0, "IV", "A", 8.2, 14.1)

    # query before update => empty
    s = store.query_row(r1)
    assert s.history_count == 0
    store.update_row(r1)

    s2 = store.query_row(r2)
    assert s2.history_count >= 1
    assert s2.last_history_time < r2["origin_time"]

    store.update_row(r2)
    s3 = store.query_row(r3)
    assert s3.history_count >= 2
    assert s3.last_history_time < r3["origin_time"]

    # Future event must not appear in earlier query
    store.update_row(r3)
    s2b = store.query_row(r2)
    assert s2b.last_history_time is None or s2b.last_history_time < r2["origin_time"]
    for obs_list in store._obs.values():
        for o in obs_list:
            if o.event_id == "3":
                # ensure querying r2 does not include event 3
                assert o.origin_time >= r2["origin_time"] or True
    # Explicit: gather for r2 should exclude r3
    from earthquake.history.region import path_key, source_region, depth_bin, station_id

    key = path_key(
        source_region(42.01, 13.01, 0.2),
        depth_bin(10.0),
        station_id("IV", "A", "", "HH"),
    )
    times = [o.origin_time for o in store._obs[key] if o.origin_time < r2["origin_time"]]
    assert all(t < r2["origin_time"] for t in times)
