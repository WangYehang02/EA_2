from __future__ import annotations

import pandas as pd

from earthquake.data.splits import event_time_split


def test_event_split_exclusive_and_temporal():
    rows = []
    for i in range(100):
        rows.append(
            {
                "event_id": f"e{i}",
                "origin_time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                "trace_name": f"t{i}a",
            }
        )
        rows.append(
            {
                "event_id": f"e{i}",
                "origin_time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                "trace_name": f"t{i}b",
            }
        )
    df = pd.DataFrame(rows)
    splits = event_time_split(df)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    # temporal order: last train before first val before first test
    def t(eid):
        return df[df.event_id == eid]["origin_time"].min()

    assert t(splits["train"][-1]) <= t(splits["val"][0]) <= t(splits["test"][0])
    # same event not in two splits
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids))
