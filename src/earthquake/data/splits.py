from __future__ import annotations

from pathlib import Path

import pandas as pd

from earthquake.utils import write_lines


def event_time_split(
    events_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Split by unique event_id ordered by origin time. Event-level exclusive."""
    if "event_id" not in events_df.columns or "origin_time" not in events_df.columns:
        raise ValueError("events_df must contain event_id and origin_time")

    event_times = (
        events_df.dropna(subset=["event_id", "origin_time"])
        .groupby("event_id", sort=False)["origin_time"]
        .min()
        .sort_values()
    )
    ids = event_times.index.astype(str).tolist()
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder to test to guarantee coverage
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def noise_time_split(
    noise_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Split noise waveforms by waveform start time."""
    df = noise_df.dropna(subset=["trace_name", "trace_start_time"]).copy()
    df = df.sort_values("trace_start_time")
    names = df["trace_name"].astype(str).tolist()
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }


def save_event_splits(splits: dict[str, list[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        write_lines(out_dir / f"{split}_events.txt", ids)


def load_event_splits(split_dir: Path) -> dict[str, list[str]]:
    out = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}_events.txt"
        with open(path, "r", encoding="utf-8") as f:
            out[split] = [ln.strip() for ln in f if ln.strip()]
    return out


def assign_split_column(events_df: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    mapping = {}
    for split, ids in splits.items():
        for eid in ids:
            mapping[str(eid)] = split
    out = events_df.copy()
    out["split"] = out["event_id"].astype(str).map(mapping)
    return out
