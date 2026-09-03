"""Future DKPN checkpoint rules: per-epoch files, valid-best vs init+min_delta."""

from __future__ import annotations

from typing import Any

FIXED_0P2_METRIC = "val_s_f1_fixed0p2"
CALIBRATED_METRIC = "val_s_f1_calibrated"
MIN_DELTA_F1 = 0.01
OFFICIAL_PROBABILITY_THRESHOLD = 0.2


def is_valid_best(
    *,
    metric: float,
    init_metric: float,
    min_delta: float = MIN_DELTA_F1,
) -> bool:
    """F1 all-zero cannot mint a valid best. Must beat init by min_delta."""
    try:
        m = float(metric)
        init = float(init_metric)
    except (TypeError, ValueError):
        return False
    if m != m or init != init:  # NaN
        return False
    if m <= 0.0:
        return False
    return m > init + float(min_delta)


def epoch_checkpoint_name(virtual_epoch: int) -> str:
    return f"epoch_{int(virtual_epoch)}.pt"


def best_metric_filename(kind: str) -> str:
    if kind == "fixed0p2":
        return "best_metric_fixed0p2.pt"
    if kind == "calibrated":
        return "best_metric_calibrated.pt"
    raise ValueError(kind)


def checkpoint_metadata(
    *,
    virtual_epoch: int,
    metric_name: str,
    metric_value: float,
    init_metric: float,
    probability_threshold: float,
    valid_best: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "epoch": int(virtual_epoch),
        "virtual_epoch": int(virtual_epoch),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value) if metric_value == metric_value else None,
        "init_metric": float(init_metric) if init_metric == init_metric else None,
        "min_delta": MIN_DELTA_F1,
        "probability_threshold": float(probability_threshold),
        "valid_best": bool(valid_best),
        "fixed0p2_and_calibrated_are_separate_files": True,
    }
    if extra:
        meta.update(extra)
    return meta
