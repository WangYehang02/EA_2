"""Path / protocol guards for pairwise stages.

Training/pilot code must not read confirm OR phaseB full-dev.
Full-dev evaluation may read phaseB, but must never read confirm.
"""

from __future__ import annotations

from typing import Any

CONFIRM_NEEDLES = (
    "confirm",
    "confirmation",
    "final_confirm",
    "internal_confirm",
)


def assert_no_confirm_path(path: str | Any) -> None:
    """Hard-fail if path looks like a confirm artifact.

    Allowed: checking that a lock *file name* exists without reading confirm data
    is not done via this helper — pass only data-loading paths.
    """
    s = str(path).lower().replace("\\", "/")
    for n in CONFIRM_NEEDLES:
        if n in s:
            raise RuntimeError(f"CONFIRM path access forbidden: {path}")


def assert_allowed_fulldev_data_path(path: str | Any) -> None:
    """Full-dev evaluator: allow phaseB, forbid confirm."""
    assert_no_confirm_path(path)


def assert_no_confirm_or_phaseb_train_path(path: str | Any) -> None:
    """Pilot/train: forbid confirm and phaseB/full-dev evaluation data."""
    assert_no_confirm_path(path)
    s = str(path).lower().replace("\\", "/")
    blocked = (
        "phaseb_eval_manifest",
        "phaseb_eval_manifest.csv",
        "dev_union.parquet",
        "/phaseb/",
        "full-dev",
        "full_dev",
    )
    # allow mentions under pairwise_fulldev *output* dirs
    if "pairwise_fulldev" in s:
        return
    for b in blocked:
        if b in s:
            raise RuntimeError(f"PHASEB/FULL-DEV path forbidden for train/pilot: {path}")
