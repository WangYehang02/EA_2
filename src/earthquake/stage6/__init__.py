"""Stage 6 package: nested-split helpers and anti-leakage guards."""

from __future__ import annotations

from pathlib import Path

from earthquake.stage6.splits import (
    STAGE6_SUBSETS,
    assert_confirm_sealed,
    load_stage6_event_ids,
    load_stage6_trace_names,
    stage6_paths,
)
from earthquake.stage6.full_splits import (
    FULL_SUBSETS,
    assert_full_confirm_access_allowed,
    full_stage6_paths,
    load_full_event_ids,
    load_full_trace_names,
)

__all__ = [
    "STAGE6_SUBSETS",
    "FULL_SUBSETS",
    "assert_confirm_sealed",
    "assert_full_confirm_access_allowed",
    "load_stage6_event_ids",
    "load_stage6_trace_names",
    "load_full_event_ids",
    "load_full_trace_names",
    "stage6_paths",
    "full_stage6_paths",
]
