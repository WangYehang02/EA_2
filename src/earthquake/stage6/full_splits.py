"""Stage 6 FULL split paths and confirm seal (independent of pilot splits/)."""

from __future__ import annotations

import json
from pathlib import Path

from earthquake.config import artifacts_dir

FULL_SUBSETS = (
    "stage6_picker_train",
    "stage6_ranker_train",
    "stage6_dev",
    "stage6_internal_confirm",
)


def full_stage6_paths() -> dict[str, Path]:
    root = artifacts_dir() / "results" / "stage6"
    return {
        "root": root,
        "splits": root / "splits_full",
        "confirm_seal": root / "splits_full" / "CONFIRM_SEALED",
        "method_lock": root / "method_lock_stage6.json",
        "full_split_audit": root / "full_split_audit.json",
    }


def load_full_event_ids(subset: str) -> list[str]:
    if subset not in FULL_SUBSETS:
        raise ValueError(subset)
    path = full_stage6_paths()["splits"] / f"{subset}_events.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_full_trace_names(subset: str) -> list[str]:
    """Load trace *names* (IDs only). Does not unlock waveform/label/prediction access."""
    if subset not in FULL_SUBSETS:
        raise ValueError(subset)
    path = full_stage6_paths()["splits"] / f"{subset}_traces.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def assert_full_confirm_access_allowed(*, purpose: str = "inference") -> None:
    """Any confirm waveform/label/prediction access requires method_lock."""
    paths = full_stage6_paths()
    if paths["method_lock"].exists():
        return
    raise RuntimeError(
        f"stage6_full internal_confirm {purpose} forbidden until "
        f"{paths['method_lock']} exists (CONFIRM_SEALED)."
    )


def assert_confirm_never_used(confirm_ids: list[str], prior_used: set[str]) -> None:
    bad = sorted(set(confirm_ids) & prior_used)
    if bad:
        raise RuntimeError(f"confirm contains {len(bad)} prior-used events, e.g. {bad[:5]}")


def write_full_confirm_seal(
    *,
    events_sha256: str,
    traces_sha256: str,
    n_events: int,
    n_traces: int,
) -> None:
    p = full_stage6_paths()["confirm_seal"]
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SEALED",
        "n_events": n_events,
        "n_traces": n_traces,
        "events_sha256": events_sha256,
        "traces_sha256": traces_sha256,
        "rule": "No confirm waveform/label/prediction access until method_lock_stage6.json",
    }
    p.write_text(json.dumps(payload, indent=2) + "\n")
