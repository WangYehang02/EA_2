"""Stage 6 split I/O and confirm seal."""

from __future__ import annotations

from pathlib import Path

from earthquake.config import artifacts_dir

STAGE6_SUBSETS = (
    "stage6_picker_train",
    "stage6_ranker_train",
    "stage6_dev",
    "stage6_internal_confirm",
)


def stage6_paths() -> dict[str, Path]:
    root = artifacts_dir() / "results" / "stage6"
    return {
        "root": root,
        "splits": root / "splits",
        "logs": root / "logs",
        "models": artifacts_dir() / "models" / "stage6",
        "cache": artifacts_dir() / "cache" / "stage6",
        "method_lock": root / "method_lock_stage6.json",
        "confirm_seal": root / "splits" / "CONFIRM_SEALED",
    }


def load_stage6_event_ids(subset: str) -> list[str]:
    if subset not in STAGE6_SUBSETS:
        raise ValueError(subset)
    path = stage6_paths()["splits"] / f"{subset}_events.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_stage6_trace_names(subset: str) -> list[str]:
    if subset not in STAGE6_SUBSETS:
        raise ValueError(subset)
    path = stage6_paths()["splits"] / f"{subset}_traces.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def assert_confirm_inference_allowed() -> None:
    """Raise unless method_lock exists (one-shot confirm gate)."""
    paths = stage6_paths()
    if paths["method_lock"].exists():
        return
    raise RuntimeError(
        "stage6_internal_confirm inference forbidden until artifacts/results/stage6/method_lock_stage6.json exists"
    )


def assert_confirm_sealed(*, allow_if_method_locked: bool = True) -> None:
    """Backward-compatible alias used by confirm-facing scripts."""
    if allow_if_method_locked and stage6_paths()["method_lock"].exists():
        return
    assert_confirm_inference_allowed()


def write_confirm_seal() -> None:
    p = stage6_paths()["confirm_seal"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if not stage6_paths()["method_lock"].exists():
        p.write_text("SEALED\n")
