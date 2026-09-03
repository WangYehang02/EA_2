from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load .env from project root without overriding existing env vars."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_instance_root() -> Path:
    """Resolve INSTANCE data root without scanning the whole user home."""
    load_env()
    env = os.environ.get("INSTANCE_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_instance(p):
            return p
        raise FileNotFoundError(
            f"INSTANCE_ROOT={p} is set but does not look like an INSTANCE dataset root."
        )

    candidates = [
        PROJECT_ROOT / "data" / "INSTANCE",
        PROJECT_ROOT / "dataset" / "INSTANCE",
    ]
    for c in candidates:
        if _looks_like_instance(c):
            return c.resolve()

    example = PROJECT_ROOT / ".env.example"
    raise FileNotFoundError(
        "INSTANCE_ROOT is not configured and local fallbacks were not found.\n"
        f"Checked: {[str(c) for c in candidates]}\n"
        f"Please copy {example} to .env and set INSTANCE_ROOT to the INSTANCE root "
        "(containing events/, noise/, inventory/)."
    )


def _looks_like_instance(root: Path) -> bool:
    if not root.is_dir():
        return False
    events = root / "events"
    noise = root / "noise"
    return (
        events.is_dir()
        and noise.is_dir()
        and any(events.glob("metadata_Instance_events*.csv*"))
        and (
            (events / "Instance_events_counts.hdf5").exists()
            or any(events.glob("Instance_events_counts.hdf5*"))
        )
    )


def artifacts_dir() -> Path:
    load_env()
    d = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
