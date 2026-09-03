"""Phase-tagged non-finite errors with live traceback. Never string-only RuntimeError."""

from __future__ import annotations

import traceback
from typing import Any

import torch

PHASES = (
    "train_input",
    "train_forward",
    "train_loss",
    "backward",
    "optimizer_step",
    "eval_input",
    "eval_forward",
    "eval_loss",
    "metric",
    "checkpoint",
)


def live_traceback() -> str:
    tb = traceback.format_exc()
    if tb and tb.strip() and tb.strip() != "NoneType: None":
        return tb
    return "".join(traceback.format_stack())


def autocast_snapshot() -> dict[str, Any]:
    enabled = False
    dtype: str | None = None
    try:
        enabled = bool(torch.is_autocast_enabled("cuda"))
        dtype = str(torch.get_autocast_dtype("cuda"))
    except TypeError:
        enabled = bool(torch.is_autocast_enabled())
        try:
            dtype = str(torch.get_autocast_gpu_dtype())
        except Exception:
            dtype = None
    except Exception:
        enabled = bool(torch.is_autocast_enabled())
        dtype = None
    return {"torch.is_autocast_enabled": enabled, "autocast_dtype": dtype}


def _raw(model):
    if model is None:
        return None
    return model.module if hasattr(model, "module") else model


def batch_brief(batch: dict | None) -> dict[str, Any] | None:
    if not batch:
        return None
    x = batch.get("x")
    names = batch.get("trace_name")
    out: dict[str, Any] = {
        "batch_size": int(x.shape[0]) if torch.is_tensor(x) else None,
        "trace_names": [str(n) for n in list(names)[:32]] if names is not None else None,
        "crop_kinds": [str(k) for k in list(batch.get("crop_kind") or [])[:32]],
    }
    if torch.is_tensor(x):
        out["x_finite"] = bool(torch.isfinite(x).all())
        xf = x.detach().float()
        out["x_min"] = float(xf.min().cpu()) if xf.numel() else None
        out["x_max"] = float(xf.max().cpu()) if xf.numel() else None
    return out


class NonfiniteError(RuntimeError):
    """Non-finite value with an explicit lifecycle phase and live traceback."""

    def __init__(
        self,
        phase: str,
        message: str,
        *,
        model=None,
        batch: dict | None = None,
        device=None,
        rank: int | None = None,
        optimizer_step: int | None = None,
        first_nonfinite_module: dict | str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        raw = _raw(model)
        self.phase = phase
        self.context: dict[str, Any] = {
            "phase": phase,
            "message": message,
            "model.training": bool(raw.training) if raw is not None else None,
            **autocast_snapshot(),
            "device": str(device) if device is not None else None,
            "rank": rank,
            "optimizer_step": optimizer_step,
            "batch": batch_brief(batch),
            "first_nonfinite_module": first_nonfinite_module,
            "traceback": live_traceback(),
            "confirm_read": False,
        }
        if extra:
            self.context.update(extra)
        super().__init__(f"[{phase}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.context)


def check_finite(t: torch.Tensor | None, phase: str, message: str, **kwargs) -> None:
    if t is None:
        return
    if not torch.is_tensor(t):
        return
    if not torch.isfinite(t).all():
        raise NonfiniteError(phase, message, **kwargs)
