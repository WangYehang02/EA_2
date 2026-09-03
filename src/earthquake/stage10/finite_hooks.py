"""Forward hooks to record the first module that emits non-finite activations."""

from __future__ import annotations

from typing import Any

import torch


def install_finite_hooks(model: torch.nn.Module) -> tuple[dict[str, Any], list]:
    first: dict[str, Any] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            if first.get("module"):
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            if t.numel() == 0:
                return
            if not torch.isfinite(t).all():
                tf = t.detach().float()
                finite = torch.isfinite(tf)
                first["module"] = name or type(_mod).__name__
                first["shape"] = list(t.shape)
                first["dtype"] = str(t.dtype)
                first["n_nan"] = int((~torch.isfinite(tf) & torch.isnan(tf)).sum().cpu()) if True else 0
                first["n_nonfinite"] = int((~finite).sum().cpu())
                if finite.any():
                    first["finite_min"] = float(tf[finite].min().cpu())
                    first["finite_max"] = float(tf[finite].max().cpu())
                else:
                    first["finite_min"] = None
                    first["finite_max"] = None

        return hook

    for n, m in model.named_modules():
        handles.append(m.register_forward_hook(make_hook(n)))
    return first, handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
