"""Finite audits for parameters, BN buffers, optimizer state, and gradients.

Forensic only. Never used as a silent skip.
"""

from __future__ import annotations

from typing import Any

import torch


def _raw(model: torch.nn.Module | None) -> torch.nn.Module | None:
    if model is None:
        return None
    return model.module if hasattr(model, "module") else model


def tensor_report(t: torch.Tensor, *, name: str | None = None) -> dict[str, Any]:
    tf = t.detach().float()
    finite = bool(torch.isfinite(tf).all())
    out: dict[str, Any] = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "finite": finite,
        "numel": int(t.numel()),
    }
    if tf.numel() == 0:
        return out
    absf = tf.abs()
    out["abs_max"] = float(absf.max().cpu()) if finite or absf.numel() else None
    if finite:
        out["min"] = float(tf.min().cpu())
        out["max"] = float(tf.max().cpu())
        out["n_nonfinite"] = 0
    else:
        mask = torch.isfinite(tf)
        out["n_nonfinite"] = int((~mask).sum().cpu())
        out["n_nan"] = int(torch.isnan(tf).sum().cpu())
        out["n_inf"] = int(torch.isinf(tf).sum().cpu())
        if mask.any():
            out["finite_min"] = float(tf[mask].min().cpu())
            out["finite_max"] = float(tf[mask].max().cpu())
            out["abs_max"] = float(tf[mask].abs().max().cpu())
        else:
            out["finite_min"] = None
            out["finite_max"] = None
            out["abs_max"] = None
    return out


def audit_parameters(model: torch.nn.Module) -> dict[str, Any]:
    raw = _raw(model)
    assert raw is not None
    n_nonfinite = 0
    abs_max = 0.0
    n_params = 0
    worst = None
    sq = 0.0
    for n, p in raw.named_parameters():
        n_params += 1
        tf = p.detach().float()
        sq += float(tf.pow(2).sum().cpu())
        if not torch.isfinite(tf).all():
            n_nonfinite += 1
            if worst is None:
                worst = tensor_report(p, name=n)
        else:
            abs_max = max(abs_max, float(tf.abs().max().cpu()))
    return {
        "finite": n_nonfinite == 0,
        "n_tensors": n_params,
        "n_nonfinite_tensors": n_nonfinite,
        "abs_max": abs_max if n_nonfinite == 0 else (worst or {}).get("abs_max"),
        "l2_norm": sq ** 0.5,
        "worst": worst,
    }


def audit_bn_buffers(model: torch.nn.Module) -> dict[str, Any]:
    raw = _raw(model)
    assert raw is not None
    n_nonfinite = 0
    var_max = 0.0
    mean_abs_max = 0.0
    worst = None
    n_var = 0
    fp16_max = 65504.0
    n_var_gt_fp16 = 0
    for n, b in raw.named_buffers():
        if not torch.is_tensor(b) or b.numel() == 0:
            continue
        tf = b.detach().float()
        ok = bool(torch.isfinite(tf).all())
        if "running_var" in n:
            n_var += 1
            if not ok:
                n_nonfinite += 1
                if worst is None:
                    worst = tensor_report(b, name=n)
            else:
                vm = float(tf.max().cpu())
                var_max = max(var_max, vm)
                if vm > fp16_max:
                    n_var_gt_fp16 += 1
        elif "running_mean" in n:
            if not ok:
                n_nonfinite += 1
                if worst is None:
                    worst = tensor_report(b, name=n)
            else:
                mean_abs_max = max(mean_abs_max, float(tf.abs().max().cpu()))
        elif not ok:
            n_nonfinite += 1
            if worst is None:
                worst = tensor_report(b, name=n)
    return {
        "finite": n_nonfinite == 0,
        "n_running_var": n_var,
        "running_var_max": var_max,
        "running_mean_abs_max": mean_abs_max,
        "n_running_var_gt_fp16_max": n_var_gt_fp16,
        "n_nonfinite_tensors": n_nonfinite,
        "worst": worst,
    }


def audit_optimizer_state(opt: torch.optim.Optimizer) -> dict[str, Any]:
    n_nonfinite = 0
    abs_max = 0.0
    n_tensors = 0
    worst = None
    for st in opt.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            if not torch.is_tensor(v):
                continue
            n_tensors += 1
            tf = v.detach().float()
            if not torch.isfinite(tf).all():
                n_nonfinite += 1
                if worst is None:
                    worst = tensor_report(v, name=str(k))
            elif tf.numel():
                abs_max = max(abs_max, float(tf.abs().max().cpu()))
    return {
        "finite": n_nonfinite == 0,
        "n_tensors": n_tensors,
        "n_nonfinite_tensors": n_nonfinite,
        "abs_max": abs_max if n_nonfinite == 0 else (worst or {}).get("abs_max"),
        "worst": worst,
    }


def audit_gradients(model: torch.nn.Module) -> dict[str, Any]:
    raw = _raw(model)
    assert raw is not None
    n_with = 0
    n_none = 0
    n_nonfinite = 0
    abs_max = 0.0
    sq = 0.0
    worst = None
    for n, p in raw.named_parameters():
        if p.grad is None:
            n_none += 1
            continue
        n_with += 1
        tf = p.grad.detach().float()
        if not torch.isfinite(tf).all():
            n_nonfinite += 1
            if worst is None:
                worst = tensor_report(p.grad, name=n)
            continue
        if tf.numel():
            abs_max = max(abs_max, float(tf.abs().max().cpu()))
            sq += float(tf.pow(2).sum().cpu())
    return {
        "n_with_grad": n_with,
        "n_none": n_none,
        "gradients_present": n_with > 0,
        "gradients_all_none": n_with == 0,
        "finite": n_nonfinite == 0,
        "n_nonfinite_tensors": n_nonfinite,
        "abs_max": abs_max if n_with and n_nonfinite == 0 else None,
        "l2_norm": (sq ** 0.5) if n_with and n_nonfinite == 0 else None,
        "worst": worst,
    }


def audit_model_opt(model: torch.nn.Module, opt: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "parameters": audit_parameters(model),
        "bn_buffers": audit_bn_buffers(model),
        "gradients": audit_gradients(model),
    }
    if opt is not None:
        out["optimizer"] = audit_optimizer_state(opt)
        out["lr"] = float(opt.param_groups[0]["lr"]) if opt.param_groups else None
    out["all_finite"] = bool(
        out["parameters"]["finite"]
        and out["bn_buffers"]["finite"]
        and out["gradients"]["finite"]
        and (out["optimizer"]["finite"] if opt is not None else True)
    )
    return out
