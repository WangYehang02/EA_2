"""Train-step monitors: per-module activation stats and real divergence (not just finite)."""

from __future__ import annotations

import math
from typing import Any

import torch

from earthquake.stage10.state_finite import _raw, audit_bn_buffers, audit_optimizer_state, audit_parameters


def _abs_quantiles(t: torch.Tensor, qs: tuple[float, ...] = (0.99, 0.999)) -> dict[str, float | None]:
    tf = t.detach().reshape(-1).float()
    finite = tf[torch.isfinite(tf)]
    out: dict[str, float | None] = {"max": None, "max_abs": None, "p99_abs": None, "p999_abs": None, "n_nonfinite": int((~torch.isfinite(tf)).sum().cpu()) if tf.numel() else 0}
    if finite.numel() == 0:
        return out
    out["max"] = float(finite.max().cpu())
    absf = finite.abs()
    out["max_abs"] = float(absf.max().cpu())
    if absf.numel() < 8:
        out["p99_abs"] = out["max_abs"]
        out["p999_abs"] = out["max_abs"]
        return out
    q = torch.quantile(absf, torch.tensor(qs, device=absf.device, dtype=absf.dtype))
    out["p99_abs"] = float(q[0].cpu())
    out["p999_abs"] = float(q[1].cpu())
    return out


class ActivationMonitor:
    """Forward hooks. Cheap no-op unless ``enabled``; capture during the real train forward."""

    def __init__(self) -> None:
        self.enabled = False
        self.by_module: dict[str, dict[str, Any]] = {}
        self._handles: list = []

    def attach(self, model: torch.nn.Module) -> None:
        self.remove()
        raw = _raw(model)
        assert raw is not None

        def make_hook(name: str):
            def hook(_mod, _inp, out):
                if not self.enabled:
                    return
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(t) or t.numel() == 0:
                    return
                self.by_module[name or type(_mod).__name__] = _abs_quantiles(t)

            return hook

        for n, m in raw.named_modules():
            self._handles.append(m.register_forward_hook(make_hook(n)))

    def begin(self) -> None:
        self.by_module = {}
        self.enabled = True

    def end(self) -> dict[str, dict[str, Any]]:
        self.enabled = False
        return dict(self.by_module)

    def summary(self, stats: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        stats = self.by_module if stats is None else stats
        worst_name = None
        worst_abs = -1.0
        n_nonfinite_mod = 0
        for n, s in stats.items():
            if int(s.get("n_nonfinite") or 0) > 0:
                n_nonfinite_mod += 1
            a = s.get("max_abs")
            if a is not None and float(a) > worst_abs:
                worst_abs = float(a)
                worst_name = n
        return {
            "n_modules": len(stats),
            "n_modules_nonfinite": n_nonfinite_mod,
            "worst_module": worst_name,
            "worst_abs_max": None if worst_name is None else worst_abs,
            "by_module": stats,
        }

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.enabled = False


def psn_means(logits: torch.Tensor) -> dict[str, float]:
    pr = torch.softmax(logits.detach().float(), dim=1)
    return {
        "p_mean": float(pr[:, 0].mean().cpu()),
        "s_mean": float(pr[:, 1].mean().cpu()),
        "n_mean": float(pr[:, 2].mean().cpu()),
        "logits_max": float(logits.detach().float().max().cpu()) if torch.isfinite(logits).any() else None,
        "logits_max_abs": float(logits.detach().float().abs().max().cpu()) if logits.numel() else None,
        "logits_finite": bool(torch.isfinite(logits).all()),
    }


def _log_slope_per_point(values: list[float]) -> float | None:
    ys = [math.log(v) for v in values if v is not None and v > 0 and math.isfinite(v)]
    n = len(ys)
    if n < 5:
        return None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / den


def divergence_report(series: list[dict[str, Any]], *, interval_steps: int = 100) -> dict[str, Any]:
    """Flag real exponential growth, not merely NaN/Inf."""

    def col(key: str) -> list[float]:
        out = []
        for row in series:
            v = row.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out.append(fv)
        return out

    act = col("act_abs_max")
    param = col("param_abs_max")
    loss = col("loss")
    flags: list[str] = []

    def exp_flag(name: str, vals: list[float], grow: float = 1.5) -> bool:
        slope = _log_slope_per_point(vals[-8:] if len(vals) >= 8 else vals)
        if slope is None:
            return False
        # slope is per monitor point (= interval_steps)
        ratio = math.exp(slope)
        if ratio >= grow:
            flags.append(f"exponential_{name}_ratio_{ratio:.3f}_per_{interval_steps}_steps")
            return True
        return False

    exp_act = exp_flag("activation", act)
    exp_param = exp_flag("parameter", param)
    exp_loss = exp_flag("loss", loss, grow=1.4)
    hard = []
    if act and act[-1] > 1e6:
        hard.append("act_abs_max_gt_1e6")
    if param and param[-1] > 1e3:
        hard.append("param_abs_max_gt_1e3")
    if loss and loss[0] > 0 and loss[-1] > max(2.0, 50.0 * loss[0]):
        hard.append("loss_exploded_vs_start")
    if act and act[0] > 0 and act[-1] > 100.0 * act[0]:
        hard.append("act_grew_100x")
    flags.extend(hard)
    finite = all(bool(r.get("finite", True)) for r in series) if series else False
    unstable = bool(flags)
    return {
        "finite_throughout": finite,
        "exponential_activation": exp_act,
        "exponential_parameter": exp_param,
        "exponential_loss": exp_loss,
        "hard_flags": hard,
        "flags": flags,
        "stable": bool(finite and not unstable),
        "n_monitor_points": len(series),
        "last_act_abs_max": act[-1] if act else None,
        "last_param_abs_max": param[-1] if param else None,
        "last_loss": loss[-1] if loss else None,
        "first_loss": loss[0] if loss else None,
    }


def snapshot_norms(model: torch.nn.Module, opt: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    p = audit_parameters(model)
    bn = audit_bn_buffers(model)
    out: dict[str, Any] = {
        "param_abs_max": p["abs_max"],
        "param_l2_norm": p["l2_norm"],
        "param_finite": p["finite"],
        "bn_running_var_max": bn["running_var_max"],
        "bn_finite": bn["finite"],
    }
    if opt is not None:
        o = audit_optimizer_state(opt)
        out["adam_abs_max"] = o["abs_max"]
        out["adam_finite"] = o["finite"]
        out["lr"] = float(opt.param_groups[0]["lr"]) if opt.param_groups else None
    return out
