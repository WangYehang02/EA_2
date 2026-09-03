"""Full-FP32 train/eval path. No autocast, no GradScaler, no half/bfloat16 modules."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from earthquake.stage10.dkpn_clean import dkpn_logits
from earthquake.stage10.nonfinite import NonfiniteError, check_finite
from earthquake.stage10.partial_label import partial_label_nll, partial_nll_numerator
from earthquake.stage10.state_finite import _raw
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr, window_prob_stats


class PrecisionMismatch(RuntimeError):
    """Train/eval left FP32. Caller must write TRAIN_BLOCKED_PRECISION_MISMATCH and stop."""


def autocast_is_enabled() -> bool:
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def assert_no_autocast() -> None:
    if autocast_is_enabled():
        raise PrecisionMismatch("torch.autocast is enabled; v4 requires precision_mode=fp32")


def assert_fp32_tensor(t: torch.Tensor, what: str) -> None:
    if not torch.is_tensor(t):
        return
    if t.dtype == torch.float16 or t.dtype == torch.bfloat16:
        raise PrecisionMismatch(f"{what} dtype={t.dtype} (want float32)")
    if t.is_floating_point() and t.dtype != torch.float32:
        raise PrecisionMismatch(f"{what} dtype={t.dtype} (want float32)")


def assert_params_fp32(model: torch.nn.Module) -> None:
    raw = _raw(model)
    assert raw is not None
    for n, p in raw.named_parameters():
        if p.is_floating_point() and p.dtype != torch.float32:
            raise PrecisionMismatch(f"parameter {n} dtype={p.dtype}")


def state_dict_tensor_sha256(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if not torch.is_tensor(v):
            continue
        h.update(k.encode("utf-8"))
        t = v.detach().contiguous().cpu()
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class Fp32ActivationGuard:
    """Forward hooks. Cheap no-op unless enabled. Flags any fp16/bf16 activation."""

    def __init__(self) -> None:
        self.enabled = False
        self.bad: list[dict[str, Any]] = []
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
                if not torch.is_tensor(t) or not t.is_floating_point():
                    return
                if t.dtype in (torch.float16, torch.bfloat16):
                    self.bad.append({"module": name or type(_mod).__name__, "dtype": str(t.dtype), "shape": list(t.shape)})
                    return
                tf = t.detach().reshape(-1).float()
                finite = tf[torch.isfinite(tf)]
                rec: dict[str, Any] = {
                    "dtype": str(t.dtype),
                    "n_nonfinite": int((~torch.isfinite(tf)).sum().cpu()) if tf.numel() else 0,
                    "max_abs": None,
                    "p99_abs": None,
                    "p999_abs": None,
                }
                if finite.numel():
                    absf = finite.abs()
                    rec["max_abs"] = float(absf.max().cpu())
                    if absf.numel() >= 8:
                        q = torch.quantile(absf, torch.tensor([0.99, 0.999], device=absf.device, dtype=absf.dtype))
                        rec["p99_abs"] = float(q[0].cpu())
                        rec["p999_abs"] = float(q[1].cpu())
                    else:
                        rec["p99_abs"] = rec["max_abs"]
                        rec["p999_abs"] = rec["max_abs"]
                self.by_module[name or type(_mod).__name__] = rec

            return hook

        for n, m in raw.named_modules():
            self._handles.append(m.register_forward_hook(make_hook(n)))

    def begin(self) -> None:
        self.bad = []
        self.by_module = {}
        self.enabled = True

    def end(self) -> dict[str, Any]:
        self.enabled = False
        worst = None
        worst_abs = -1.0
        for n, s in self.by_module.items():
            a = s.get("max_abs")
            if a is not None and float(a) > worst_abs:
                worst_abs = float(a)
                worst = n
        return {
            "bad": list(self.bad),
            "n_modules": len(self.by_module),
            "worst_module": worst,
            "worst_abs_max": None if worst is None else worst_abs,
            "worst_p99_abs": (self.by_module.get(worst) or {}).get("p99_abs") if worst else None,
            "worst_p999_abs": (self.by_module.get(worst) or {}).get("p999_abs") if worst else None,
        }

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.enabled = False


def _kw(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}


def batch_loss_fp32(model, batch, device, *, optimizer_step=None, guard: Fp32ActivationGuard | None = None):
    """Train forward+NLL in FP32. Never enters autocast."""
    assert_no_autocast()
    assert_params_fp32(model)
    from earthquake.stage10.amp_forward import assert_input_finite

    assert_input_finite(batch, phase="train_input", model=model, device=device, optimizer_step=optimizer_step)
    x = batch["x"].to(device, non_blocking=True).float()
    assert_fp32_tensor(x, "train_input")
    kw = _kw(batch, device)
    if guard is not None:
        guard.begin()
    try:
        logits = dkpn_logits(model, x)
    finally:
        gsum = guard.end() if guard is not None else None
    if guard is not None and gsum and gsum["bad"]:
        raise PrecisionMismatch(f"non-fp32 activations: {gsum['bad'][:4]}")
    assert_fp32_tensor(logits, "logits")
    if not torch.isfinite(logits).all():
        raise NonfiniteError(
            "train_forward",
            "non-finite train logits",
            model=model,
            batch=batch,
            device=device,
            optimizer_step=optimizer_step,
        )
    loss = partial_label_nll(logits, **kw)
    assert_fp32_tensor(loss, "loss")
    check_finite(
        loss,
        "train_loss",
        "non-finite train loss",
        model=model,
        batch=batch,
        device=device,
        optimizer_step=optimizer_step,
    )
    return loss, logits, gsum


def loss_terms_fp32(logits, batch, device) -> dict[str, float]:
    kw = _kw(batch, device)
    _, _, terms = partial_nll_numerator(logits.float(), **kw)
    return {k: float(v.detach().cpu()) for k, v in terms.items()}


@torch.no_grad()
def eval_dev_plain_fp32(model, loader, device, *, pick_metrics=pick_metrics_at_thr, psn_dist, collapse_flag, height: float = OFFICIAL_HEIGHT) -> dict:
    """Eval without touching torch.autocast."""
    assert_no_autocast()
    model.eval()
    s_probs, p_probs, n_probs = [], [], []
    true_s, true_p, vis_s, vis_p = [], [], [], []
    vloss = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        assert_fp32_tensor(x, "eval_input")
        kw = _kw(batch, device)
        logits = dkpn_logits(model, x)
        assert_fp32_tensor(logits, "eval_logits")
        if not torch.isfinite(logits).all():
            raise NonfiniteError("eval_forward", "non-finite eval logits", model=model, device=device)
        loss = partial_label_nll(logits, **kw)
        assert_fp32_tensor(loss, "eval_loss")
        check_finite(loss, "eval_loss", "non-finite eval loss", model=model, device=device)
        vloss.append(float(loss.detach().cpu()))
        pr = torch.softmax(logits, dim=1).cpu().numpy()
        for i in range(pr.shape[0]):
            s_probs.append(pr[i, 1])
            p_probs.append(pr[i, 0])
            n_probs.append(pr[i, 2])
            true_s.append(float(batch["s_c"][i]))
            true_p.append(float(batch["p_c"][i]))
            vis_s.append(bool(batch["vis_s"][i] > 0.5 and float(batch["s_c"][i]) >= 0))
            vis_p.append(bool(batch["vis_p"][i] > 0.5 and float(batch["p_c"][i]) >= 0))
    s_arr = np.stack(s_probs) if s_probs else np.zeros((0, 1))
    p_arr = np.stack(p_probs) if p_probs else np.zeros((0, 1))
    n_arr = np.stack(n_probs) if n_probs else np.zeros((0, 1))
    ts = np.asarray(true_s, dtype=float)
    tp = np.asarray(true_p, dtype=float)
    vs = np.asarray(vis_s, dtype=bool)
    vp = np.asarray(vis_p, dtype=bool)
    m = pick_metrics(s_arr, ts, vs, height)
    wstats = window_prob_stats(s_arr, p_arr, ts, tp, vs, vp)
    dist = psn_dist({"s_probs": s_arr, "p_probs": p_arr, "n_probs": n_arr})
    return {
        "val_loss": float(np.mean(vloss) if vloss else float("nan")),
        "f1": m["f1@0.5s"],
        "height": height,
        "metrics_height_0p2": m,
        "psn": dist,
        **wstats,
        "all_n_collapse": collapse_flag(dist),
        "never_used_calibrated_threshold": True,
        "eval_dtype": "fp32",
        "eval_autocast": False,
    }
