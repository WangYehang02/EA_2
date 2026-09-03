"""Official eval/inference: model.eval(), autocast off, FP32 logits/loss/metrics.

Train may use AMP FP16 forward. Eval must not: large BN running_var overflows FP16.
BF16 eval is diagnostic only, not the official method.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from earthquake.stage10.dkpn_clean import dkpn_logits
from earthquake.stage10.nonfinite import NonfiniteError, autocast_snapshot, check_finite
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr, window_prob_stats


def _ac_off(device):
    if device is not None and getattr(device, "type", None) == "cuda":
        return torch.autocast("cuda", enabled=False)
    return nullcontext()


def eval_forward_fp32(model, x: torch.Tensor, device) -> torch.Tensor:
    """Official eval forward. Caller should model.eval(); this also forces eval()."""
    model.eval()
    x = x.to(device, non_blocking=True).float()
    check_finite(x, "eval_input", "non-finite eval input", model=model, device=device, batch={"x": x})
    with torch.no_grad(), _ac_off(device):
        logits = dkpn_logits(model, x)
    check_finite(logits, "eval_forward", "non-finite eval logits", model=model, device=device)
    return logits.float()


def eval_forward_amp_fp16_diagnostic(model, x: torch.Tensor, device) -> torch.Tensor:
    """Diagnostic contrast only. Not used for metrics or checkpoint selection."""
    if device.type != "cuda":
        raise RuntimeError("FP16 eval diagnostic requires CUDA")
    from earthquake.stage10.finite_hooks import install_finite_hooks, remove_hooks

    model.eval()
    x = x.to(device, non_blocking=True)
    check_finite(x, "eval_input", "non-finite eval input", model=model, device=device, batch={"x": x})
    amp_during = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = dkpn_logits(model, x)
        amp_during = autocast_snapshot()
    if not torch.isfinite(logits).all():
        first, handles = install_finite_hooks(model.module if hasattr(model, "module") else model)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                dkpn_logits(model, x)
        finally:
            remove_hooks(handles)
        extra = {"diagnostic": True, "official_eval": "fp32"}
        if amp_during:
            extra.update(amp_during)
        raise NonfiniteError(
            "eval_forward",
            "non-finite eval logits under AMP FP16 (diagnostic)",
            model=model,
            device=device,
            first_nonfinite_module=dict(first) if first else None,
            extra=extra,
        )
    return logits


def eval_loss_fp32(logits: torch.Tensor, kw: dict, *, model=None, device=None) -> torch.Tensor:
    with _ac_off(device if device is not None else logits.device):
        loss = partial_label_nll(logits.float(), **kw)
    check_finite(loss, "eval_loss", "non-finite eval loss", model=model, device=device)
    return loss


def probs_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Softmax for extract_picks / height=0.2 metrics. Always FP32."""
    with torch.autocast("cuda", enabled=False) if logits.is_cuda else nullcontext():
        p = torch.softmax(logits.float(), dim=1)
    check_finite(p, "metric", "non-finite softmax probabilities")
    return p


def eval_dev_fp32(model, loader, device, *, pick_metrics, psn_dist, collapse_flag, height: float = OFFICIAL_HEIGHT) -> dict:
    """S-centered (or whatever the loader already cropped) validation. Official FP32 path."""
    model.eval()
    s_probs, p_probs, n_probs = [], [], []
    true_s, true_p, vis_s, vis_p = [], [], [], []
    vloss = []
    for batch in loader:
        x = batch["x"]
        kw = {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
        logits = eval_forward_fp32(model, x, device)
        loss = eval_loss_fp32(logits, kw, model=model, device=device)
        vloss.append(float(loss.detach().cpu()))
        pr = probs_fp32(logits).cpu().numpy()
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
