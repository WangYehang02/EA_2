"""AMP forward + FP32 partial-label loss (autocast does not wrap the NLL)."""

from __future__ import annotations

import torch

from earthquake.stage10.dkpn_clean import dkpn_logits
from earthquake.stage10.finite_hooks import install_finite_hooks, remove_hooks
from earthquake.stage10.nonfinite import NonfiniteError, check_finite
from earthquake.stage10.partial_label import partial_label_nll, partial_label_nll_legacy_clamped, token_weights


def _kw(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}


def assert_input_finite(batch, *, phase: str = "train_input", model=None, device=None, optimizer_step=None) -> None:
    x = batch["x"]
    names = batch.get("trace_name") or ["?"]
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=-1)).nonzero(as_tuple=False).view(-1).tolist()
        ids = [str(names[i]) for i in bad]
        raise NonfiniteError(phase, f"NaN/Inf input traces={ids}", model=model, batch=batch, device=device, optimizer_step=optimizer_step)
    for k in ("p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"):
        if k in batch and torch.is_tensor(batch[k]) and not torch.isfinite(batch[k]).all():
            t = batch[k]
            bad = (~torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=-1)).nonzero(as_tuple=False).view(-1).tolist()
            ids = [str(names[i]) for i in bad]
            raise NonfiniteError(phase, f"NaN/Inf {k} traces={ids}", model=model, batch=batch, device=device, optimizer_step=optimizer_step)


def _first_nonfinite_on_replay(model, x, amp_on, ac_dtype):
    raw = model.module if hasattr(model, "module") else model
    first, handles = install_finite_hooks(raw)
    try:
        if amp_on:
            with torch.autocast("cuda", dtype=ac_dtype):
                dkpn_logits(model, x)
        else:
            dkpn_logits(model, x)
        return dict(first) if first else None
    finally:
        remove_hooks(handles)


def batch_loss_amp_forward_fp32_nll(model, batch, device, *, amp: bool, dtype: torch.dtype | None = None, optimizer_step=None):
    """Forward under autocast(dtype); loss always FP32 with autocast off."""
    assert_input_finite(batch, phase="train_input", model=model, device=device, optimizer_step=optimizer_step)
    x = batch["x"].to(device, non_blocking=True)
    kw = _kw(batch, device)
    amp_on = bool(amp) and device.type == "cuda"
    ac_dtype = dtype or torch.float16
    if amp_on:
        with torch.autocast("cuda", dtype=ac_dtype):
            logits = dkpn_logits(model, x)
    else:
        logits = dkpn_logits(model, x)
    if not torch.isfinite(logits).all():
        first = _first_nonfinite_on_replay(model, x, amp_on, ac_dtype)
        raise NonfiniteError(
            "train_forward",
            "non-finite train logits",
            model=model,
            batch=batch,
            device=device,
            optimizer_step=optimizer_step,
            first_nonfinite_module=first,
        )
    with torch.autocast("cuda", enabled=False) if device.type == "cuda" else torch.autocast("cpu", enabled=False):
        loss = partial_label_nll(logits.float(), **kw)
    check_finite(
        loss,
        "train_loss",
        "non-finite train loss",
        model=model,
        batch=batch,
        device=device,
        optimizer_step=optimizer_step,
    )
    return loss, logits


def batch_loss_legacy_amp_path_a(model, batch, device, *, optimizer_step=None):
    """Failed training path: autocast wraps both forward and NLL (legacy clamp denom)."""
    assert_input_finite(batch, phase="train_input", model=model, device=device, optimizer_step=optimizer_step)
    x = batch["x"].to(device, non_blocking=True)
    kw = _kw(batch, device)
    amp_on = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=amp_on):
        logits = dkpn_logits(model, x)
        if not torch.isfinite(logits).all():
            first = _first_nonfinite_on_replay(model, x, amp_on, torch.float16)
            raise NonfiniteError(
                "train_forward",
                "non-finite train logits (legacy AMP path A)",
                model=model,
                batch=batch,
                device=device,
                optimizer_step=optimizer_step,
                first_nonfinite_module=first,
            )
        loss = partial_label_nll_legacy_clamped(logits, **kw)
    check_finite(
        loss,
        "train_loss",
        "non-finite train loss (legacy AMP path A)",
        model=model,
        batch=batch,
        device=device,
        optimizer_step=optimizer_step,
        extra={"legacy_amp_wraps_nll": True},
    )
    return loss, logits


def slice_batch(batch: dict, sl) -> dict:
    """Slice a collated batch (tensors + list fields) to a rank shard."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[sl]
        elif isinstance(v, (list, tuple)):
            out[k] = list(v[sl]) if isinstance(sl, slice) else [v[i] for i in sl]
        else:
            out[k] = v
    return out


def shard_valid_counts(batch, n_ranks: int) -> list[float]:
    w = token_weights(
        p_pos=batch["p_pos"],
        s_pos=batch["s_pos"],
        n_pos=batch["n_pos"],
        not_p=batch["not_p"],
        not_s=batch["not_s"],
        pad_mask=batch["pad_mask"],
    )
    b = w.shape[0]
    bs = b // n_ranks
    return [float(w[r * bs : (r + 1) * bs].sum()) for r in range(n_ranks)]
