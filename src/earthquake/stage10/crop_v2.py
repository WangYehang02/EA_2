"""30 s DKPN crop geometry and phase-visibility helpers (100 Hz)."""

from __future__ import annotations

import hashlib

import numpy as np

IN_SAMPLES = 3001
FP_STAB_S = 4.0
SR = 100.0
FSTAB = int(SR * FP_STAB_S)  # 400
WIN_RAW = IN_SAMPLES + FSTAB  # 3401  (~34.01 s raw, 30.01 s label window)
CROP_CYCLE = ("p_centered", "s_centered", "background")
LONG_PS_S = 30.0



def label_coord(phase_sample: float | None, start: int, fstab: int = FSTAB) -> float | None:
    if phase_sample is None or not np.isfinite(phase_sample):
        return None
    return float(phase_sample) - float(start) - float(fstab)


def phase_visible(phase_sample: float | None, start: int, *, in_samples: int = IN_SAMPLES, fstab: int = FSTAB) -> bool:
    c = label_coord(phase_sample, start, fstab)
    if c is None:
        return False
    return 0.0 <= c < float(in_samples)


def centered_start(center: float, n: int, win_raw: int = WIN_RAW, fstab: int = FSTAB, in_samples: int = IN_SAMPLES) -> int:
    """Start so `center` maps near the middle of the post-fstab label window."""
    max_start = max(n - win_raw, 0)
    # label index = center - start - fstab ≈ in_samples/2
    start = int(round(float(center) - fstab - in_samples / 2.0))
    return int(np.clip(start, 0, max_start))


def background_start(
    n: int,
    p: float | None,
    s: float | None,
    rng: np.random.Generator,
    win_raw: int = WIN_RAW,
    fstab: int = FSTAB,
    in_samples: int = IN_SAMPLES,
    tries: int = 16,
) -> int:
    """Prefer a crop where neither labeled phase is in the label window."""
    max_start = max(n - win_raw, 0)
    if max_start <= 0:
        return 0
    for _ in range(tries):
        st = int(rng.integers(0, max_start + 1))
        pv = phase_visible(p, st, in_samples=in_samples, fstab=fstab)
        sv = phase_visible(s, st, in_samples=in_samples, fstab=fstab)
        if not pv and not sv:
            return st
    return int(rng.integers(0, max_start + 1))


def crop_kind_start(
    kind: str,
    n: int,
    p: float | None,
    s: float | None,
    rng: np.random.Generator,
) -> int:
    if kind == "p_centered":
        if p is None or not np.isfinite(p):
            return background_start(n, p, s, rng)
        return centered_start(float(p), n)
    if kind == "s_centered":
        if s is None or not np.isfinite(s):
            return background_start(n, p, s, rng)
        return centered_start(float(s), n)
    if kind in {"background", "noise"}:
        return background_start(n, p, s, rng) if kind == "background" else int(rng.integers(0, max(n - WIN_RAW, 0) + 1))
    raise ValueError(kind)


def gaussian_in_window(
    n: int,
    center: float | None,
    sigma: float,
    pad_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Truncated Gaussian; renormalize peak to 1 if any mass in window. Zeros if invisible."""
    g = np.zeros(n, dtype=np.float32)
    if center is None or not np.isfinite(center):
        return g
    t = np.arange(n, dtype=np.float64)
    sig = max(float(sigma), 1.0)
    g = np.exp(-0.5 * ((t - float(center)) / sig) ** 2).astype(np.float32)
    if pad_mask is not None:
        g = g * pad_mask.astype(np.float32)
    m = float(g.max())
    if m < 1e-8:
        return np.zeros(n, dtype=np.float32)
    g = g / m
    return g


def visibility_bucket(vis_p: bool, vis_s: bool) -> str:
    if vis_p and vis_s:
        return "ps_both"
    if vis_p and not vis_s:
        return "p_only"
    if vis_s and not vis_p:
        return "s_only"
    return "neither"


def trace_schedule_hash(trace_name: str) -> int:
    """Stable non-Python-hash (md5) so scheduling is reproducible across processes."""
    return int(hashlib.md5(str(trace_name).encode("utf-8")).hexdigest()[:8], 16)


def scheduled_crop_kind(
    *,
    virtual_epoch: int,
    trace_name: str,
    is_noise: bool,
    has_p_label: bool,
    has_s_label: bool,
) -> str:
    """One crop per trace per virtual epoch. 3 virtual epochs = one P/S/background cycle.

    Partial-label semantics are unchanged: kind only selects the window, not the loss.
    """
    if is_noise:
        return "noise"
    h = trace_schedule_hash(trace_name)
    slot = (int(h) + int(virtual_epoch)) % 3
    if has_p_label and has_s_label:
        # mix within each virtual epoch: crop_type = (hash(trace) + epoch) % 3
        return CROP_CYCLE[slot]
    if has_p_label and not has_s_label:
        return "p_centered" if (slot % 2 == 0) else "background"
    if has_s_label and not has_p_label:
        return "s_centered" if (slot % 2 == 0) else "background"
    return "background"


def long_ps_sample_weight(p_sample: float | None, s_sample: float | None, sr: float = 100.0) -> int:
    """Upsample traces with P–S > 30 s so both phase-centered crops appear often enough."""
    if p_sample is None or s_sample is None:
        return 1
    if not (np.isfinite(p_sample) and np.isfinite(s_sample)):
        return 1
    if (float(s_sample) - float(p_sample)) / sr > LONG_PS_S:
        return 2
    return 1


def kinds_over_cycle(trace_name: str, has_p: bool, has_s: bool, is_noise: bool = False) -> list[str]:
    return [scheduled_crop_kind(virtual_epoch=e, trace_name=trace_name, is_noise=is_noise, has_p_label=has_p, has_s_label=has_s) for e in range(3)]

