"""v2 DKPN crops: P-centered / S-centered / background + partial-label weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from earthquake.models.phasenet_finetune import AugmentConfig
from earthquake.stage10.crop_v2 import (
    FSTAB,
    IN_SAMPLES,
    WIN_RAW,
    crop_kind_start,
    gaussian_in_window,
    label_coord,
    long_ps_sample_weight,
    phase_visible,
    scheduled_crop_kind,
    visibility_bucket,
)
from earthquake.stage10.dkpn_clean import compute_cf_5ch, ensure_dkpn_on_path, normalize_cf_window
from earthquake.stage10.protocol import enz_to_zne


def build_partial_weights(
    *,
    n: int,
    p_c: float | None,
    s_c: float | None,
    vis_p: bool,
    vis_s: bool,
    is_noise: bool,
    pad_mask: np.ndarray,
    sigma: float,
    has_p_label: bool,
    has_s_label: bool,
) -> dict[str, np.ndarray]:
    """Construct per-timestep partial-label weights. Phases outside crop never become N-negatives."""
    z = np.zeros(n, dtype=np.float32)
    pad = pad_mask.astype(np.float32)
    if is_noise:
        return {"p_pos": z, "s_pos": z, "n_pos": pad, "not_p": z, "not_s": z, "pad_mask": pad}

    gp = gaussian_in_window(n, p_c if vis_p else None, sigma, pad)
    gs = gaussian_in_window(n, s_c if vis_s else None, sigma, pad)

    p_pos = z.copy()
    s_pos = z.copy()
    n_pos = z.copy()
    not_p = z.copy()
    not_s = z.copy()

    if vis_p and vis_s:
        # complete P+S in window → simplex CE
        phase = np.clip(gp + gs, 0, 1)
        n_ch = np.clip(1.0 - phase, 0, 1) * pad
        stack = np.stack([gp, gs, n_ch], axis=0)
        stack = stack / np.maximum(stack.sum(axis=0, keepdims=True), 1e-8)
        p_pos, s_pos, n_pos = stack[0], stack[1], stack[2]
    elif vis_p and not vis_s:
        p_pos = gp
        # known not-P, S unknown (or S labeled but outside crop)
        not_p = pad * (1.0 - gp)
        # do NOT set n_pos
    elif vis_s and not vis_p:
        s_pos = gs
        not_s = pad * (1.0 - gs)
    else:
        # neither labeled phase is in this crop
        if has_p_label and has_s_label:
            # Event background: P and S exist on the trace but not in this window.
            # Do NOT treat unknown event context as certified noise / N supervision.
            pass
        elif has_p_label and not has_s_label:
            not_p = pad  # known not-P, S unknown
        elif has_s_label and not has_p_label:
            not_s = pad
        else:
            pass  # fully unknown: all zeros

    return {"p_pos": p_pos, "s_pos": s_pos, "n_pos": n_pos, "not_p": not_p, "not_s": not_s, "pad_mask": pad}


def annotate_online_catalog(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per trace. Crop kind is chosen online from virtual_epoch (no 3× expansion)."""
    out = meta.copy().reset_index(drop=True)
    if "crop_kind" in out.columns:
        out = out.drop(columns=["crop_kind"])
    has_p, has_s, weights = [], [], []
    for _, r in out.iterrows():
        is_noise = bool(r.get("is_noise", False))
        p = r.get("p_arrival_sample", np.nan)
        s = r.get("s_arrival_sample", np.nan)
        hp = (not is_noise) and pd.notna(p) and np.isfinite(float(p))
        hs = (not is_noise) and pd.notna(s) and np.isfinite(float(s))
        has_p.append(bool(hp))
        has_s.append(bool(hs))
        weights.append(long_ps_sample_weight(float(p) if hp else None, float(s) if hs else None))
    out["has_p_label"] = has_p
    out["has_s_label"] = has_s
    out["sample_weight"] = weights
    return out


def expand_crop_catalog(meta: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """One row per crop. Long P–S traces are kept (P-centered + S-centered + background)."""
    rows = []
    for _, r in meta.iterrows():
        is_noise = bool(r.get("is_noise", False))
        p = r.get("p_arrival_sample", np.nan)
        s = r.get("s_arrival_sample", np.nan)
        has_p = (not is_noise) and pd.notna(p) and np.isfinite(float(p))
        has_s = (not is_noise) and pd.notna(s) and np.isfinite(float(s))
        kinds: list[str]
        if is_noise:
            kinds = ["noise"]
        elif has_p and has_s:
            kinds = ["p_centered", "s_centered", "background"]
        elif has_p:
            kinds = ["p_centered", "background"]
        elif has_s:
            kinds = ["s_centered", "background"]
        else:
            continue
        for k in kinds:
            rec = dict(r)
            rec["crop_kind"] = k
            rec["has_p_label"] = bool(has_p)
            rec["has_s_label"] = bool(has_s)
            rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True)


class EventStationBalancedSampler(Sampler[int]):
    def __init__(self, catalog: pd.DataFrame, *, num_samples: int | None = None, seed: int = 0):
        self.cat = catalog.reset_index(drop=True)
        self.num_samples = int(num_samples if num_samples is not None else len(self.cat))
        self.seed = int(seed)
        from collections import defaultdict

        self.by_es: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_e: dict[str, set[str]] = defaultdict(set)
        self.row_w = np.ones(len(self.cat), dtype=np.float64)
        for i, row in self.cat.iterrows():
            e = str(row.get("event_id", "NA"))
            st = str(row.get("station_id", row.get("station", "NA")))
            self.by_es[(e, st)].append(int(i))
            self.by_e[e].add(st)
            w = row.get("sample_weight", 1)
            self.row_w[int(i)] = float(w) if pd.notna(w) else 1.0
        self.events = list(self.by_e.keys())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            e = self.events[int(rng.integers(0, len(self.events)))]
            stations = list(self.by_e[e])
            st = stations[int(rng.integers(0, len(stations)))]
            idxs = self.by_es[(e, st)]
            ww = self.row_w[np.asarray(idxs, dtype=int)]
            ww = ww / ww.sum()
            yield int(idxs[int(rng.choice(len(idxs), p=ww))])

    def __len__(self) -> int:
        return self.num_samples


class DKPNPartialCropDataset(Dataset):
    def __init__(
        self,
        catalog: pd.DataFrame,
        read_fn,
        *,
        augment: bool = True,
        seed: int = 0,
        sigma_s: float = 0.1,
        n_waveform: int = 12000,
    ):
        self.cat = catalog.reset_index(drop=True)
        self.read_fn = read_fn
        self.augment = bool(augment)
        self.seed = int(seed)
        self.sigma = float(sigma_s) * 100.0
        self.n_waveform = int(n_waveform)
        self.aug_cfg = AugmentConfig()
        self.virtual_epoch = 0
        ensure_dkpn_on_path()
        from dkpn.core import DKPN

        self._default_args = DKPN().default_args
        self._xcache: dict[tuple, np.ndarray] = {}

    def set_virtual_epoch(self, epoch: int) -> None:
        self.virtual_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cat)

    def _crop_kind_for_row(self, row) -> str:
        if "crop_kind" in row.index and pd.notna(row.get("crop_kind")):
            ck = str(row["crop_kind"]).strip()
            if ck and ck.lower() not in {"nan", "none"}:
                return ck
        is_noise = bool(row.get("is_noise", False))
        has_p = bool(row.get("has_p_label", False))
        has_s = bool(row.get("has_s_label", False))
        if not has_p and not is_noise:
            p = row.get("p_arrival_sample", np.nan)
            has_p = pd.notna(p) and np.isfinite(float(p)) if p is not None else False
        if not has_s and not is_noise:
            s = row.get("s_arrival_sample", np.nan)
            has_s = pd.notna(s) and np.isfinite(float(s)) if s is not None else False
        return scheduled_crop_kind(
            virtual_epoch=self.virtual_epoch,
            trace_name=str(row["trace_name"]),
            is_noise=is_noise,
            has_p_label=has_p,
            has_s_label=has_s,
        )

    def _read(self, name: str, is_noise: bool) -> np.ndarray:
        try:
            return self.read_fn(name, is_noise=is_noise)
        except TypeError:
            return self.read_fn(name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.cat.iloc[idx]
        rng = np.random.default_rng(self.seed + idx * 10007)
        is_noise = bool(row.get("is_noise", False))
        kind = self._crop_kind_for_row(row)
        p = float(row["p_arrival_sample"]) if (not is_noise and pd.notna(row.get("p_arrival_sample"))) else None
        s = float(row["s_arrival_sample"]) if (not is_noise and pd.notna(row.get("s_arrival_sample"))) else None
        if p is not None and not np.isfinite(p):
            p = None
        if s is not None and not np.isfinite(s):
            s = None
        enz = np.asarray(self._read(str(row["trace_name"]), is_noise), dtype=np.float32)
        n = int(enz.shape[-1])
        start = crop_kind_start(kind, n, p, s, rng)
        win = WIN_RAW
        crop = enz[:, start : start + win]
        valid_len = int(crop.shape[-1])
        if valid_len < win:
            pad = np.zeros((3, win), dtype=np.float32)
            pad[:, :valid_len] = crop
            crop = pad
        # pad_mask on label timeline (after dropping fstab)
        lab_valid = max(0, valid_len - FSTAB)
        pad_mask = np.zeros(IN_SAMPLES, dtype=np.float32)
        pad_mask[: min(IN_SAMPLES, lab_valid)] = 1.0

        crop_zne = enz_to_zne(crop)
        p_c = label_coord(p, start)
        s_c = label_coord(s, start)
        vis_p = phase_visible(p, start)
        vis_s = phase_visible(s, start)
        cache_key = (str(row["trace_name"]), int(start), int(is_noise), 0 if not self.augment else None)
        if not self.augment and cache_key in self._xcache:
            x = self._xcache[cache_key]
        else:
            if self.augment:
                shift = int(rng.integers(-self.aug_cfg.time_shift, self.aug_cfg.time_shift + 1))
                crop_zne = np.roll(crop_zne, shift, axis=-1)
                pad_mask = np.roll(pad_mask, shift)
                if p_c is not None:
                    p_c = p_c + shift
                if s_c is not None:
                    s_c = s_c + shift
                vis_p = p_c is not None and 0.0 <= p_c < IN_SAMPLES
                vis_s = s_c is not None and 0.0 <= s_c < IN_SAMPLES
                crop_zne = crop_zne * float(rng.uniform(*self.aug_cfg.amp_scale))
                if rng.random() < self.aug_cfg.polarity_flip_prob:
                    crop_zne = -crop_zne
                if rng.random() < self.aug_cfg.gap_prob:
                    g0 = int(rng.integers(0, crop_zne.shape[-1]))
                    g1 = min(crop_zne.shape[-1], g0 + int(rng.integers(10, 200)))
                    crop_zne[:, g0:g1] = 0.0

            cf = compute_cf_5ch(crop_zne, self._default_args)
            x = normalize_cf_window(cf, in_samples=IN_SAMPLES, fp_stabilization_s=4.0, sr=100.0)
            if not np.isfinite(x).all():
                raise RuntimeError(f"non-finite CF {row['trace_name']}")
            if not self.augment:
                self._xcache[cache_key] = x

        w = build_partial_weights(
            n=IN_SAMPLES,
            p_c=p_c,
            s_c=s_c,
            vis_p=vis_p,
            vis_s=vis_s,
            is_noise=is_noise,
            pad_mask=pad_mask,
            sigma=self.sigma,
            has_p_label=bool(row.get("has_p_label", p is not None)),
            has_s_label=bool(row.get("has_s_label", s is not None)),
        )
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "p_pos": torch.from_numpy(w["p_pos"]),
            "s_pos": torch.from_numpy(w["s_pos"]),
            "n_pos": torch.from_numpy(w["n_pos"]),
            "not_p": torch.from_numpy(w["not_p"]),
            "not_s": torch.from_numpy(w["not_s"]),
            "pad_mask": torch.from_numpy(w["pad_mask"]),
            "vis_p": torch.tensor(float(vis_p)),
            "vis_s": torch.tensor(float(vis_s)),
            "p_c": torch.tensor(float(p_c) if p_c is not None and vis_p else -1.0),
            "s_c": torch.tensor(float(s_c) if s_c is not None and vis_s else -1.0),
            "crop_kind": kind,
            "trace_name": str(row["trace_name"]),
            "event_id": str(row.get("event_id", "")),
            "is_noise": bool(is_noise),
            "bucket": visibility_bucket(vis_p, vis_s),
        }
