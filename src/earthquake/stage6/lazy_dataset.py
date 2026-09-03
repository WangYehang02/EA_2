"""Lazy HDF5 PhaseNet training dataset with per-worker file handles (no parent sharing)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.phasenet_finetune import AugmentConfig, soft_phase_targets, random_crop_window
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed, load_full_event_ids


class WorkerHDF5WaveformSource:
    """Open HDF5 inside each DataLoader worker; never share h5py handles across processes."""

    def __init__(self, events_h5: Path | str, noise_h5: Path | str | None = None):
        self.events_h5 = str(events_h5)
        self.noise_h5 = str(noise_h5) if noise_h5 else None
        self._events: InstanceHDF5Reader | None = None
        self._noise: InstanceHDF5Reader | None = None
        self._worker_pid: int | None = None

    def _ensure(self) -> None:
        import os

        pid = os.getpid()
        if self._events is not None and self._worker_pid == pid:
            return
        # Re-open after fork
        if self._events is not None:
            try:
                self._events.close()
            except Exception:
                pass
        if self._noise is not None:
            try:
                self._noise.close()
            except Exception:
                pass
        self._events = InstanceHDF5Reader(self.events_h5).open()
        self._noise = InstanceHDF5Reader(self.noise_h5).open() if self.noise_h5 else None
        self._worker_pid = pid

    def read(self, trace_name: str, *, is_noise: bool = False) -> np.ndarray:
        self._ensure()
        assert self._events is not None
        if is_noise:
            if self._noise is None:
                raise RuntimeError("noise HDF5 not configured")
            return self._noise.read_waveform(trace_name)
        try:
            return self._events.read_waveform(trace_name)
        except KeyError:
            if self._noise is not None:
                return self._noise.read_waveform(trace_name)
            raise


class LazyPhaseNetCropDataset(Dataset):
    """Crop training without copying waveforms to /home.

    First-version supervision: prefer rows with both P and S finite labels for events;
    noise rows are is_noise=True. P-only is excluded unless allow_p_only=True (masked loss TBD).
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        wave_source: WorkerHDF5WaveformSource,
        *,
        in_samples: int = 3001,
        sigma_s: float = 0.1,
        label_order: str = "PSN",
        augment: bool = True,
        seed: int = 0,
        aug_cfg: AugmentConfig | None = None,
        allow_p_only: bool = False,
        confirm_event_ids: set[str] | None = None,
    ):
        self.meta = meta.reset_index(drop=True)
        self.wave_source = wave_source
        self.in_samples = int(in_samples)
        self.sigma_s = float(sigma_s)
        self.label_order = str(label_order)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.aug_cfg = aug_cfg or AugmentConfig()
        self.allow_p_only = bool(allow_p_only)
        self.confirm_event_ids = set(confirm_event_ids or [])
        if not self.allow_p_only and "is_noise" in self.meta.columns:
            # keep noise; filter event rows to P+S
            is_n = self.meta["is_noise"].astype(bool)
            p = pd.to_numeric(self.meta.get("p_arrival_sample"), errors="coerce")
            s = pd.to_numeric(self.meta.get("s_arrival_sample"), errors="coerce")
            keep = is_n | (p.notna() & s.notna())
            self.meta = self.meta.loc[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        eid = str(row.get("event_id", ""))
        if eid in self.confirm_event_ids:
            assert_full_confirm_access_allowed(purpose="waveform_read")
        is_noise = bool(row.get("is_noise", False))
        name = str(row["trace_name"])
        wave = self.wave_source.read(name, is_noise=is_noise)
        if wave.shape[0] != 3:
            raise ValueError(f"expected (3,T) got {wave.shape}")
        # HDF5 reader returns ENZ; SeisBench PhaseNet component_order is ZNE.
        wave_zne = np.stack([wave[2], wave[1], wave[0]], axis=0)
        sr = float(row.get("sampling_rate_hz", 100.0) or 100.0)
        if not np.isfinite(sr) or sr <= 0:
            dt = row.get("trace_dt_s", np.nan)
            sr = float(1.0 / float(dt)) if pd.notna(dt) and float(dt) > 0 else 100.0
        p = float(row["p_arrival_sample"]) if pd.notna(row.get("p_arrival_sample")) else None
        s = float(row["s_arrival_sample"]) if pd.notna(row.get("s_arrival_sample")) else None
        if is_noise:
            p = s = None
        rng = np.random.default_rng(self.seed + idx * 10007)
        mode = "noise" if is_noise else ("both" if (p is not None and s is not None) else "p_only")
        start = random_crop_window(wave_zne.shape[1], self.in_samples, p, s, mode, rng)
        crop = wave_zne[:, start : start + self.in_samples]
        if crop.shape[1] < self.in_samples:
            pad = np.zeros((3, self.in_samples), dtype=np.float32)
            pad[:, : crop.shape[1]] = crop
            crop = pad
        # normalize per-channel; guard silent/degenerate traces
        x = crop.astype(np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x - x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = x / np.maximum(std, 1e-6)
        x = np.clip(x, -50.0, 50.0)
        if self.augment:
            cfg = self.aug_cfg
            if rng.random() < cfg.polarity_flip_prob:
                x = -x
            lo, hi = cfg.amp_scale
            x = x * float(rng.uniform(lo, hi))
            if cfg.noise_std > 0:
                x = x + rng.normal(0, cfg.noise_std, size=x.shape).astype(np.float32)
            if rng.random() < cfg.channel_dropout_prob:
                # drop at most one channel, keep 3-comp semantics
                ch = int(rng.integers(0, 3))
                x[ch] = 0.0
            x = np.clip(x, -50.0, 50.0)
        p_c = None if p is None else p - start
        s_c = None if s is None else s - start
        y = soft_phase_targets(
            self.in_samples,
            p_c,
            s_c,
            sampling_rate=sr,
            sigma_s=self.sigma_s,
            label_order=self.label_order,
        )
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "trace_name": name,
            "event_id": eid,
            "is_noise": is_noise,
        }


class EventBalancedSampler(Sampler[int]):
    """Approximate: uniform over events, then uniform over that event's rows; noise as own event."""

    def __init__(self, meta: pd.DataFrame, *, num_samples: int | None = None, seed: int = 0):
        self.meta = meta.reset_index(drop=True)
        self.num_samples = int(num_samples if num_samples is not None else len(self.meta))
        self.seed = int(seed)
        eids = self.meta["event_id"].astype(str).tolist()
        from collections import defaultdict

        self.by_event: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(eids):
            self.by_event[e].append(i)
        self.events = list(self.by_event.keys())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            e = self.events[int(rng.integers(0, len(self.events)))]
            idxs = self.by_event[e]
            yield int(idxs[int(rng.integers(0, len(idxs)))])

    def __len__(self) -> int:
        return self.num_samples


def filter_ps_and_noise(
    events_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    *,
    noise_ratio: float = 0.2,
    seed: int = 0,
    confirm_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build training meta: P+S event traces + real noise. Never uses confirm events."""
    confirm_ids = set(confirm_ids or [])
    if confirm_ids:
        # even listing confirm for training is forbidden conceptually
        bad = set(events_df["event_id"].astype(str)) & confirm_ids
        if bad:
            raise RuntimeError(f"confirm events leaked into training meta: {len(bad)}")
    p = pd.to_numeric(events_df["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(events_df["s_arrival_sample"], errors="coerce")
    ev = events_df.loc[p.notna() & s.notna()].copy()
    ev["is_noise"] = False
    n_noise = max(1, int(len(ev) * float(noise_ratio)))
    nz = noise_df.sample(n=min(n_noise, len(noise_df)), random_state=seed).copy()
    nz["is_noise"] = True
    nz["event_id"] = "NOISE"
    nz["p_arrival_sample"] = np.nan
    nz["s_arrival_sample"] = np.nan
    return pd.concat([ev, nz], ignore_index=True)
