"""Pairwise waveform crop cache helpers (frozen preprocessing).

Must stay bit-compatible with scripts/run_pairwise_pilot.py crop path:
  ENZ from InstanceHDF5Reader -> ZNE stack -> full-trace mean/std norm -> crop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earthquake.pairwise.model import PairwiseWindowConfig, crop_candidate_window

WIN = PairwiseWindowConfig()

# Which rows need crops (same policy as aborted crop_cache builder)
def rows_needing_cache(pairs: pd.DataFrame) -> pd.DataFrame:
    need = pairs[
        ((pairs.split == "train") & pairs.y_choose_c2.isin([0, 1]))
        | ((pairs.split.isin(["calibration", "heldout_eval"])) & (pairs.n_candidates >= 2))
    ].copy()
    return need.drop_duplicates("trace_name").reset_index(drop=True)


def waveform_to_zne(wave_enz: np.ndarray) -> np.ndarray:
    """InstanceHDF5Reader returns ENZ (3,T); Stage-6/pairwise use ZNE."""
    assert wave_enz.ndim == 2 and wave_enz.shape[0] == 3
    return np.stack([wave_enz[2], wave_enz[1], wave_enz[0]], axis=0).astype(np.float32)


def crop_pair_from_wave(
    wave_enz: np.ndarray,
    c1_sample: float,
    c2_sample: float,
    *,
    cfg: PairwiseWindowConfig = WIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return x1,x2,m1,m2 and padding stats. Full-trace normalization."""
    zne = waveform_to_zne(wave_enz)
    mean = zne.mean(axis=1)
    std = np.maximum(zne.std(axis=1), 1e-6)
    x1, m1 = crop_candidate_window(zne, float(c1_sample), cfg=cfg, full_trace_mean=mean, full_trace_std=std)
    x2, m2 = crop_candidate_window(zne, float(c2_sample), cfg=cfg, full_trace_mean=mean, full_trace_std=std)
    meta = {
        "padded_c1": bool(m1.min() < 1.0),
        "padded_c2": bool(m2.min() < 1.0),
        "mask_sum_c1": float(m1.sum()),
        "mask_sum_c2": float(m2.sum()),
    }
    return x1, x2, m1, m2, meta


class PairWaveformCache:
    """Memmap-backed [N,2,3,T] waves + [N,2,T] masks + trace_name->idx."""

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.split = split
        self.waves = np.load(self.root / f"pair_cache_{split}.npy", mmap_mode="r")
        self.masks = np.load(self.root / f"pair_cache_{split}_masks.npy", mmap_mode="r")
        man = pd.read_parquet(self.root / f"pair_cache_{split}_manifest.parquet")
        self.index = {str(r.trace_name): int(r.idx) for r in man.itertuples(index=False)}
        self.manifest = man

    def get(self, trace_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.index[str(trace_name)]
        return (
            np.asarray(self.waves[idx, 0]),
            np.asarray(self.waves[idx, 1]),
            np.asarray(self.masks[idx, 0]),
            np.asarray(self.masks[idx, 1]),
        )

    def as_pilot_pack(self) -> dict:
        """Compatible with run_pairwise_pilot PairDataset crop_cache dict."""
        return {"waves": self.waves, "masks": self.masks, "index": self.index}


def load_split_cache(cache_root: Path, split: str) -> PairWaveformCache:
    return PairWaveformCache(cache_root, split)


def load_unified_crop_pack(cache_root: Path, splits: list[str] | None = None) -> dict:
    """Merge per-split caches into one index for PairDataset (disjoint traces)."""
    splits = splits or ["train", "calibration", "heldout_eval"]
    packs = [load_split_cache(cache_root, s) for s in splits]
    # Build concatenated views via index remapping into a virtual pack
    # Prefer simple dict of per-trace arrays refs via multi-store lookup
    stores = {s: load_split_cache(cache_root, s) for s in splits}
    index: dict[str, tuple[str, int]] = {}
    for s, store in stores.items():
        for tn, idx in store.index.items():
            index[tn] = (s, idx)
    return {"stores": stores, "index": index, "mode": "multi_split"}


def get_from_unified(pack: dict, tn: str):
    if pack.get("mode") == "multi_split":
        s, idx = pack["index"][tn]
        store = pack["stores"][s]
        return (
            np.asarray(store.waves[idx, 0]),
            np.asarray(store.waves[idx, 1]),
            np.asarray(store.masks[idx, 0]),
            np.asarray(store.masks[idx, 1]),
        )
    idx = pack["index"][tn]
    return (
        np.asarray(pack["waves"][idx, 0]),
        np.asarray(pack["waves"][idx, 1]),
        np.asarray(pack["masks"][idx, 0]),
        np.asarray(pack["masks"][idx, 1]),
    )


def read_cache_lock(cache_root: Path) -> dict:
    return json.loads((Path(cache_root) / "CACHE.LOCK.json").read_text(encoding="utf-8"))
