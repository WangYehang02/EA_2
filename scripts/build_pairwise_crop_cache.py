#!/usr/bin/env python
"""Pre-cache pairwise crops to speed waveform pilot training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.pairwise.model import PairwiseWindowConfig, crop_candidate_window
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_pilot"
WIN = PairwiseWindowConfig()


def main() -> None:
    out = ensure_dir(OUT / "crop_cache")
    pairs = pd.read_parquet(OUT / "pairs_ranker_train.parquet")
    need = pairs[
        ((pairs.split == "train") & pairs.y_choose_c2.isin([0, 1]))
        | ((pairs.split.isin(["calibration", "heldout_eval"])) & (pairs.n_candidates >= 2))
    ].copy()
    need = need.drop_duplicates("trace_name").reset_index(drop=True)
    n = len(need)
    T = WIN.n_samples
    print(f"caching {n} traces, T={T}", flush=True)

    index = {}
    waves = np.lib.format.open_memmap(out / "waves.npy", mode="w+", dtype=np.float32, shape=(n, 2, 3, T))
    masks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.float32, shape=(n, 2, T))

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    for idx, (_, row) in enumerate(tqdm(need.iterrows(), total=n)):
        tn = str(row.trace_name)
        w = reader.read_waveform(tn)
        zne = np.stack([w[2], w[1], w[0]], axis=0).astype(np.float32)
        mean = zne.mean(axis=1)
        std = np.maximum(zne.std(axis=1), 1e-6)
        x1, m1 = crop_candidate_window(zne, float(row.c1_sample), cfg=WIN, full_trace_mean=mean, full_trace_std=std)
        x2, m2 = crop_candidate_window(zne, float(row.c2_sample), cfg=WIN, full_trace_mean=mean, full_trace_std=std)
        waves[idx, 0] = x1
        waves[idx, 1] = x2
        masks[idx, 0] = m1
        masks[idx, 1] = m2
        index[tn] = idx
        if (idx + 1) % 5000 == 0:
            waves.flush()
            masks.flush()
    waves.flush()
    masks.flush()
    save_json({"n": n, "T": T, "index_size": len(index)}, out / "meta.json")
    # save index
    pd.Series(index, dtype=object).to_frame("idx").to_csv(out / "trace_index.csv")
    # also json
    import json as _json

    (out / "trace_index.json").write_text(_json.dumps(index), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()
