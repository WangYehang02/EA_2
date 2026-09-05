#!/usr/bin/env python
"""Build sequential pairwise waveform crop cache (memmap).

Avoids training-time random HDF5 seeks. Preprocessing is frozen identical to
run_pairwise_pilot.py: ENZ->ZNE, full-trace mean/std, [c-1.5s, c+2.5s].

Does not read confirm / full-dev. Does not modify Stage-6 locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.pairwise.cache import WIN, crop_pair_from_wave, rows_needing_cache
from earthquake.utils import ensure_dir

CACHE = artifacts_dir() / "cache" / "pairwise_waveform"
PILOT = artifacts_dir() / "results" / "pairwise_pilot"
SPLIT_NAMES = ("train", "calibration", "heldout_eval")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _worker_shard(payload: dict) -> dict:
    """Process a contiguous sorted shard; write into shared memmap by absolute idx."""
    h5_path = payload["h5_path"]
    waves_path = payload["waves_path"]
    masks_path = payload["masks_path"]
    rows = payload["rows"]  # list of dicts with idx, trace_name, c1_sample, c2_sample
    T = int(payload["T"])
    n_total = int(payload["n_total"])

    waves = np.lib.format.open_memmap(waves_path, mode="r+", dtype=np.float32, shape=(n_total, 2, 3, T))
    masks = np.lib.format.open_memmap(masks_path, mode="r+", dtype=np.float32, shape=(n_total, 2, T))

    # One HDF5 handle per worker; rows pre-sorted by trace_name for locality
    n_open = 0
    failed = 0
    padded = 0
    t0 = time.time()
    bytes_read = 0
    with h5py.File(h5_path, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=400_003) as f:
        n_open += 1
        g = f["data"]
        for r in rows:
            idx = int(r["idx"])
            tn = str(r["trace_name"])
            try:
                arr = np.asarray(g[tn][...], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    arr = arr.T
                if arr.shape[0] != 3:
                    raise ValueError(f"bad shape {arr.shape}")
                bytes_read += int(arr.nbytes)
                x1, x2, m1, m2, meta = crop_pair_from_wave(arr, r["c1_sample"], r["c2_sample"])
                waves[idx, 0] = x1
                waves[idx, 1] = x2
                masks[idx, 0] = m1
                masks[idx, 1] = m2
                if meta["padded_c1"] or meta["padded_c2"]:
                    padded += 1
            except Exception:
                failed += 1
                waves[idx] = 0
                masks[idx] = 0
    waves.flush()
    masks.flush()
    return {
        "n": len(rows),
        "failed": failed,
        "padded": padded,
        "n_open": n_open,
        "bytes_read": bytes_read,
        "wall_s": time.time() - t0,
        "worker": payload.get("worker_id"),
    }


def build_split(pairs: pd.DataFrame, split: str, h5_path: Path, n_workers: int) -> dict:
    sub = rows_needing_cache(pairs[pairs.split == split]).copy()
    # Sort by trace_name for sequential-ish HDF5 locality
    sub = sub.sort_values("trace_name").reset_index(drop=True)
    n = len(sub)
    T = WIN.n_samples
    out_w = CACHE / f"pair_cache_{split}.npy"
    out_m = CACHE / f"pair_cache_{split}_masks.npy"
    print(f"[{split}] n={n} workers={n_workers} -> {out_w.name}", flush=True)

    waves = np.lib.format.open_memmap(out_w, mode="w+", dtype=np.float32, shape=(n, 2, 3, T))
    masks = np.lib.format.open_memmap(out_m, mode="w+", dtype=np.float32, shape=(n, 2, T))
    waves.flush()
    masks.flush()
    del waves, masks

    rows = [
        {
            "idx": int(i),
            "trace_name": str(r.trace_name),
            "c1_sample": float(r.c1_sample),
            "c2_sample": float(r.c2_sample),
        }
        for i, r in enumerate(sub.itertuples(index=False))
    ]
    # Contiguous sorted shards (few HDF5 handles; sequential within each)
    n_workers = max(1, min(n_workers, n if n else 1))
    shards = []
    block = int(np.ceil(n / n_workers)) if n else 0
    for wi in range(n_workers):
        a = wi * block
        b = min(n, (wi + 1) * block)
        if a >= b:
            continue
        shards.append(
            {
                "h5_path": str(h5_path),
                "waves_path": str(out_w),
                "masks_path": str(out_m),
                "rows": rows[a:b],
                "T": T,
                "n_total": n,
                "worker_id": wi,
            }
        )

    t0 = time.time()
    stats = []
    if n_workers == 1 or len(shards) <= 1:
        for sh in shards:
            stats.append(_worker_shard(sh))
            print(f"  [{split}] worker done n={stats[-1]['n']} fail={stats[-1]['failed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as ex:
            futs = [ex.submit(_worker_shard, sh) for sh in shards]
            for fut in as_completed(futs):
                st = fut.result()
                stats.append(st)
                print(
                    f"  [{split}] worker{st['worker']} n={st['n']} "
                    f"fail={st['failed']} {st['n']/max(st['wall_s'],1e-6):.1f} it/s",
                    flush=True,
                )

    wall = time.time() - t0
    man = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int32),
            "trace_name": sub.trace_name.astype(str).to_numpy(),
            "event_id": sub.event_id.astype(str).to_numpy(),
            "c1_sample": sub.c1_sample.to_numpy(float),
            "c2_sample": sub.c2_sample.to_numpy(float),
            "split": split,
        }
    )
    man_path = CACHE / f"pair_cache_{split}_manifest.parquet"
    man.to_parquet(man_path, index=False)

    nbytes = out_w.stat().st_size + out_m.stat().st_size
    failed = sum(s["failed"] for s in stats)
    padded = sum(s["padded"] for s in stats)
    opens = sum(s["n_open"] for s in stats)
    bread = sum(s["bytes_read"] for s in stats)
    meta = {
        "split": split,
        "n_traces": n,
        "n_pairs": n,
        "shape_waves": [n, 2, 3, T],
        "dtype": "float32",
        "window": {"pre_s": WIN.pre_s, "post_s": WIN.post_s, "n_samples": T},
        "preprocessing": "ENZ->ZNE; full-trace mean/std; crop; zero-pad masked",
        "bytes": nbytes,
        "wall_s": wall,
        "hdf5_open_count": opens,
        "bytes_read_from_hdf5": bread,
        "avg_MB_s": (bread / 1e6) / max(wall, 1e-6),
        "avg_traces_per_s": n / max(wall, 1e-6),
        "failed_crops": failed,
        "padded_crops": padded,
        "n_workers": len(shards),
        "waves_sha256": _sha_file(out_w),
        "masks_sha256": _sha_file(out_m),
        "manifest_sha256": _sha_file(man_path),
    }
    save_json(meta, CACHE / f"pair_cache_{split}_meta.json")
    print(
        f"[{split}] DONE wall={wall:.1f}s {meta['avg_traces_per_s']:.1f} tr/s "
        f"fail={failed} padded={padded} bytes={nbytes/1e9:.3f}GB",
        flush=True,
    )
    if failed > 0:
        raise RuntimeError(f"{split}: {failed} failed crops")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="HDF5 handles = workers (contiguous shards)")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES))
    args = ap.parse_args()

    ensure_dir(CACHE)
    if not (PILOT / "pairs_ranker_train.parquet").exists():
        raise SystemExit("missing pairs_ranker_train.parquet")
    if not (PILOT / "SPLIT.LOCK.json").exists():
        raise SystemExit("missing SPLIT.LOCK.json")

    pairs = pd.read_parquet(PILOT / "pairs_ranker_train.parquet")
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    print("HDF5", h5, flush=True)
    print("CACHE", CACHE, flush=True)

    t0 = time.time()
    all_meta = {}
    for split in args.splits:
        all_meta[split] = build_split(pairs, split, h5, args.workers)

    wall = time.time() - t0
    total_bytes = sum(m["bytes"] for m in all_meta.values())
    total_n = sum(m["n_traces"] for m in all_meta.values())
    split_lock = json.loads((PILOT / "SPLIT.LOCK.json").read_text())
    lock = {
        "created_unix": time.time(),
        "wall_s": wall,
        "total_traces": total_n,
        "total_pairs": total_n,
        "total_bytes": total_bytes,
        "hdf5_path": str(h5),
        "split_lock_event_sha256": split_lock["event_sha256"],
        "window": {"pre_s": WIN.pre_s, "post_s": WIN.post_s, "n_samples": WIN.n_samples},
        "component_order": "ZNE (from Instance ENZ)",
        "normalization": "full_trace_mean_std",
        "splits": all_meta,
        "n_workers_per_split": args.workers,
        "note": "built for pairwise IO fix; does not modify Stage-6",
    }
    # overall hash of lock body without self-hash
    body = json.dumps(lock, sort_keys=True).encode()
    lock["cache_lock_sha256"] = hashlib.sha256(body).hexdigest()
    save_json(lock, CACHE / "CACHE.LOCK.json")
    save_json(lock, CACHE / "pair_cache_meta.json")
    # mirror summary into pilot results
    save_json(lock, PILOT / "waveform_cache_meta.json")
    print("CACHE.LOCK written", CACHE / "CACHE.LOCK.json", flush=True)
    print(f"TOTAL n={total_n} bytes={total_bytes/1e9:.3f}GB wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    # ProcessPool needs spawn-safe main
    main()
