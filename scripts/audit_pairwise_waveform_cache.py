#!/usr/bin/env python
"""Audit pairwise waveform cache vs live HDF5 crops (bit/near-bit equivalence)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.pairwise.cache import crop_pair_from_wave, load_split_cache
from earthquake.utils import ensure_dir

CACHE = artifacts_dir() / "cache" / "pairwise_waveform"
PILOT = artifacts_dir() / "results" / "pairwise_pilot"
REPORT = ROOT / "reports" / "pairwise"


def audit_split(split: str, n: int, seed: int, reader: InstanceHDF5Reader) -> dict:
    store = load_split_cache(CACHE, split)
    man = store.manifest
    rng = np.random.default_rng(seed)
    n = min(n, len(man))
    ix = rng.choice(len(man), size=n, replace=False)
    sample = man.iloc[ix]

    max_diffs = []
    mean_diffs = []
    n_exact = 0
    mismatches = []
    for row in sample.itertuples(index=False):
        tn = str(row.trace_name)
        w = reader.read_waveform(tn)
        x1, x2, m1, m2, _ = crop_pair_from_wave(w, float(row.c1_sample), float(row.c2_sample))
        c1, c2, cm1, cm2 = store.get(tn)
        d = max(
            float(np.max(np.abs(x1 - c1))),
            float(np.max(np.abs(x2 - c2))),
            float(np.max(np.abs(m1 - cm1))),
            float(np.max(np.abs(m2 - cm2))),
        )
        md = float(
            np.mean(
                [
                    np.mean(np.abs(x1 - c1)),
                    np.mean(np.abs(x2 - c2)),
                    np.mean(np.abs(m1 - cm1)),
                    np.mean(np.abs(m2 - cm2)),
                ]
            )
        )
        max_diffs.append(d)
        mean_diffs.append(md)
        if d == 0.0:
            n_exact += 1
        elif d > 1e-6:
            mismatches.append({"trace_name": tn, "max_abs_diff": d})

    return {
        "split": split,
        "n_checked": n,
        "n_exact": n_exact,
        "max_observed_diff": float(max(max_diffs) if max_diffs else 0.0),
        "mean_observed_diff": float(np.mean(mean_diffs) if mean_diffs else 0.0),
        "n_mismatch_gt_1e-6": len(mismatches),
        "mismatches_head": mismatches[:10],
        "passed": len(mismatches) == 0,
    }


def main() -> None:
    ensure_dir(REPORT)
    if not (CACHE / "CACHE.LOCK.json").exists():
        raise SystemExit("CACHE.LOCK.json missing; build cache first")

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    plan = [("train", 200), ("calibration", 100), ("heldout_eval", 200)]
    t0 = time.time()
    results = []
    for split, n in plan:
        print(f"audit {split} n={n}", flush=True)
        results.append(audit_split(split, n, seed=42 + hash(split) % 1000, reader=reader))
        print(results[-1], flush=True)

    n_checked = sum(r["n_checked"] for r in results)
    passed = all(r["passed"] for r in results) and n_checked >= 500
    out = {
        "n_checked_total": n_checked,
        "n_exact_total": sum(r["n_exact"] for r in results),
        "max_observed_diff_global": max(r["max_observed_diff"] for r in results),
        "mean_observed_diff_global": float(np.mean([r["mean_observed_diff"] for r in results])),
        "passed": passed,
        "threshold": "max_abs_diff == 0 preferred; fail if >1e-6",
        "by_split": results,
        "wall_s": time.time() - t0,
        "cache_lock_sha256": json.loads((CACHE / "CACHE.LOCK.json").read_text()).get("cache_lock_sha256"),
    }
    save_json(out, PILOT / "waveform_cache_equivalence.json")
    lines = [
        "# Waveform cache equivalence audit",
        "",
        f"**passed={passed}**  n_checked={n_checked}  n_exact={out['n_exact_total']}",
        f"max_diff={out['max_observed_diff_global']}  mean_diff={out['mean_observed_diff_global']}",
        "",
        "Method: reload ENZ from InstanceHDF5Reader, apply frozen crop_pair_from_wave,",
        "compare to memmap cache float32 crops/masks.",
        "",
        "```json",
        json.dumps(out, indent=2),
        "```",
        "",
    ]
    (REPORT / "waveform_cache_equivalence_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("PASSED" if passed else "FAILED", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
