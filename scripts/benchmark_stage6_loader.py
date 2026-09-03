#!/usr/bin/env python
"""Benchmark lazy HDF5 PhaseNet crop loader on 1/4/8 GPUs (short smoke)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import (
    EventBalancedSampler,
    LazyPhaseNetCropDataset,
    WorkerHDF5WaveformSource,
    filter_ps_and_noise,
)


def _setup_ddp(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)


def _cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_bench(rank: int, world: int, args: argparse.Namespace) -> dict:
    if world > 1:
        _setup_ddp(rank, world)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    picker_tr = set(load_full_trace_names("stage6_picker_train"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    train_ev = events[events["trace_name"].astype(str).isin(picker_tr)].copy()
    # smoke subset
    rng = np.random.default_rng(0)
    eids = train_ev["event_id"].astype(str).unique()
    take_e = set(rng.choice(eids, size=min(args.n_events, len(eids)), replace=False).tolist())
    train_ev = train_ev[train_ev["event_id"].astype(str).isin(take_e)]
    meta = filter_ps_and_noise(train_ev, noise, noise_ratio=0.1, seed=0, confirm_ids=confirm)
    if args.max_rows > 0:
        meta = meta.head(args.max_rows)

    root = resolve_instance_root()
    src = WorkerHDF5WaveformSource(
        root / "events" / "Instance_events_counts.hdf5",
        root / "noise" / "Instance_noise.hdf5",
    )
    ds = LazyPhaseNetCropDataset(
        meta,
        src,
        in_samples=3001,
        label_order="PSN",
        augment=True,
        seed=0,
        allow_p_only=False,
        confirm_event_ids=confirm,
    )
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
    else:
        sampler = EventBalancedSampler(meta, num_samples=min(len(meta), args.steps * args.batch_size), seed=0)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained("stead").to(device)
    set_train_bn_eval(model)
    if world > 1:
        model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    n_seen = 0
    t0 = time.time()
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(x)
            loss = -(y * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        n_seen += int(x.size(0))
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    traces_per_s = n_seen / max(elapsed, 1e-6)

    # gather
    payload = {
        "rank": rank,
        "world": world,
        "n_seen": n_seen,
        "elapsed_s": elapsed,
        "traces_per_s": traces_per_s,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "steps": args.steps,
    }
    if world > 1:
        objs = [None] * world
        dist.all_gather_object(objs, payload)
        if rank == 0:
            total_tps = sum(o["traces_per_s"] for o in objs)
            out = {
                "world_size": world,
                "per_rank": objs,
                "aggregate_traces_per_s_sum_ranks": total_tps,
                "approx_global_traces_per_s": total_tps,  # each rank measured locally over same wall
            }
            return out
        return payload
    return {"world_size": 1, "per_rank": [payload], "approx_global_traces_per_s": traces_per_s}


def _ddp_worker(rank: int, world: int, args: argparse.Namespace) -> None:
    try:
        res = run_bench(rank, world, args)
        if rank == 0:
            print(json.dumps(res, indent=2))
            if args.out:
                save_json(res, ROOT / args.out)
    finally:
        _cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.world_size <= 1:
        result = run_bench(0, 1, args)
        print(json.dumps(result, indent=2))
        if args.out:
            save_json(result, ROOT / args.out)
        return

    import torch.multiprocessing as mp

    mp.spawn(_ddp_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
