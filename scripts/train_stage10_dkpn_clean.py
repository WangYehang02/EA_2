#!/usr/bin/env python
"""Stage 10B — clean DKPN (random init) on Stage-6 picker_train.

Modes: overfit | smoke | throughput | train
Never loads official INSTANCE pretrained weights for main runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage10.dkpn_clean import (
    CleanDKPNCropDataset,
    build_dkpn_random,
    dkpn_logits,
    masked_soft_ce,
)
from earthquake.utils import ensure_dir

CACHE = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10")


def _build_train_meta(*, include_p_only: bool, noise_ratio: float, max_traces: int | None, seed: int) -> pd.DataFrame:
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    picker = set(load_full_trace_names("stage6_picker_train"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    dev = set(load_full_event_ids("stage6_dev"))

    ev = events[events["trace_name"].astype(str).isin(picker)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK: picker_train overlaps confirm")
    if set(ev["event_id"].astype(str)) & dev:
        raise SystemExit("LEAK: picker_train overlaps dev")

    p = pd.to_numeric(ev["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ps = ev[p.notna() & s.notna()].copy()
    po = ev[p.notna() & s.isna()].copy()
    ps["is_noise"] = False
    po["is_noise"] = False
    ps["supervision"] = "ps_both"
    po["supervision"] = "p_only"

    parts = [ps]
    if include_p_only:
        parts.append(po)
    train_ev = pd.concat(parts, ignore_index=True)

    n_noise = int(min(len(noise), max(1, round(len(train_ev) * noise_ratio))))
    rng = np.random.default_rng(seed)
    nsel = noise.sample(n=n_noise, random_state=int(seed)).copy()
    nsel["is_noise"] = True
    nsel["supervision"] = "noise"
    if "event_id" not in nsel.columns:
        nsel["event_id"] = "NOISE"
    if "sampling_rate_hz" not in nsel.columns:
        nsel["sampling_rate_hz"] = 100.0

    cols = ["trace_name", "event_id", "p_arrival_sample", "s_arrival_sample", "sampling_rate_hz", "is_noise", "supervision"]
    for c in cols:
        if c not in train_ev.columns:
            train_ev[c] = np.nan if c.endswith("sample") else None
        if c not in nsel.columns:
            nsel[c] = np.nan if c.endswith("sample") else (False if c == "is_noise" else None)
    out = pd.concat([train_ev[cols], nsel[cols]], ignore_index=True)
    if max_traces is not None and len(out) > max_traces:
        out = out.sample(n=max_traces, random_state=seed).reset_index(drop=True)
    return out


def _make_loader(meta, wave, *, batch, workers, augment, seed, shuffle=True):
    confirm = set(load_full_event_ids("stage6_internal_confirm"))

    def read_fn(name, is_noise=False):
        return wave.read(name, is_noise=is_noise)

    ds = CleanDKPNCropDataset(
        meta,
        read_fn,
        augment=augment,
        seed=seed,
        allow_p_only=True,
        confirm_event_ids=confirm,
    )
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    ), ds


def train_loop(model, loader, opt, device, *, max_steps=None, amp=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    losses = []
    t0 = time.time()
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf in input batch — fail-fast")
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = dkpn_logits(model, x)
            loss = masked_soft_ce(logits, y, m)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        n += x.shape[0]
    dt = time.time() - t0
    return {
        "mean_loss": float(np.mean(losses)) if losses else None,
        "n_steps": len(losses),
        "n_traces": n,
        "sec": dt,
        "traces_per_sec": (n / dt) if dt > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "smoke", "throughput", "train"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--noise-ratio", type=float, default=0.2)
    ap.add_argument("--include-p-only", action="store_true", default=True)
    ap.add_argument("--no-p-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gpu", type=int, default=None, help="CUDA_VISIBLE single GPU index")
    args = ap.parse_args()

    if args.no_p_only:
        args.include_p_only = False
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = ensure_dir(
        Path(args.out_dir)
        if args.out_dir
        else (CACHE / "dkpn_clean" / f"{args.mode}_seed{args.seed}")
    )
    # also symlink-friendly under repo
    repo_out = ensure_dir(ROOT / "artifacts/results/stage10/dkpn_clean" / f"{args.mode}_seed{args.seed}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(
        inst / "events" / "Instance_events_counts.hdf5",
        inst / "noise" / "Instance_noise.hdf5",
    )

    if args.mode == "overfit":
        max_tr = args.max_traces or 64
        meta = _build_train_meta(include_p_only=True, noise_ratio=0.1, max_traces=max_tr, seed=args.seed)
        # Prefer PS-both for overfit
        meta = meta[meta["supervision"] == "ps_both"].head(max_tr).reset_index(drop=True)
        loader, ds = _make_loader(meta, wave, batch=min(4, len(meta)), workers=0, augment=False, seed=args.seed)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = []
        for ep in range(args.epochs or 30):
            stats = train_loop(model, loader, opt, device, amp=(device.type == "cuda"))
            hist.append({"epoch": ep, **stats})
            print({"epoch": ep, **stats}, flush=True)
            if stats["mean_loss"] is not None and stats["mean_loss"] < 0.05:
                break
        ckpt = out / "overfit.pt"
        torch.save({"model": model.state_dict(), "seed": args.seed}, ckpt)
        report = {
            "mode": "overfit",
            "n_traces": len(ds),
            "device": str(device),
            "history": hist,
            "final_loss": hist[-1]["mean_loss"] if hist else None,
            "overfit_ok": bool(hist and hist[-1]["mean_loss"] is not None and hist[-1]["mean_loss"] < 0.2),
            "ckpt": str(ckpt),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(report, out / "overfit_report.json")
        save_json(report, repo_out / "overfit_report.json")
        print(json.dumps(report, indent=2))
        return

    if args.mode == "smoke":
        max_tr = args.max_traces or 128
        meta = _build_train_meta(include_p_only=True, noise_ratio=0.2, max_traces=max_tr, seed=args.seed)
        loader, ds = _make_loader(meta, wave, batch=args.batch_size, workers=args.workers, augment=True, seed=args.seed)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = train_loop(model, loader, opt, device, max_steps=20, amp=(device.type == "cuda"))
        # one forward finite check
        batch = next(iter(loader))
        with torch.no_grad():
            y = model(batch["x"].to(device))
        report = {
            "mode": "smoke",
            "n_dataset": len(ds),
            "device": str(device),
            "train_stats": stats,
            "output_finite": bool(torch.isfinite(y).all().item()),
            "smoke_ok": bool(stats["mean_loss"] is not None and np.isfinite(stats["mean_loss"]) and torch.isfinite(y).all().item()),
            "utc": datetime.now(timezone.utc).isoformat(),
            "counts": meta["supervision"].value_counts().to_dict(),
        }
        save_json(report, out / "smoke_report.json")
        save_json(report, repo_out / "smoke_report.json")
        print(json.dumps(report, indent=2))
        if not report["smoke_ok"]:
            raise SystemExit(1)
        return

    if args.mode == "throughput":
        meta = _build_train_meta(include_p_only=True, noise_ratio=0.2, max_traces=args.max_traces or 256, seed=args.seed)
        loader, ds = _make_loader(meta, wave, batch=args.batch_size, workers=args.workers, augment=False, seed=args.seed)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # warmup
        train_loop(model, loader, opt, device, max_steps=2, amp=(device.type == "cuda"))
        stats = train_loop(model, loader, opt, device, max_steps=15, amp=(device.type == "cuda"))
        # estimate full picker_train PS+noise
        n_ps = 377087
        n_po = 631795 - 377087  # approx from audit
        n_noise = int(0.2 * (n_ps + (n_po if args.include_p_only else 0)))
        n_train = n_ps + (n_po if args.include_p_only else 0) + n_noise
        tps = stats["traces_per_sec"] or 1e-9
        # CF dominates; DDP scales ~linear until HDF5 bottleneck
        eta = {}
        for ng in (1, 4, 8):
            # conservative: 0.85 * ng scaling for 4/8
            scale = 1.0 if ng == 1 else 0.85 * ng
            sec_ep = n_train / (tps * scale)
            eta[f"{ng}gpu_sec_per_epoch"] = sec_ep
            eta[f"{ng}gpu_hours_per_epoch"] = sec_ep / 3600
            eta[f"{ng}gpu_hours_30ep"] = 30 * sec_ep / 3600
        report = {
            "mode": "throughput",
            "device": str(device),
            "measured": stats,
            "assumed_n_train_traces": n_train,
            "eta": eta,
            "note": "ETA from measured 1-GPU traces/sec; multi-GPU assumes 0.85*N scaling until HDF5 cache",
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(report, out / "throughput_report.json")
        save_json(report, repo_out / "throughput_report.json")
        print(json.dumps(report, indent=2))
        return

    # full train seed
    meta = _build_train_meta(
        include_p_only=not args.no_p_only,
        noise_ratio=args.noise_ratio,
        max_traces=args.max_traces,
        seed=args.seed,
    )
    meta.to_parquet(out / "train_index.parquet", index=False)
    counts = {
        "n_total": len(meta),
        "by_supervision": meta["supervision"].value_counts().to_dict(),
        "n_events": int(meta.loc[~meta["is_noise"].astype(bool), "event_id"].nunique()),
        "include_p_only": bool(args.include_p_only and not args.no_p_only),
        "noise_ratio": args.noise_ratio,
        "seed": args.seed,
        "refuse_instance_pretrained": True,
    }
    save_json(counts, out / "train_counts.json")
    save_json(counts, repo_out / "train_counts.json")

    loader, ds = _make_loader(meta, wave, batch=args.batch_size, workers=args.workers, augment=True, seed=args.seed, shuffle=True)
    model = build_dkpn_random().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    pid_path = out / "TRAIN.PID"
    pid_path.write_text(str(os.getpid()))
    (out / "TRAIN.RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    try:
        for ep in range(args.epochs):
            stats = train_loop(model, loader, opt, device, amp=(device.type == "cuda"))
            hist.append({"epoch": ep, **stats, "utc": datetime.now(timezone.utc).isoformat()})
            ensure_dir(out / "checkpoints")
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "seed": args.seed, "opt": opt.state_dict()},
                out / "checkpoints" / "last.pt",
            )
            save_json({"history": hist, "counts": counts}, out / "train_history.json")
            print({"epoch": ep, **stats}, flush=True)
        (out / "TRAIN.DONE").write_text(datetime.now(timezone.utc).isoformat())
        if (out / "TRAIN.RUNNING").exists():
            (out / "TRAIN.RUNNING").unlink()
    except Exception as e:
        (out / "TRAIN.FAILED").write_text(repr(e))
        raise
    finally:
        if pid_path.exists():
            pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
