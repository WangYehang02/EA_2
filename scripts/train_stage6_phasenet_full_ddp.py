#!/usr/bin/env python
"""Stage 6 FULL Phase A: DDP in-domain PhaseNet on splits_full picker_train (lazy HDF5)."""

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
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import report_pick_timing_bundle
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.full_splits import (
    assert_full_confirm_access_allowed,
    load_full_event_ids,
    load_full_trace_names,
)
from earthquake.stage6.lazy_dataset import (
    EventBalancedSampler,
    LazyPhaseNetCropDataset,
    WorkerHDF5WaveformSource,
    filter_ps_and_noise,
)
from earthquake.utils import ensure_dir


def _is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _row_sampling_rate(row: pd.Series, default: float = 100.0) -> float:
    sr = row.get("sampling_rate_hz", np.nan)
    if pd.notna(sr) and float(sr) > 0:
        return float(sr)
    dt = row.get("trace_dt_s", np.nan)
    if pd.notna(dt) and float(dt) > 0:
        return float(1.0 / float(dt))
    return float(default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6/phasenet_indomain_full.yaml")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/last.pt + train_history.json")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    # Auto-resume only from finite last.pt
    last_candidate = ROOT / cfg.get("out_dir", "artifacts/models/stage6/phasenet_full") / "checkpoints" / "last.pt"
    if (not cfg.get("smoke")) and last_candidate.exists():
        try:
            probe = torch.load(last_candidate, map_location="cpu", weights_only=False)
            n_nan = sum(int(torch.isnan(v).sum()) for v in probe["model"].values() if torch.is_floating_point(v))
            if n_nan == 0 and np.isfinite(float(probe.get("train_loss", 0.0) if probe.get("train_loss") is not None else 0.0)):
                args.resume = True
            else:
                print({"refuse_resume_nan_checkpoint": str(last_candidate), "n_nan": n_nan, "train_loss": probe.get("train_loss")}, flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"resume_probe_failed": repr(exc)}, flush=True)

    # DDP init if launched via torchrun
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(args.local_rank)
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight = cfg.get("init_weight", "stead")
    if cfg.get("refuse_instance_pretrained", True) and "instance" in str(weight).lower():
        raise SystemExit("Refusing INSTANCE pretrained init")

    out = ensure_dir(ROOT / cfg.get("out_dir", "artifacts/models/stage6/phasenet_full"))
    if (out / "TRAIN.DONE").exists() and not cfg.get("smoke"):
        if _is_main():
            print({"skip": "already done", "path": str(out)})
        return

    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    # Ensure seal blocks confirm waveform access paths
    try:
        assert_full_confirm_access_allowed(purpose="train_script_boot_check")
        raise SystemExit("CONFIRM SEAL BROKEN: access allowed without method_lock")
    except RuntimeError:
        pass

    index_dir = ROOT / cfg.get("index_dir", "artifacts/index_full")
    events = pd.read_parquet(index_dir / "events.parquet")
    noise = pd.read_parquet(index_dir / "noise.parquet")
    picker_tr = set(load_full_trace_names("stage6_picker_train"))
    dev_tr = set(load_full_trace_names("stage6_dev"))
    train_df = events[events["trace_name"].astype(str).isin(picker_tr)].copy()
    dev_df = events[events["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(train_df.event_id.astype(str)) & confirm:
        raise SystemExit("LEAK: picker_train overlaps confirm")
    if set(dev_df.event_id.astype(str)) & confirm:
        raise SystemExit("LEAK: dev overlaps confirm")

    if not cfg.get("use_all_picker_train_ps_traces", True):
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        eids = train_df.event_id.astype(str).unique()
        n_ev = int(cfg.get("n_train_events", 64))
        take = set(rng.choice(eids, size=min(n_ev, len(eids)), replace=False).tolist())
        train_df = train_df[train_df.event_id.astype(str).isin(take)]
        n_dv = int(cfg.get("n_dev_traces", 64))
        dev_df = dev_df.head(n_dv)

    meta = filter_ps_and_noise(
        train_df,
        noise,
        noise_ratio=float(cfg.get("noise_ratio", 0.2)),
        seed=int(cfg.get("seed", 0)),
        confirm_ids=confirm,
    )
    if cfg.get("include_p_only", False):
        raise SystemExit("P-only not enabled until masked loss is implemented+tested")

    if _is_main():
        save_json(
            {
                "role_label": cfg.get("role_label", "stage6_phaseA_full_instance"),
                "n_train_ps_traces": int((~meta.is_noise).sum()),
                "n_train_noise": int(meta.is_noise.sum()),
                "n_train_events": int(meta.loc[~meta.is_noise, "event_id"].nunique()),
                "n_dev_traces": int(len(dev_df)),
                "include_p_only": False,
                "cfg": cfg,
            },
            out / "run_meta.json",
        )
        meta.to_parquet(out / "train_meta.parquet", index=False)
        dev_df.to_parquet(out / "dev_meta.parquet", index=False)

    root = resolve_instance_root()
    src = WorkerHDF5WaveformSource(
        root / "events" / "Instance_events_counts.hdf5",
        root / "noise" / "Instance_noise.hdf5",
    )

    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained(weight)
    model.to(device)
    label_order = "".join(getattr(model, "labels", "PSN") or "PSN")
    if label_order.upper() != "PSN":
        raise SystemExit(f"Expected PSN, got {label_order}")

    ds = LazyPhaseNetCropDataset(
        meta,
        src,
        in_samples=int(getattr(model, "in_samples", 3001) or 3001),
        sigma_s=float(cfg.get("sigma_s", 0.1)),
        label_order=label_order,
        augment=True,
        seed=int(cfg.get("seed", 0)) + rank,
        allow_p_only=False,
        confirm_event_ids=confirm,
    )
    if world > 1:
        sampler = DistributedSampler(ds, shuffle=True, seed=int(cfg.get("seed", 0)))
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 24)),
            sampler=sampler,
            num_workers=int(cfg.get("workers", 4)),
            pin_memory=True,
            persistent_workers=int(cfg.get("workers", 4)) > 0,
        )
        # BN kept in eval/frozen; PhaseNet may leave unused grads under that policy
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        sampler = EventBalancedSampler(meta, seed=int(cfg.get("seed", 0)))
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 24)),
            sampler=sampler,
            num_workers=int(cfg.get("workers", 4)),
            pin_memory=True,
            persistent_workers=int(cfg.get("workers", 4)) > 0,
        )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", 50))
    scaler = GradScaler(enabled=bool(cfg.get("amp", False)))
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = ensure_dir(out / "checkpoints")

    # annotate eval uses independent reader on main rank only
    def annotate_score(df: pd.DataFrame) -> dict:
        raw_model.eval()
        ref = SeisBenchPhaseNetReference(weight=weight, device=str(device))
        ref.model = raw_model
        reader = InstanceHDF5Reader(root / "events" / "Instance_events_counts.hdf5").open()
        pp, tp, ps, ts, srs = [], [], [], [], []
        n_fail = 0
        with torch.no_grad():
            for _, row in df.iterrows():
                sr = _row_sampling_rate(row)
                try:
                    wave = reader.read_waveform(str(row.trace_name))
                    outp = ref.predict_row(wave, row, remap_to_waveform=True)
                    pp.append(float(outp["p_pred_sample_on_waveform"]))
                    ps.append(float(outp["s_pred_sample_on_waveform"]))
                except Exception as exc:  # noqa: BLE001 — keep eval robust on bad traces
                    n_fail += 1
                    if n_fail <= 5 and _is_main():
                        print({"annotate_skip": str(row.trace_name), "err": repr(exc)}, flush=True)
                    pp.append(float("nan"))
                    ps.append(float("nan"))
                tp.append(float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan)
                ts.append(float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan)
                srs.append(sr)
        reader.close()
        bundle_s = report_pick_timing_bundle(np.array(ps), np.array(ts), np.array(srs))
        bundle_p = report_pick_timing_bundle(np.array(pp), np.array(tp), np.array(srs))
        return {
            "p_f1@0.5": float(bundle_p["f1@0.5"]),
            "s_f1@0.5": float(bundle_s["f1@0.5"]),
            "s_f1@0.1": float(bundle_s["f1@0.1"]),
            "s_detected_ae_p95": float(bundle_s["detected_ae_p95"]),
            "s_miss_rate": float(bundle_s["miss_rate"]),
            "s_wrong_peak_rate": float(bundle_s["wrong_peak_rate"]),
            "score": float(bundle_s["f1@0.5"]),
            "n_annotate_fail": int(n_fail),
            "naming_warning": bundle_s["p95_naming_warning"],
        }

    history = []
    start_epoch = 0
    best = -1.0
    bad = 0
    patience = int(cfg.get("early_stopping_patience", 8))
    t0 = time.time()
    pick_n = min(int(cfg.get("annotate_eval_max_traces", 1024)), len(dev_df))
    pick_df = dev_df.head(pick_n).copy()

    hist_path = out / "train_history.json"
    last_path = ckpt / "last.pt"
    if args.resume and last_path.exists():
        blob = torch.load(last_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(blob["model"])
        start_epoch = int(blob.get("epoch", -1)) + 1
        if hist_path.exists():
            history = json.loads(hist_path.read_text()).get("history", [])
            scored = [h for h in history if isinstance(h.get("epoch"), int) and h.get("epoch", -1) >= 0]
            if scored:
                best = float(max(h.get("score", -1) for h in scored + [h for h in history if h.get("tag") == "pretrained"]))
            elif history:
                best = float(history[0].get("score", -1))
        if _is_main():
            print({"resume": True, "start_epoch": start_epoch, "best": best, "from": str(last_path)}, flush=True)
    elif _is_main():
        base = annotate_score(pick_df)
        history.append({"epoch": -1, **base, "tag": "pretrained"})
        best = float(base["score"])
        torch.save({"model": raw_model.state_dict(), "epoch": -1, "label_order": label_order, "cfg": cfg}, ckpt / "best.pt")
        save_json({"history": history}, hist_path)

    # Broadcast start_epoch / best
    if world > 1:
        t = torch.tensor([start_epoch, best], device=device, dtype=torch.float64)
        dist.broadcast(t, src=0)
        start_epoch = int(t[0].item())
        best = float(t[1].item())

    for epoch in range(start_epoch, epochs):
        if world > 1 and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        set_train_bn_eval(raw_model)
        model.train()
        set_train_bn_eval(raw_model)  # keep BN eval even in train mode for non-BN layers
        tr_loss = 0.0
        n_batches = 0
        n_skipped = 0
        clip_v = float(cfg.get("input_clip", 50.0))
        skip_bad = bool(cfg.get("skip_nonfinite_batches", True))
        for batch in tqdm(loader, desc=f"train:{epoch}", disable=not _is_main()):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            if clip_v > 0:
                x = torch.clamp(x, -clip_v, clip_v)
            if skip_bad and (not torch.isfinite(x).all() or not torch.isfinite(y).all()):
                n_skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg.get("amp", False))):
                pred = model(x)
                loss = -(y * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
            if skip_bad and (not torch.isfinite(loss)):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            if skip_bad and (not torch.isfinite(gn)):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.detach())
            n_batches += 1
            if cfg.get("smoke") and n_batches >= 20:
                break
        if n_batches == 0:
            raise RuntimeError(f"epoch {epoch}: all batches non-finite (skipped={n_skipped})")
        tr_loss /= n_batches
        if not np.isfinite(tr_loss):
            raise RuntimeError(f"epoch {epoch}: non-finite mean train_loss={tr_loss}")

        # Save weights BEFORE annotate so a peak/eval crash does not lose the epoch
        if _is_main():
            # refuse to persist NaN weights
            n_nan = sum(int(torch.isnan(v).sum()) for v in raw_model.state_dict().values() if torch.is_floating_point(v))
            if n_nan > 0:
                raise RuntimeError(f"epoch {epoch}: model has {n_nan} NaN params; refusing checkpoint")
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "n_skipped_batches": n_skipped,
                    "label_order": label_order,
                    "cfg": cfg,
                },
                ckpt / "last.pt",
            )
            (out / f"EPOCH_{epoch}_TRAIN.DONE").write_text(
                json.dumps({"train_loss": tr_loss, "n_skipped_batches": n_skipped}) + "\n"
            )
        if world > 1:
            dist.barrier()

        if _is_main():
            stats = annotate_score(pick_df) if cfg.get("annotate_eval_every_epoch", True) else {"score": -1}
            row = {"epoch": epoch, "train_loss": tr_loss, **stats, "elapsed_h": (time.time() - t0) / 3600}
            history.append(row)
            print(row, flush=True)
            save_json({"history": history, "best_score": best}, hist_path)
            if stats.get("score", -1) >= best:
                best = float(stats["score"])
                bad = 0
                torch.save(
                    {"model": raw_model.state_dict(), "epoch": epoch, "metrics": stats, "label_order": label_order, "cfg": cfg},
                    ckpt / "best.pt",
                )
            else:
                bad += 1
                if bad >= patience:
                    print({"early_stop": epoch, "best": best}, flush=True)
                    break
            (out / f"EPOCH_{epoch}_ANNOTATE.DONE").write_text(json.dumps(row, default=str) + "\n")

        if world > 1:
            # sync early-stop decision
            stop_t = torch.tensor([1 if bad >= patience else 0], device=device)
            dist.broadcast(stop_t, src=0)
            if int(stop_t.item()) == 1:
                break
            dist.barrier()

    if _is_main():
        if cfg.get("final_annotate_full_dev", False) and not cfg.get("smoke"):
            final = annotate_score(dev_df)
            save_json(final, out / "final_dev_annotate.json")
        save_json({"history": history, "best_score": best, "label_order": label_order}, hist_path)
        (out / "TRAIN.DONE").write_text("ok\n")
        print({"saved": str(out), "best": best}, flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
