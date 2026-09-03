#!/usr/bin/env python
"""DKPN clean v2: partial-label crops, new run dir, never resume buggy seed42."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage10.crop_v2 import IN_SAMPLES
from earthquake.stage10.dataset_v2 import (
    DKPNPartialCropDataset,
    EventStationBalancedSampler,
    annotate_online_catalog,
    expand_crop_catalog,
)
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.nonfinite import NonfiniteError, check_finite
from earthquake.stage10.dkpn_picks import forced_choice_peak, s_f1_at_tolerance
from earthquake.stage10.gpu_policy import MIN_GPUS, idle_gpu_indices, launch_block_reason, status_dict, training_process_running
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.checkpoint_policy import (
    FIXED_0P2_METRIC,
    OFFICIAL_PROBABILITY_THRESHOLD,
    best_metric_filename,
    checkpoint_metadata,
    epoch_checkpoint_name,
    is_valid_best,
)
from earthquake.utils import ensure_dir

CFG_PATH = ROOT / "configs/stage10/dkpn_clean_v2_seed42.yaml"
BUGGY = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean/train_seed42")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _meta_picker(include_p_only: bool, noise_ratio: float, seed: int, max_event: int | None = None) -> pd.DataFrame:
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    picker = set(load_full_trace_names("stage6_picker_train"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    dev = set(load_full_event_ids("stage6_dev"))
    ev = events[events["trace_name"].astype(str).isin(picker)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK picker_train ∩ confirm")
    if set(ev["event_id"].astype(str)) & dev:
        raise SystemExit("LEAK picker_train ∩ dev")
    p = pd.to_numeric(ev["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ps = ev[p.notna() & s.notna()].copy()
    po = ev[p.notna() & s.isna()].copy()
    ps["is_noise"] = False
    po["is_noise"] = False
    parts = [ps]
    if include_p_only:
        parts.append(po)
    tr = pd.concat(parts, ignore_index=True)
    if max_event is not None:
        keep_e = tr["event_id"].drop_duplicates().head(max_event)
        tr = tr[tr["event_id"].isin(keep_e)].copy()
    n_noise = int(min(len(noise), max(1, round(len(tr) * noise_ratio))))
    nsel = noise.sample(n=n_noise, random_state=seed).copy()
    nsel["is_noise"] = True
    if "event_id" not in nsel.columns:
        nsel["event_id"] = "NOISE"
    if "station_id" not in nsel.columns:
        nsel["station_id"] = "NOISE"
    if "sampling_rate_hz" not in nsel.columns:
        nsel["sampling_rate_hz"] = 100.0
    cols = [c for c in tr.columns if c in nsel.columns or c in tr.columns]
    # align
    for c in tr.columns:
        if c not in nsel.columns:
            nsel[c] = np.nan
    out = pd.concat([tr, nsel[tr.columns]], ignore_index=True)
    return out


def freeze_val_subset(n_traces: int, seed: int, out: Path) -> pd.DataFrame:
    """Event-balanced S-labelled Stage6 dev traces. No confirm."""
    path = out / "val_subset.parquet"
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    dev_tr = set(load_full_trace_names("stage6_dev"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    ev = events[events["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK val ∩ confirm")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ev = ev[s.notna()].copy()
    rng = np.random.default_rng(seed)
    # one random S-labelled trace per event, then cap
    picks = []
    for eid, g in ev.groupby("event_id", sort=False):
        picks.append(g.iloc[int(rng.integers(0, len(g)))])
    sub = pd.DataFrame(picks)
    if len(sub) > n_traces:
        sub = sub.sample(n=n_traces, random_state=seed)
    sub = sub.reset_index(drop=True)
    sub["is_noise"] = False
    sub.to_parquet(path, index=False)
    (out / "val_subset.sha256").write_text(sha256_file(path) + "\n")
    save_json(
        {
            "n_traces": int(len(sub)),
            "n_events": int(sub["event_id"].nunique()),
            "sha256": sha256_file(path),
            "source": "stage6_dev_S_labelled_event_balanced",
            "confirm_read": False,
        },
        out / "val_subset_manifest.json",
    )
    return sub


def collate(batch):
    keys_t = ["x", "p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask", "vis_p", "vis_s", "p_c", "s_c"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_t}
    out["crop_kind"] = [b["crop_kind"] for b in batch]
    out["trace_name"] = [b["trace_name"] for b in batch]
    out["event_id"] = [b["event_id"] for b in batch]
    out["is_noise"] = torch.tensor([b["is_noise"] for b in batch])
    return out


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def batch_loss(model, batch, device, amp: bool, optimizer_step=None):
    """AMP wraps forward only. Partial-label NLL is always FP32 (autocast off)."""
    x = batch["x"].to(device, non_blocking=True)
    if not torch.isfinite(x).all():
        names = batch.get("trace_name") or ["?"]
        bad = (~torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=-1)).nonzero(as_tuple=False).view(-1).tolist()
        ids = [str(names[i]) for i in bad]
        raise NonfiniteError(
            "train_input",
            f"NaN/Inf input traces={ids}",
            model=model,
            batch=batch,
            device=device,
            optimizer_step=optimizer_step,
        )
    kw = {k: batch[k].to(device, non_blocking=True) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    for k, t in kw.items():
        if not torch.isfinite(t).all():
            raise NonfiniteError(
                "train_input",
                f"NaN/Inf {k}",
                model=model,
                batch=batch,
                device=device,
                optimizer_step=optimizer_step,
            )
    amp_on = bool(amp) and device.type == "cuda"
    if amp_on:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = dkpn_logits(model, x)
    else:
        logits = dkpn_logits(model, x)
    if not torch.isfinite(logits).all():
        raise NonfiniteError(
            "train_forward",
            "non-finite train logits",
            model=model,
            batch=batch,
            device=device,
            optimizer_step=optimizer_step,
        )
    ctx = torch.autocast("cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with ctx:
        loss = partial_label_nll(logits.float(), **kw)
    check_finite(
        loss,
        "train_loss",
        "non-finite train loss",
        model=model,
        batch=batch,
        device=device,
        optimizer_step=optimizer_step,
    )
    return loss, logits


def loc_acc(logits, p_c, s_c, vis_p, vis_s, tol=20):
    pred_p = logits[:, 0, :].argmax(-1)
    pred_s = logits[:, 1, :].argmax(-1)
    accp = accs = []
    vp = vis_p.bool()
    vs = vis_s.bool()
    ok_p = ok_s = tot_p = tot_s = 0
    for i in range(len(pred_p)):
        if bool(vp[i]) and float(p_c[i]) >= 0:
            tot_p += 1
            ok_p += int(abs(int(pred_p[i]) - float(p_c[i])) <= tol)
        if bool(vs[i]) and float(s_c[i]) >= 0:
            tot_s += 1
            ok_s += int(abs(int(pred_s[i]) - float(s_c[i])) <= tol)
    return {
        "p_acc": ok_p / max(tot_p, 1),
        "s_acc": ok_s / max(tot_s, 1),
        "n_vis_p": tot_p,
        "n_vis_s": tot_s,
    }


def make_loader(cat, wave, *, batch, workers, augment, seed, shuffle, balanced, num_samples=None, drop_last=False):
    def read_fn(name, is_noise=False):
        return wave.read(name, is_noise=is_noise)

    ds = DKPNPartialCropDataset(cat, read_fn, augment=augment, seed=seed)
    sampler = EventStationBalancedSampler(cat, num_samples=num_samples or len(cat), seed=seed) if balanced else None
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    ), ds


def _eval_dev_s_f1(model, loader, device) -> dict:
    model.eval()
    preds, trues, vis = [], [], []
    vloss = []
    with torch.no_grad():
        for batch in loader:
            loss, logits = batch_loss(model, batch, device, amp=False)
            vloss.append(float(loss.cpu()))
            pr = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for i in range(pr.shape[0]):
                preds.append(forced_choice_peak(pr[i, 1], thr=0.2))
                trues.append(float(batch["s_c"][i]))
                vis.append(bool(batch["vis_s"][i] > 0.5 and float(batch["s_c"][i]) >= 0))
    pred_arr = np.array([(-1 if p is None else p) for p in preds], dtype=float)
    pred_arr[pred_arr < 0] = np.nan
    metrics = s_f1_at_tolerance(pred_arr, np.asarray(trues), np.asarray(vis), tol_samples=50)
    metrics["val_loss"] = float(np.mean(vloss)) if vloss else float("nan")
    return metrics


def _run_throughput(cfg, wave, device, seed, out, workers) -> None:
    from earthquake.stage10.gpu_policy import THROUGHPUT_GPUS, THROUGHPUT_STEPS

    art = ensure_dir(ROOT / "artifacts/results/stage10")
    idle = idle_gpu_indices()
    n = len(idle)
    n_steps = int(cfg.get("throughput_steps", THROUGHPUT_STEPS))
    report = {
        "mode": "throughput",
        "utc": datetime.now(timezone.utc).isoformat(),
        "confirm_read": False,
        "idle_gpu_indices": idle,
        "n_idle": n,
        "requested_steps": n_steps,
        "requested_gpus": THROUGHPUT_GPUS,
        "runs": {},
        "skipped_reason": None,
        "never_started_full_train": True,
    }
    if n < 1 or device.type != "cuda":
        report["skipped_reason"] = "no_truly_idle_gpu_or_not_cuda"
        report["note"] = (
            "Idle requires memory.used<500MiB AND util≈0 AND no compute processes. "
            "Bench uses exactly 1 idle GPU and 50 steps."
        )
        save_json(report, art / "dkpn_v2_throughput.json")
        print(json.dumps(report, indent=2))
        return
    meta = _meta_picker(True, float(cfg["noise_ratio"]), seed)
    batch_sz = int(cfg["batch_size"])
    cat = annotate_online_catalog(meta).head(max(512, n_steps * batch_sz))
    loader, ds = make_loader(
        cat,
        wave,
        batch=batch_sz,
        workers=min(workers, 2),
        augment=False,
        seed=seed,
        shuffle=False,
        balanced=False,
        num_samples=n_steps * batch_sz,
    )
    ds.set_virtual_epoch(0)
    model = build_dkpn_random().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    t_gpu = 0.0
    n_win = 0
    steps = 0
    t0 = time.time()
    for batch in loader:
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, batch, device, amp=bool(cfg.get("amp", True)))
        loss.backward()
        opt.step()
        t_gpu += time.time() - t1
        n_win += int(batch["x"].shape[0])
        steps += 1
        if steps >= n_steps:
            break
    dt = time.time() - t0
    st = status_dict(out_dir=out)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", str(idle[0])).split(",")[0].strip()
    try:
        phys = int(vis)
    except ValueError:
        phys = idle[0]
    win_s = n_win / max(dt, 1e-6)
    report["runs"]["1"] = {
        "n_gpus": 1,
        "cuda_visible_devices": vis,
        "physical_gpu": phys,
        "steps": steps,
        "windows": n_win,
        "sec": dt,
        "windows_per_sec": win_s,
        "windows_per_sec_per_gpu": win_s,
        "gpu_step_fraction": t_gpu / max(dt, 1e-6),
        "gpu_utilization_pct": [g["utilization_pct"] for g in st["gpus"] if g["index"] == phys],
        "memory_used_mib": [g["memory_used_mib"] for g in st["gpus"] if g["index"] == phys],
    }
    n_win_epoch = max(1, len(annotate_online_catalog(_meta_picker(True, float(cfg["noise_ratio"]), seed))))
    sec_ep = n_win_epoch / max(win_s, 1e-9)
    report["eta_online_sampling"] = {
        "1gpu": {
            "virtual_epoch_hours": sec_ep / 3600.0,
            "crop_cycle_hours": 3 * sec_ep / 3600.0,
            "windows_per_virtual_epoch": n_win_epoch,
            "note": "extrapolated from 50-step 1-GPU bench; 4/8 GPU not measured",
        }
    }
    save_json(report, art / "dkpn_v2_throughput.json")
    print(json.dumps(report, indent=2))


def _pilot_gate(history, first500, kinds_seen, partial_seen) -> tuple[bool, list[str]]:
    reasons = []
    if not history:
        return False, ["no_history"]
    losses = [h["train_loss"] for h in history]
    if first500 is None or not np.isfinite(first500):
        reasons.append("no_first500")
    elif losses[-1] >= first500 * 0.98:
        reasons.append("train_loss_not_down_vs_first500")
    psn = history[-1].get("prob_mean_psn") or [0, 0, 1]
    if psn[0] < 1e-6 or psn[1] < 1e-6:
        reasons.append("ps_collapsed")
    f1s = [h.get("val_s_f1_0p5", 0) for h in history]
    init_f1 = history[0].get("init_s_f1_0p5", 0)
    if f1s[-1] <= init_f1:
        reasons.append("dev_s_f1_not_above_init")
    if len(f1s) >= 3 and f1s[2] < f1s[0]:
        reasons.append("epoch3_below_epoch1")
    if history[-1].get("grad_norm", 0) <= 0 or not np.isfinite(history[-1].get("grad_norm", 0)):
        reasons.append("bad_grad")
    if history[-1].get("lr", 0) <= 0:
        reasons.append("bad_lr")
    if any(not np.isfinite(h["train_loss"]) for h in history):
        reasons.append("nan_loss")
    need_kinds = {"p_centered", "s_centered", "background"}
    if not need_kinds.issubset(set(kinds_seen)):
        reasons.append(f"missing_crop_kinds:{sorted(need_kinds - set(kinds_seen))}")
    if not partial_seen:
        reasons.append("partial_label_not_seen")
    return (len(reasons) == 0), reasons


def _run_corrected_train(mode, cfg, wave, device, seed, out, workers, root: Path) -> None:
    meta = _meta_picker(True, float(cfg["noise_ratio"]), seed)
    cat = annotate_online_catalog(meta)
    cat.to_parquet(out / "train_online_catalog.parquet", index=False)
    val_meta = freeze_val_subset(int(cfg["val_traces"]), seed, out)
    val_cat = annotate_online_catalog(val_meta)
    hashes = {
        "run": cfg["run_name"],
        "seed": seed,
        "online_crop": True,
        "windows_per_virtual_epoch": int(len(cat)),
        "crop_cycle_virtual_epochs": 3,
        "config_sha256": sha256_file(root / "configs/stage10/dkpn_clean_v2_seed42.yaml"),
        "partial_label_sha256": sha256_file(root / "src/earthquake/stage10/partial_label.py"),
        "dataset_v2_sha256": sha256_file(root / "src/earthquake/stage10/dataset_v2.py"),
        "init": "random",
        "buggy_resume": False,
        "confirm_read": False,
    }
    save_json(hashes, out / "run_hashes.json")
    model = build_dkpn_random().to(device)
    ensure_dir(out / "checkpoints")
    torch.save({"model": model.state_dict(), "seed": seed, "epoch": -1}, out / "checkpoints" / "init.pt")
    nvis = torch.cuda.device_count() if device.type == "cuda" else 1
    if nvis >= 2:
        model = torch.nn.DataParallel(model)
        hashes["data_parallel_ngpu"] = int(nvis)
        hashes["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        save_json(hashes, out / "run_hashes.json")
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=float(cfg["lr_factor"]), patience=int(cfg["lr_patience"]))
    history = []
    best_loss = 1e9
    best_metric = -1.0
    bad = 0
    optimizer_steps = 0
    windows_seen = 0
    first500 = []
    kinds_seen: set[str] = set()
    partial_seen = False
    batch_sz = int(cfg["batch_size"])
    (out / "TRAIN.RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    (out / "TRAIN.PID").write_text(str(os.getpid()))
    max_ep = 3 if mode == "pilot" else int(cfg["max_epoch"])
    continue_after = mode == "train"
    try:
        val_loader, val_ds = make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
        val_ds.set_virtual_epoch(1)  # S-centered so S labels are in-window
        init_metrics = _eval_dev_s_f1(model, val_loader, device)
        for ep in range(max_ep if not continue_after else int(cfg["max_epoch"])):
            if continue_after and ep == 3 and not (out / "PILOT.PASSED").is_file():
                # train mode always gates after one crop cycle
                pass
            train_loader, ds = make_loader(
                cat, wave, batch=batch_sz, workers=workers, augment=True, seed=seed + ep, shuffle=False, balanced=True, num_samples=len(cat)
            )
            ds.set_virtual_epoch(ep)
            model.train()
            t0 = time.time()
            tr_losses = []
            gnorms = []
            vis_p = vis_s = 0
            kinds: dict[str, int] = {}
            pmeans = []
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                loss, logits = batch_loss(model, batch, device, amp=bool(cfg.get("amp", True)))
                loss.backward()
                g = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
                gnorms.append(float(g.detach().cpu()))
                opt.step()
                optimizer_steps += 1
                windows_seen += int(batch["x"].shape[0])
                tr_losses.append(float(loss.detach().cpu()))
                if optimizer_steps <= 500:
                    first500.append(tr_losses[-1])
                vis_p += int(batch["vis_p"].sum())
                vis_s += int(batch["vis_s"].sum())
                for k in batch["crop_kind"]:
                    kinds[k] = kinds.get(k, 0) + 1
                    kinds_seen.add(k)
                if int((batch["not_p"] + batch["not_s"]).sum().item()) > 0:
                    partial_seen = True
                with torch.no_grad():
                    pr = torch.softmax(logits.float(), dim=1)
                    pmeans.append([float(pr[:, i].mean().cpu()) for i in range(3)])
            val_loader, val_ds = make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
            val_ds.set_virtual_epoch(1)
            vm = _eval_dev_s_f1(model, val_loader, device)
            valid = is_valid_best(metric=vm["f1"], init_metric=init_metrics["f1"])
            rec = {
                "virtual_epoch": ep,
                "crop_cycles": (ep + 1) / 3.0,
                "optimizer_steps": optimizer_steps,
                "windows_seen": windows_seen,
                "train_loss": float(np.mean(tr_losses)),
                "val_loss": vm["val_loss"],
                "val_s_f1_0p5": vm["f1"],
                "val_s_f1_fixed0p2": vm["f1"],
                "init_s_f1_0p5": init_metrics["f1"],
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": float(np.mean(gnorms)),
                "n_vis_p": vis_p,
                "n_vis_s": vis_s,
                "kinds": kinds,
                "prob_mean_psn": list(np.mean(pmeans, axis=0)),
                "sec": time.time() - t0,
                "utc": datetime.now(timezone.utc).isoformat(),
                "confirm_read": False,
                "valid_best_fixed0p2": valid,
                "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                "metric_name_fixed0p2": FIXED_0P2_METRIC,
                "metric_name_calibrated": "val_s_f1_calibrated",
            }
            history.append(rec)
            save_json({"history": history, "hashes": hashes}, out / "train_history.json")
            raw = _raw_model(model)
            epoch_meta = checkpoint_metadata(
                virtual_epoch=ep,
                metric_name=FIXED_0P2_METRIC,
                metric_value=vm["f1"],
                init_metric=init_metrics["f1"],
                probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                valid_best=False,
                extra={"kind": "epoch_snapshot", "val_loss": vm["val_loss"]},
            )
            torch.save(
                {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "opt": opt.state_dict(),
                    "seed": seed,
                    "optimizer_steps": optimizer_steps,
                    "windows_seen": windows_seen,
                    **epoch_meta,
                },
                out / "checkpoints" / epoch_checkpoint_name(ep),
            )
            torch.save(
                {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "opt": opt.state_dict(),
                    "seed": seed,
                    "optimizer_steps": optimizer_steps,
                    "windows_seen": windows_seen,
                    **epoch_meta,
                },
                out / "checkpoints" / "last.pt",
            )
            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]
                torch.save({"model": raw.state_dict(), "epoch": ep, "val_loss": best_loss}, out / "checkpoints" / "best_loss.pt")
                bad = 0
            else:
                bad += 1
            if valid:
                best_metric = vm["f1"]
                best_payload = {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "val_s_f1": best_metric,
                    **checkpoint_metadata(
                        virtual_epoch=ep,
                        metric_name=FIXED_0P2_METRIC,
                        metric_value=best_metric,
                        init_metric=init_metrics["f1"],
                        probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                        valid_best=True,
                    ),
                }
                torch.save(best_payload, out / "checkpoints" / best_metric_filename("fixed0p2"))
            save_json(
                {
                    "valid_best": bool(valid and best_metric > 0),
                    "best_metric_fixed0p2": None if best_metric < 0 else best_metric,
                    "init_s_f1_fixed0p2": init_metrics["f1"],
                    "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                    "f1_all_zero_cannot_mint_valid_best": True,
                    "calibrated_best_not_written_here": True,
                },
                out / "checkpoints" / "best_metric_status.json",
            )
            sch.step(vm["val_loss"])
            print(rec, flush=True)
            if ep == 2:
                ok, reasons = _pilot_gate(history, float(np.mean(first500)) if first500 else None, kinds_seen, partial_seen)
                gate = {"ok": ok, "reasons": reasons, "first500_loss": float(np.mean(first500)) if first500 else None}
                save_json(gate, out / "pilot_gate.json")
                if ok:
                    (out / "PILOT.PASSED").write_text(datetime.now(timezone.utc).isoformat())
                    if mode == "pilot":
                        break
                else:
                    (out / "PILOT.FAILED").write_text(json.dumps(gate, indent=2))
                    break
            if (out / "PILOT.PASSED").is_file() and ep >= 3 and bad >= int(cfg["patience"]):
                rec["early_stop"] = True
                break
        if (out / "PILOT.FAILED").is_file():
            pass
        elif mode == "train" and (out / "PILOT.PASSED").is_file() and not (out / "PILOT.FAILED").is_file():
            (out / "TRAIN.DONE").write_text(datetime.now(timezone.utc).isoformat())
        if (out / "TRAIN.RUNNING").exists() and (
            (out / "PILOT.FAILED").is_file() or (out / "TRAIN.DONE").is_file() or mode == "pilot"
        ):
            (out / "TRAIN.RUNNING").unlink()
    except Exception as e:
        (out / "TRAIN.FAILED").write_text(repr(e))
        raise
    finally:
        if (out / "TRAIN.PID").exists():
            (out / "TRAIN.PID").unlink(missing_ok=True)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit32", "mix128", "smoke1000", "train", "pilot", "throughput", "status"], required=True)
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gpus", default=None, help="comma-separated physical GPU ids; skips MIN_GPUS=4 if all listed are idle")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = ensure_dir(Path(cfg["out_dir"]))
    if args.mode == "status":
        print(json.dumps(status_dict(out_dir=out), indent=2))
        return
    if args.mode in {"train", "pilot", "throughput"}:
        idle = idle_gpu_indices()
        already = bool(training_process_running())
        flag = (out / "TRAIN.RUNNING").is_file()
        if args.mode == "throughput":
            reason = "training_already_running" if (already or flag) else None
        elif args.gpus:
            req = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
            occ = [i for i in req if i not in idle]
            if already or flag:
                reason = "training_already_running"
            elif occ:
                reason = f"requested_gpus_occupied_{occ}"
            else:
                reason = None
        else:
            reason = launch_block_reason(
                mode=args.mode,
                n_idle=len(idle),
                already_training=already,
                train_running_flag=flag,
            )
        if reason:
            save_json(
                {"ok": False, "reason": reason, "idle": idle, "mode": args.mode},
                ROOT / "artifacts/results/stage10/dkpn_v2_last_launch_refuse.json",
            )
            print(json.dumps({"refused": reason, "n_idle": len(idle), "idle": idle}))
            raise SystemExit(3 if "need_min_gpus" in reason else 4)
    if args.mode == "train" and (BUGGY / "checkpoints/last.pt").resolve() == (out / "checkpoints/last.pt"):
        raise SystemExit("refusing to write into buggy run dir")
    if out.resolve() == BUGGY.resolve():
        raise SystemExit("refusing to overwrite buggy seed42")

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    workers = args.workers if args.workers is not None else (0 if args.mode not in {"train", "pilot", "throughput"} else int(cfg.get("workers", 2)))

    save_json({"confirm_read": False, "confirm_waveforms": False, "run": cfg["run_name"]}, ROOT / "artifacts/results/stage10/dkpn_v2/confirm_guard.json")

    if args.mode == "overfit32":
        meta = _meta_picker(False, 0.0, seed, max_event=None)
        d = (meta.s_arrival_sample.astype(float) - meta.p_arrival_sample.astype(float))
        short = meta[(~meta.is_noise.astype(bool)) & (d < 1500)].head(32).copy()
        cat = expand_crop_catalog(short)
        cat = cat[cat.crop_kind.isin(["p_centered", "s_centered"])].reset_index(drop=True)
        loader, ds = make_loader(cat, wave, batch=4, workers=0, augment=False, seed=seed, shuffle=True, balanced=False)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        hist = []
        acc0 = None
        for ep in range(80):
            model.train()
            losses = []
            agg = {"p_ok": 0, "s_ok": 0, "p_n": 0, "s_n": 0}
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                loss, logits = batch_loss(model, batch, device, amp=False)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
                a = loc_acc(logits.detach().cpu(), batch["p_c"], batch["s_c"], batch["vis_p"], batch["vis_s"])
                agg["p_ok"] += int(round(a["p_acc"] * a["n_vis_p"]))
                agg["s_ok"] += int(round(a["s_acc"] * a["n_vis_s"]))
                agg["p_n"] += a["n_vis_p"]
                agg["s_n"] += a["n_vis_s"]
            acc = {
                "p_acc": agg["p_ok"] / max(agg["p_n"], 1),
                "s_acc": agg["s_ok"] / max(agg["s_n"], 1),
                "n_vis_p": agg["p_n"],
                "n_vis_s": agg["s_n"],
            }
            if acc0 is None:
                acc0 = acc
            hist.append({"epoch": ep, "loss": float(np.mean(losses)), **acc})
            print(hist[-1], flush=True)
            if hist[-1]["s_acc"] >= 0.95 and hist[-1]["loss"] < hist[0]["loss"] * 0.7:
                break
        ok = bool(hist[-1]["s_acc"] >= 0.95 and hist[-1]["loss"] < hist[0]["loss"] * 0.85)
        report = {"mode": "overfit32", "ok": ok, "n_crops": len(ds), "init": hist[0], "final": hist[-1], "n_epochs": len(hist)}
        save_json(report, out / "overfit32.json")
        save_json(report, ROOT / "artifacts/results/stage10/dkpn_v2/overfit32.json")
        print(json.dumps(report, indent=2))
        if not ok:
            raise SystemExit(2)
        return

    if args.mode == "mix128":
        meta = _meta_picker(True, 0.05, seed)
        d = meta.s_arrival_sample.astype(float) - meta.p_arrival_sample.astype(float)
        short = meta[(~meta.is_noise) & d.notna() & (d < 800)].head(40)
        long = meta[(~meta.is_noise) & d.notna() & (d > 3000)].head(24)
        po = meta[(~meta.is_noise) & meta.p_arrival_sample.notna() & meta.s_arrival_sample.isna()].head(24)
        nz = meta[meta.is_noise.astype(bool)].head(16)
        mix = pd.concat([short, long, po, nz], ignore_index=True)
        cat = expand_crop_catalog(mix)
        loader, ds = make_loader(cat, wave, batch=4, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        # cache one pass tensors for overfit-style mix
        buf = []
        kinds = {}
        for batch in loader:
            buf.append(batch)
            for k in batch["crop_kind"]:
                kinds[k] = kinds.get(k, 0) + 1
        def run_epoch():
            model.train()
            losses = []
            accs = []
            nonzero = {k: 0.0 for k in kinds}
            for batch in buf:
                opt.zero_grad()
                loss, logits = batch_loss(model, batch, device, amp=False)
                loss.backward()
                g = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
                if not torch.isfinite(g):
                    raise RuntimeError("non-finite grad")
                opt.step()
                losses.append(float(loss.detach()))
                accs.append(loc_acc(logits.detach().cpu(), batch["p_c"], batch["s_c"], batch["vis_p"], batch["vis_s"]))
                for k in batch["crop_kind"]:
                    nonzero[k] += float(loss.detach())
            p_acc = np.mean([a["p_acc"] for a in accs if a["n_vis_p"]])
            s_acc = np.mean([a["s_acc"] for a in accs if a["n_vis_s"]])
            return float(np.mean(losses)), float(p_acc), float(s_acc), nonzero
        l0, *_ = run_epoch()
        last = None
        for ep in range(60):
            last = run_epoch()
            if last[1] >= 0.85 and last[2] >= 0.85:
                break
        ok = bool(last[1] >= 0.85 and last[2] >= 0.85 and last[0] < l0 and all(v != 0 for v in last[3].values()))
        # shuffle invariance of join already unit-tested; mix order shuffle: recompute loc on shuffled buf metrics equal is N/A for training
        report = {"mode": "mix128", "ok": ok, "kinds": kinds, "init_loss": l0, "final_loss": last[0], "p_acc": last[1], "s_acc": last[2], "loss_by_kind": last[3]}
        save_json(report, out / "mix128.json")
        save_json(report, ROOT / "artifacts/results/stage10/dkpn_v2/mix128.json")
        print(json.dumps(report, indent=2))
        if not ok:
            raise SystemExit(2)
        return

    if args.mode == "smoke1000":
        meta = _meta_picker(True, 0.2, seed)
        cat = expand_crop_catalog(meta.sample(n=min(2000, len(meta)), random_state=seed))
        loader, ds = make_loader(cat, wave, batch=8, workers=workers, augment=True, seed=seed, shuffle=False, balanced=True, num_samples=8000)
        model = build_dkpn_random().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            if step >= 1000:
                break
            opt.zero_grad(set_to_none=True)
            loss, logits = batch_loss(model, batch, device, amp=device.type == "cuda")
            loss.backward()
            opt.step()
            with torch.no_grad():
                pr = torch.softmax(logits.float(), dim=1)
                hist.append({
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "p_mean": float(pr[:, 0].mean().cpu()),
                    "s_mean": float(pr[:, 1].mean().cpu()),
                    "n_mean": float(pr[:, 2].mean().cpu()),
                    "n_max": float(pr[:, 2].max().cpu()),
                })
            if step % 100 == 0:
                print(hist[-1], flush=True)
                ensure_dir(out / "checkpoints")
                torch.save({"model": model.state_dict(), "step": step, "seed": seed}, out / "checkpoints" / "smoke_last.pt")
        dt = time.time() - t0
        head = np.mean([h["loss"] for h in hist[:50]])
        tail = np.mean([h["loss"] for h in hist[-50:]])
        n_mean_tail = np.mean([h["n_mean"] for h in hist[-50:]])
        s_mean_tail = np.mean([h["s_mean"] for h in hist[-50:]])
        collapse = bool(n_mean_tail > 0.98 or s_mean_tail < 1e-6)
        ok = bool(tail < head and not collapse and np.isfinite(tail))
        report = {"mode": "smoke1000", "ok": ok, "n_steps": len(hist), "sec": dt, "loss_head50": head, "loss_tail50": tail, "n_mean_tail": n_mean_tail, "s_mean_tail": s_mean_tail, "collapse": collapse, "device": str(device)}
        save_json(report, out / "smoke1000.json")
        save_json(report, ROOT / "artifacts/results/stage10/dkpn_v2/smoke1000.json")
        print(json.dumps(report, indent=2))
        if not ok:
            (out / "TRAIN_BLOCKED").write_text("smoke1000 failed\n")
            raise SystemExit(2)
        return

    if args.mode == "throughput":
        _run_throughput(cfg, wave, device, seed, out, workers)
        return

    # corrected train / pilot: one window per trace per virtual epoch
    _run_corrected_train(args.mode, cfg, wave, device, seed, out, workers, ROOT)


if __name__ == "__main__":
    main()
