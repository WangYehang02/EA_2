#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import match_picks
from earthquake.models.phasenet_finetune import AugmentConfig, PhaseNetFinetuneDataset
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.utils import ensure_dir


def split_hash(ids: list[str]) -> str:
    blob = "\n".join(map(str, ids)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phasenet_finetune_debug.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)

    weight = cfg.get("init_weight", "stead")
    if "instance" in str(weight).lower():
        raise SystemExit("Refusing INSTANCE pretrained init for finetune")

    import seisbench.models as sbm

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = sbm.PhaseNet.from_pretrained(weight)
    model.to(device)
    model.train()
    # unfreeze
    for p in model.parameters():
        p.requires_grad_(True)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index" / "noise.parquet")
    train_events = events[events["split"] == "train"].copy()
    val_events = events[events["split"] == "val"].copy()

    n_train = int(cfg.get("n_train_traces", 10000))
    n_val = int(cfg.get("n_val_traces", 2000))
    # event-stratified sample
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    def sample_by_event(df, n):
        parts = []
        events_ids = df["event_id"].astype(str).unique().tolist()
        rng.shuffle(events_ids)
        for eid in events_ids:
            g = df[df["event_id"].astype(str) == eid]
            # oversample low SNR / has S
            if "snr_db" in g.columns:
                g = g.assign(_w=np.where(g["snr_db"].fillna(50) < 5, 2.0, 1.0))
                if "s_arrival_sample" in g.columns:
                    g["_w"] = g["_w"] + g["s_arrival_sample"].notna().astype(float)
                probs = g["_w"].to_numpy()
                probs = probs / probs.sum()
                idx = rng.choice(len(g), size=min(len(g), 5), replace=False, p=probs)
                parts.append(g.iloc[idx])
            else:
                parts.append(g.sample(n=min(len(g), 5), random_state=int(rng.integers(0, 1e6))))
            if sum(len(p) for p in parts) >= n:
                break
        out = pd.concat(parts, ignore_index=True).head(n)
        return out

    train_df = sample_by_event(train_events, n_train)
    val_df = sample_by_event(val_events, n_val)
    # add train noise
    if len(noise):
        n_noise = min(int(cfg.get("n_train_noise", n_train // 5)), len(noise[noise.get("split", "train") == "train"]) if "split" in noise.columns else len(noise))
        noise_train = noise.copy()
        if "split" in noise_train.columns:
            noise_train = noise_train[noise_train["split"] == "train"]
        noise_train = noise_train.sample(n=min(n_noise, len(noise_train)), random_state=int(cfg.get("seed", 0)))
        noise_train = noise_train.assign(is_noise=True, event_id="NOISE", p_arrival_sample=np.nan, s_arrival_sample=np.nan)
        # ensure required cols
        for c in ["sampling_rate_hz"]:
            if c not in noise_train.columns:
                noise_train[c] = 100.0
        train_df = pd.concat([train_df.assign(is_noise=False), noise_train], ignore_index=True)

    out = ensure_dir(artifacts_dir() / "phasenet_finetune")
    train_df.to_parquet(out / "debug_train_index.parquet", index=False)
    val_df.to_parquet(out / "debug_val_index.parquet", index=False)
    meta = {
        "init_weight": weight,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_event_hash": split_hash(sorted(train_df["event_id"].astype(str).unique())),
        "val_event_hash": split_hash(sorted(val_df["event_id"].astype(str).unique())),
        "cfg": cfg,
    }
    save_json(meta, out / "debug_run_meta.json")

    h5_events = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    h5_noise = resolve_instance_root() / "noise" / "Instance_noise.hdf5"
    event_reader = InstanceHDF5Reader(h5_events).open()
    noise_reader = InstanceHDF5Reader(h5_noise).open() if h5_noise.exists() else None

    # Preload waveforms into RAM (≈1–2GB). Persist a local npy cache for restarts.
    cache_dir = ensure_dir(out / "wave_cache")
    cache: dict[str, np.ndarray] = {}
    all_names = list(dict.fromkeys(train_df["trace_name"].astype(str).tolist() + val_df["trace_name"].astype(str).tolist()))
    print({"preloading_waveforms": len(all_names), "cache_dir": str(cache_dir)})
    for name in tqdm(all_names, desc="preload"):
        npy = cache_dir / f"{name.replace('/', '_')}.npy"
        if npy.exists():
            cache[name] = np.load(npy)
            continue
        try:
            wave = event_reader.read_waveform(name)
        except Exception:
            if noise_reader is None:
                raise
            wave = noise_reader.read_waveform(name)
        np.save(npy, wave)
        cache[name] = wave

    def read_fn(name: str):
        return cache[name]
    in_samples = int(getattr(model, "in_samples", 3001) or 3001)
    label_order = "".join(getattr(model, "labels", "PSN") or "PSN")
    print({"in_samples": in_samples, "label_order": label_order, "component_order": getattr(model, "component_order", None)})
    aug_cfg = AugmentConfig(
        noise_std=float(cfg.get("aug_noise_std", 0.02)),
        channel_dropout_prob=float(cfg.get("aug_channel_dropout_prob", 0.05)),
        polarity_flip_prob=float(cfg.get("aug_polarity_flip_prob", 0.1)),
    )
    train_ds = PhaseNetFinetuneDataset(
        train_df,
        read_fn,
        in_samples=in_samples,
        augment=True,
        seed=int(cfg.get("seed", 0)),
        label_order=label_order,
    )
    train_ds.aug_cfg = aug_cfg
    val_ds = PhaseNetFinetuneDataset(
        val_df, read_fn, in_samples=in_samples, augment=False, seed=1, label_order=label_order
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    epochs = int(cfg.get("epochs", 5))
    sched = None
    if cfg.get("cosine_scheduler", True):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history = []
    best_loss = float("inf")
    best_score = -1.0
    ckpt_dir = ensure_dir(out / "checkpoints")

    # Fixed annotate-val subset for checkpoint selection (avoid loss/F1 mismatch)
    pick_n = min(int(cfg.get("val_pick_traces", 256)), len(val_df))
    pick_df = val_df.head(pick_n).copy()
    ref = SeisBenchPhaseNetReference(weight=weight, device=device)
    ref.model = model

    def annotate_val_score() -> dict:
        model.eval()
        pp, tp, ps, ts, srs = [], [], [], [], []
        with torch.no_grad():
            for _, row in pick_df.iterrows():
                wave = cache[str(row.trace_name)]
                out = ref.predict_row(wave, row, remap_to_waveform=True)
                pp.append(float(out["p_pred_sample_on_waveform"]))
                ps.append(float(out["s_pred_sample_on_waveform"]))
                tp.append(float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan)
                ts.append(float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan)
                srs.append(float(row.get("sampling_rate_hz", 100.0)))
        mp = match_picks(np.array(pp), np.array(tp), np.array(srs))
        ms = match_picks(np.array(ps), np.array(ts), np.array(srs))
        score = 0.5 * (float(mp["f1@0.5s"]) + float(ms["f1@0.5s"]))
        return {
            "score": score,
            "p_f1@0.5s": float(mp["f1@0.5s"]),
            "s_f1@0.5s": float(ms["f1@0.5s"]),
            "p_median_ae": float(mp["median_ae"]),
            "s_median_ae": float(ms["median_ae"]),
        }

    def set_train_mode():
        """Train conv weights but keep BatchNorm in eval to preserve pretrained running stats.

        Updating BN running stats on short INSTANCE crops breaks SeisBench annotate() on full traces.
        """
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()

    baseline = annotate_val_score()
    print({"pretrained_val_pick": baseline})
    history.append({"epoch": -1, "train_loss": None, "val_loss": None, **baseline, "tag": "pretrained"})
    best_score = float(baseline["score"])
    torch.save(
        {"model": model.state_dict(), "epoch": -1, "cfg": cfg, "label_order": label_order, "tag": "pretrained"},
        ckpt_dir / "best.pt",
    )

    for epoch in range(epochs):
        set_train_mode()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train:{epoch}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            # PhaseNet.forward already applies softmax -> soft NLL
            loss = -(y * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            tr_loss += float(loss.detach())
        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = -(y * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
                va_loss += float(loss.detach())
        va_loss /= max(len(val_loader), 1)
        pick_stats = annotate_val_score()
        if sched is not None:
            sched.step()
        row_hist = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **pick_stats}
        history.append(row_hist)
        print(row_hist)
        payload = {"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "label_order": label_order}
        torch.save(payload, ckpt_dir / "last.pt")
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(payload, ckpt_dir / "best_loss.pt")
        if pick_stats["score"] >= best_score:
            best_score = pick_stats["score"]
            torch.save(payload, ckpt_dir / "best.pt")

    save_json(
        {"history": history, "best_val_loss": best_loss, "best_pick_score": best_score, "label_order": label_order},
        out / "debug_train_history.json",
    )
    event_reader.close()
    if noise_reader:
        noise_reader.close()
    # basic sanity: loss should drop among real training epochs
    train_hist = [h for h in history if h.get("train_loss") is not None]
    if len(train_hist) >= 2 and train_hist[-1]["train_loss"] >= train_hist[0]["train_loss"] * 0.98:
        print("WARNING: train loss did not clearly decrease")
    print({"saved": str(ckpt_dir), "best_val_loss": best_loss, "best_pick_score": best_score})


if __name__ == "__main__":
    main()
