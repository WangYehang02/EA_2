#!/usr/bin/env python
"""Stage 6 Phase A: leakage-free in-domain PhaseNet on stage6_picker_train / stage6_dev."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import match_picks
from earthquake.models.phasenet_finetune import AugmentConfig, PhaseNetFinetuneDataset
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.splits import load_stage6_event_ids, load_stage6_trace_names
from earthquake.utils import ensure_dir


def event_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Uniform over events, then uniform over traces within event."""
    counts = df.groupby(df["event_id"].astype(str))["trace_name"].transform("count").to_numpy(dtype=float)
    # per-trace weight = 1 / n_traces_in_event  (event-uniform when sampling with replacement)
    return 1.0 / np.maximum(counts, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6/phasenet_indomain.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)

    # Do NOT touch internal_confirm here; only refuse if lists leak into it.
    weight = cfg.get("init_weight", "stead")
    if cfg.get("refuse_instance_pretrained", True) and "instance" in str(weight).lower():
        raise SystemExit("Refusing INSTANCE pretrained init")

    out = ensure_dir(ROOT / cfg.get("out_dir", "artifacts/models/stage6/phasenet"))
    log_dir = ensure_dir(artifacts_dir() / "results" / "stage6" / "logs")
    done = out / "TRAIN.DONE"
    if done.exists() and not cfg.get("smoke"):
        print({"skip": "already done", "path": str(done)})
        return

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained(weight)
    model.to(device)
    label_order = "".join(getattr(model, "labels", "PSN") or "PSN")
    if label_order.upper() != "PSN":
        raise SystemExit(f"Expected PSN labels for STEAD PhaseNet, got {label_order}")

    # Optional ID-B: freeze encoder initially
    if str(cfg.get("variant", "")).upper().startswith("ID-B"):
        for name, p in model.named_parameters():
            if "decoder" not in name.lower() and "out" not in name.lower():
                p.requires_grad_(False)

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index" / "noise.parquet")
    picker_traces = set(load_stage6_trace_names("stage6_picker_train"))
    dev_traces = set(load_stage6_trace_names("stage6_dev"))
    # leakage check vs confirm
    confirm = set(load_stage6_event_ids("stage6_internal_confirm"))
    train_df = events[events["trace_name"].astype(str).isin(picker_traces)].copy()
    dev_df = events[events["trace_name"].astype(str).isin(dev_traces)].copy()
    if set(train_df.event_id.astype(str)) & confirm:
        raise SystemExit("LEAK: picker_train overlaps internal_confirm events")
    if set(dev_df.event_id.astype(str)) & confirm:
        raise SystemExit("LEAK: dev overlaps internal_confirm events")
    if set(train_df.event_id.astype(str)) & set(dev_df.event_id.astype(str)):
        raise SystemExit("LEAK: picker_train overlaps stage6_dev events")

    if not cfg.get("use_all_picker_train_traces", True):
        n_tr = int(cfg.get("n_train_traces", 256))
        n_dv = int(cfg.get("n_dev_traces", 128))
        # event-balanced subsample without reading labels for selection order
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        eids = train_df.event_id.astype(str).unique().tolist()
        rng.shuffle(eids)
        parts = []
        for eid in eids:
            g = train_df[train_df.event_id.astype(str) == eid]
            parts.append(g.sample(n=min(len(g), 3), random_state=int(rng.integers(1e9))))
            if sum(len(p) for p in parts) >= n_tr:
                break
        train_df = pd.concat(parts, ignore_index=True).head(n_tr)
        dev_df = dev_df.sample(n=min(n_dv, len(dev_df)), random_state=int(cfg.get("seed", 0)))

    # noise
    n_noise = int(cfg.get("n_train_noise", max(1, int(len(train_df) * float(cfg.get("noise_ratio", 0.2))))))
    noise_train = noise.copy()
    if "split" in noise_train.columns:
        noise_train = noise_train[noise_train["split"] == "train"]
    noise_train = noise_train.sample(n=min(n_noise, len(noise_train)), random_state=int(cfg.get("seed", 0)))
    noise_train = noise_train.assign(is_noise=True, event_id="NOISE", p_arrival_sample=np.nan, s_arrival_sample=np.nan)
    train_df = pd.concat([train_df.assign(is_noise=False), noise_train], ignore_index=True)

    train_df.to_parquet(out / "train_index.parquet", index=False)
    dev_df.to_parquet(out / "dev_index.parquet", index=False)
    save_json(
        {
            "n_train_traces": int((~train_df.is_noise).sum()) if "is_noise" in train_df.columns else len(train_df),
            "n_train_noise": int(train_df.is_noise.sum()) if "is_noise" in train_df.columns else 0,
            "n_dev": len(dev_df),
            "label_order": label_order,
            "cfg": cfg,
        },
        out / "run_meta.json",
    )

    if args.dry_run:
        print({"dry_run": True, "out": str(out)})
        return

    h5_events = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    h5_noise = resolve_instance_root() / "noise" / "Instance_noise.hdf5"
    event_reader = InstanceHDF5Reader(h5_events).open()
    noise_reader = InstanceHDF5Reader(h5_noise).open() if h5_noise.exists() else None
    cache_dir = ensure_dir(out / "wave_cache")
    cache: dict[str, np.ndarray] = {}

    def load_wave(name: str) -> np.ndarray:
        if name in cache:
            return cache[name]
        npy = cache_dir / f"{name.replace('/', '_')}.npy"
        if npy.exists():
            w = np.load(npy)
            cache[name] = w
            return w
        try:
            w = event_reader.read_waveform(name)
        except Exception:
            w = noise_reader.read_waveform(name)
        np.save(npy, w)
        cache[name] = w
        return w

    names = list(dict.fromkeys(train_df.trace_name.astype(str).tolist() + dev_df.trace_name.astype(str).tolist()))
    print({"preloading": len(names)}, flush=True)
    for name in tqdm(names, desc="preload"):
        load_wave(name)

    in_samples = int(getattr(model, "in_samples", 3001) or 3001)
    aug = AugmentConfig(
        noise_std=float(cfg.get("aug_noise_std", 0.02)),
        channel_dropout_prob=float(cfg.get("aug_channel_dropout_prob", 0.05)),
        polarity_flip_prob=float(cfg.get("aug_polarity_flip_prob", 0.1)),
        amp_scale=(float(cfg.get("aug_amp_scale_min", 0.5)), float(cfg.get("aug_amp_scale_max", 1.5))),
    )
    train_ds = PhaseNetFinetuneDataset(
        train_df,
        load_wave,
        in_samples=in_samples,
        sigma_s=float(cfg.get("sigma_s", 0.1)),
        augment=True,
        seed=int(cfg.get("seed", 0)),
        label_order=label_order,
    )
    train_ds.aug_cfg = aug
    # sampler: event-balanced on non-noise; noise kept with natural weight
    w = event_balanced_weights(train_df)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(train_df), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 32)), sampler=sampler, num_workers=0)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 50))
    warmup = int(cfg.get("warmup_epochs", 0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1)) if cfg.get("cosine_scheduler", True) else None
    scaler = GradScaler(enabled=bool(cfg.get("amp", False)))

    ref = SeisBenchPhaseNetReference(weight=weight, device=device)
    ref.model = model
    pick_n = min(int(cfg.get("annotate_eval_max_traces", 800)), len(dev_df))
    pick_df = dev_df.head(pick_n).copy()

    def annotate_score(df: pd.DataFrame) -> dict:
        model.eval()
        pp, tp, ps, ts, srs = [], [], [], [], []
        with torch.no_grad():
            for _, row in df.iterrows():
                wave = load_wave(str(row.trace_name))
                outp = ref.predict_row(wave, row, remap_to_waveform=True)
                pp.append(float(outp["p_pred_sample_on_waveform"]))
                ps.append(float(outp["s_pred_sample_on_waveform"]))
                tp.append(float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan)
                ts.append(float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan)
                srs.append(float(row.sampling_rate_hz))
        mp = match_picks(np.array(pp), np.array(tp), np.array(srs))
        ms = match_picks(np.array(ps), np.array(ts), np.array(srs))
        return {
            "p_f1@0.5": float(mp["f1@0.5s"]),
            "s_f1@0.5": float(ms["f1@0.5s"]),
            "s_f1@0.1": float(ms["f1@0.1s"]),
            "s_e2e_p95": float(ms["e2e_p95_ae"]),
            "score": float(ms["f1@0.5s"]),
        }

    history = []
    baseline = annotate_score(pick_df)
    history.append({"epoch": -1, **baseline, "tag": "pretrained"})
    best_score = float(baseline["score"])
    ckpt_dir = ensure_dir(out / "checkpoints")
    torch.save({"model": model.state_dict(), "epoch": -1, "label_order": label_order, "cfg": cfg, "tag": "pretrained"}, ckpt_dir / "best.pt")
    bad_epochs = 0
    patience = int(cfg.get("early_stopping_patience", 8))
    t0 = time.time()

    for epoch in range(epochs):
        # ID-B unfreeze after epoch 5
        if str(cfg.get("variant", "")).upper().startswith("ID-B") and epoch == 5:
            for p in model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))

        set_train_bn_eval(model)
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train:{epoch}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg.get("amp", False))):
                pred = model(x)
                loss = -(y * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.detach())
        tr_loss /= max(len(train_loader), 1)
        if epoch >= warmup and sched is not None:
            sched.step()

        stats = annotate_score(pick_df) if cfg.get("annotate_eval_every_epoch", True) else {"score": -1}
        row = {"epoch": epoch, "train_loss": tr_loss, **stats, "elapsed_h": (time.time() - t0) / 3600}
        history.append(row)
        print(row, flush=True)
        save_json({"history": history}, out / "train_history.json")
        torch.save({"model": model.state_dict(), "epoch": epoch, "label_order": label_order, "cfg": cfg}, ckpt_dir / "last.pt")

        if stats.get("score", -1) >= best_score:
            best_score = float(stats["score"])
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "label_order": label_order, "cfg": cfg, "metrics": stats}, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print({"early_stop": epoch, "best_score": best_score}, flush=True)
                break

    # final full-dev annotate optional
    if cfg.get("final_annotate_full_dev", False):
        final = annotate_score(dev_df)
        save_json(final, out / "final_dev_annotate.json")
        history.append({"epoch": "final_full_dev", **final})

    save_json({"history": history, "best_score": best_score, "label_order": label_order}, out / "train_history.json")
    done.write_text("ok\n")
    event_reader.close()
    if noise_reader:
        noise_reader.close()
    print({"saved": str(out), "best_score": best_score}, flush=True)


if __name__ == "__main__":
    main()
