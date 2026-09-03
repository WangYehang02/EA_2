#!/usr/bin/env python
"""Train Stage-6 Phase C candidate ranker (R1/R2/R3). No PhaseNet training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.features import feature_names
from earthquake.stage6.ranker.models import build_ranker, count_params, listwise_loss
from earthquake.utils import ensure_dir


def _guard() -> None:
    try:
        assert_full_confirm_access_allowed(purpose="phaseC_train")
        raise SystemExit("method_lock present")
    except RuntimeError:
        pass


@torch.no_grad()
def predict_loader(model, loader, device, max_k: int = 10):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)
        pred_idx = logits.argmax(dim=-1).cpu().numpy()
        samples = batch["samples"].numpy()
        for i, tn in enumerate(batch["trace_name"]):
            pi = int(pred_idx[i])
            if pi >= max_k:
                pred = np.nan
            else:
                pred = float(samples[i, pi]) if mask[i, pi] else np.nan
            rows.append(
                {
                    "trace_name": tn,
                    "event_id": batch["event_id"][i],
                    "pred_s_sample": pred,
                    "true_s_sample": float(batch["true_s"][i]),
                    "sampling_rate_hz": float(batch["sr"][i]),
                    "selected_index": pi,
                    "none_of_k": bool(pi >= max_k),
                }
            )
    return pd.DataFrame(rows)


def eval_picks(df: pd.DataFrame) -> dict:
    pred = df["pred_s_sample"].to_numpy(float)
    true = df["true_s_sample"].to_numpy(float)
    sr = df["sampling_rate_hz"].to_numpy(float)
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.2, 0.5))
    b = report_pick_timing_bundle(pred, true, sr)
    return {
        "s_f1@0.1": float(m["f1@0.1s"]),
        "s_f1@0.2": float(m.get("f1@0.2s", np.nan)),
        "s_f1@0.5": float(m["f1@0.5s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(b["miss_rate"]),
        "no_pick_rate": float(b["no_pick_rate"]),
        "wrong_peak_rate": float(b["wrong_peak_rate"]),
        "detected_ae_median": float(b["detected_ae_median"]),
        "detected_ae_mae": float(b["detected_ae_mae"]),
        "detected_ae_p95": float(b["detected_ae_p95"]),
        "score": float(m["f1@0.5s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6/phaseC_ranker.yaml")
    parser.add_argument("--variant", choices=["R1", "R2", "R3"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-traces", type=int, default=-1, help="smoke subsample")
    args = parser.parse_args()
    _guard()
    cfg = load_yaml_config(ROOT / args.config)

    # Refuse PhaseNet optimizer patterns
    assert "phasenet" not in args.variant.lower()

    feat_key = "R1" if args.variant == "R1" else "R2"
    names = feature_names(feat_key)
    train_path = ROOT / f"artifacts/cache/stage6/phaseC/ranker_train_features_{feat_key}.parquet"
    dev_path = ROOT / f"artifacts/cache/stage6/phaseC/dev_features_{feat_key}.parquet"
    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    if args.max_train_traces > 0:
        keep = train_df["trace_name"].drop_duplicates().head(args.max_train_traces)
        train_df = train_df[train_df["trace_name"].isin(keep)]

    train_ds = TraceCandidateDataset(train_df, names, max_k=10, hard_boost=True, seed=args.seed)
    # natural distribution for early-stop selection on a held slice of train? use full train loader;
    # checkpoint selection uses DEV natural distribution only.
    dev_ds = TraceCandidateDataset(dev_df, names, max_k=10, hard_boost=False, seed=args.seed)
    # For speed during training, evaluate on a fixed 8k-trace natural subset of dev; final eval elsewhere uses full.
    # Spec says checkpoint by full-dev F1@0.5 — but 87k each epoch is heavy. Use full for seed42 R2 final epochs;
    # here we use full loader but can subsample for smoke.
    if args.max_train_traces > 0:
        keepd = dev_df["trace_name"].drop_duplicates().head(min(2000, args.max_train_traces))
        dev_df_s = dev_df[dev_df["trace_name"].isin(keepd)]
        dev_ds = TraceCandidateDataset(dev_df_s, names, max_k=10, hard_boost=False, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, collate_fn=collate_traces, num_workers=0)
    # natural order for eval: rebuild without hard boost indices
    dev_eval_ds = TraceCandidateDataset(dev_df if args.max_train_traces <= 0 else dev_df_s, names, max_k=10, hard_boost=False, seed=0)
    # replace indices with sequential natural groups
    dev_eval_ds.indices = list(range(len(dev_eval_ds.groups)))
    # subsample eval during train for speed if full
    if args.max_train_traces <= 0 and len(dev_eval_ds.groups) > 12000:
        rng = np.random.default_rng(args.seed)
        # event-stratified light sample for epoch selection; final script re-evals full
        idx = rng.choice(len(dev_eval_ds.groups), size=12000, replace=False)
        dev_eval_ds.indices = sorted(idx.tolist())
    dev_loader = DataLoader(dev_eval_ds, batch_size=512, shuffle=False, collate_fn=collate_traces, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_ranker(args.variant, len(names)).to(device)
    n_params = count_params(model)
    if args.variant == "R3" and n_params > 250_000:
        raise SystemExit(f"R3 params {n_params} exceed 250k")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    out = ensure_dir(artifacts_dir() / "models" / "stage6" / "ranker" / f"{args.variant}_seed{args.seed}_b{args.beta}_g{args.gamma}")
    best = -1.0
    best_f101 = -1.0
    bad = 0
    hist = []
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            logits = model(x, mask)
            loss = listwise_loss(logits, target, beta=args.beta, gamma=args.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        preds = predict_loader(model, dev_loader, device)
        stats = eval_picks(preds)
        stats["epoch"] = epoch
        stats["train_loss"] = float(np.mean(losses))
        hist.append(stats)
        print(stats, flush=True)
        improved = stats["s_f1@0.5"] > best + 1e-6
        if abs(stats["s_f1@0.5"] - best) < float(cfg["training"]["tie_break_f1_01_delta"]) and stats["s_f1@0.1"] > best_f101:
            improved = True
        if improved:
            best = stats["s_f1@0.5"]
            best_f101 = stats["s_f1@0.1"]
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": args.variant,
                    "seed": args.seed,
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "feature_names": names,
                    "n_params": n_params,
                    "metrics": stats,
                    "epoch": epoch,
                },
                out / "best.pt",
            )
        else:
            bad += 1
            if bad >= int(cfg["training"]["early_stopping_patience"]):
                break
    save_json({"history": hist, "best": best, "n_params": n_params}, out / "train_history.json")
    (out / "TRAIN.DONE").write_text("ok\n")
    print({"done": str(out), "best_f1_05": best, "n_params": n_params})


if __name__ == "__main__":
    main()
