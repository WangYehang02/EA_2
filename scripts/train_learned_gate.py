#!/usr/bin/env python
"""Train learned scalar gate (or auxiliary candidate ranker)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.gating.cache_io import load_gate_records
from earthquake.gating.candidate_ranker import CandidateRanker
from earthquake.gating.dataset import GateTraceDataset, collate_gate
from earthquake.gating.inference import predict_gate_pick
from earthquake.gating.losses import fallback_bce, listwise_ce, pairwise_margin_loss, soft_target_from_errors
from earthquake.gating.scalar_gate import ScalarGate, gated_candidate_logprobs, history_scores, wave_scores
from earthquake.metrics import match_picks
from earthquake.utils import ensure_dir


def _abcd_weights(records: list[dict], focus=("B", "C")) -> list[float]:
    counts = {}
    for r in records:
        counts[r.get("abcd_class", "D")] = counts.get(r.get("abcd_class", "D"), 0) + 1
    w = []
    for r in records:
        c = r.get("abcd_class", "D")
        base = 1.0 / max(counts.get(c, 1), 1)
        if c == "C":
            base *= 6.0  # rare but critical: PhaseNet wrong, history correct
        elif c == "B":
            base *= 3.0
        elif c == "A":
            base *= 1.0
        else:
            base *= 0.5
        w.append(base)
    return w


def eval_s_f1(model, loader, device, prominence_weight, model_type="scalar_gate"):
    model.eval()
    preds, trues, srs = [], [], []
    gates = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            if model_type == "candidate_ranker":
                # build cand feat
                z = (batch["sample"] - batch["expected"]) / batch["sigma"].clamp_min(1e-6)
                sp = (batch["sample"] - batch.get("sample")[:, :1])  # unused placeholder
                cand_feat = torch.stack(
                    [
                        batch["prob"],
                        batch["prominence"],
                        batch["width"],
                        batch["rank"],
                        z,
                        torch.zeros_like(batch["prob"]),
                    ],
                    dim=-1,
                )
                scores = model(batch["x"], batch["missing"], cand_feat, batch["cand_mask"])
                idx = torch.argmax(scores, dim=-1)
                b = torch.arange(scores.size(0), device=device)
                pick = batch["sample"][b, idx]
                gate = torch.full((scores.size(0),), float("nan"), device=device)
            else:
                out = predict_gate_pick(
                    model,
                    batch["x"],
                    batch["missing"],
                    batch["prob"],
                    batch["prominence"],
                    batch["sample"],
                    batch["cand_mask"],
                    batch["expected"],
                    batch["sigma"],
                    batch["history_available"],
                    prominence_weight=prominence_weight,
                )
                pick = out["pick_sample"]
                gate = out["gate"]
            preds.extend(pick.detach().cpu().numpy().tolist())
            trues.extend(batch["true_s"].view(-1).detach().cpu().numpy().tolist())
            # sampling rate not in batch — recover from errors scale? store 100 default — fix: add to collate
            # Dataset doesn't include sr in tensor; use 100.0 placeholder then fix below
            gates.extend(gate.detach().cpu().numpy().tolist())
    return preds, trues, gates


def train_one(cfg: dict, seed: int, model_type: str = "scalar_gate") -> dict:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cand_dir = artifacts_dir() / "candidates"
    out = ensure_dir(artifacts_dir() / "models" / "learned_gate" / f"{model_type}_seed{seed}")
    min_h = float(cfg.get("min_history", 5))
    mad_t = float(cfg.get("mad_threshold", 1.0))
    train_rec = load_gate_records(cand_dir / "stage3_train.zarr", min_history=min_h, mad_threshold=mad_t)
    val_rec = load_gate_records(cand_dir / "stage3_val.zarr", min_history=min_h, mad_threshold=mad_t)
    if not train_rec:
        raise SystemExit("Empty train zarr — run build_gate_dataset.py first")

    max_k = int(cfg.get("candidate_k", 5))
    pw = float(cfg.get("prominence_weight", 0.1))
    ds_tr = GateTraceDataset(train_rec, max_k=max_k, train=True, corrupt_prob=float(cfg.get("corrupt_prob", 0.3)), seed=seed, prominence_weight=pw)
    ds_va = GateTraceDataset(val_rec, max_k=max_k, train=False, corrupt_prob=0.0, seed=seed, prominence_weight=pw)

    weights = _abcd_weights(train_rec)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg.get("batch_size", 128)), sampler=sampler, collate_fn=collate_gate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=int(cfg.get("batch_size", 128)), shuffle=False, collate_fn=collate_gate, num_workers=0)

    n_feat = len(train_rec[0]["x"])
    if model_type == "candidate_ranker":
        model = CandidateRanker(n_feat, n_cand_features=6, hidden_dim=int(cfg.get("hidden_dim", 64)), dropout=float(cfg.get("dropout", 0.1))).to(device)
    else:
        model = ScalarGate(n_feat, hidden_dim=int(cfg.get("hidden_dim", 64)), dropout=float(cfg.get("dropout", 0.1)), init_gate=float(cfg.get("init_gate", 0.8))).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    temp = float(cfg.get("target_temperature", 0.2))
    # optional temp search on val using frozen init — skip heavy; use config
    hist_path = out / "training_history.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_f1_0.5", "gate_mean", "gate_std"])

    best_f1 = -1.0
    best_state = None
    patience = int(cfg.get("early_stopping_patience", 8))
    bad = 0
    n_params = sum(p.numel() for p in model.parameters())

    for epoch in range(1, int(cfg.get("max_epochs", 50)) + 1):
        model.train()
        losses = []
        for batch in dl_tr:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            opt.zero_grad(set_to_none=True)
            if model_type == "candidate_ranker":
                z = (batch["sample"] - batch["expected"]) / batch["sigma"].clamp_min(1e-6)
                cand_feat = torch.stack(
                    [batch["prob"], batch["prominence"], batch["width"], batch["rank"], z, torch.zeros_like(batch["prob"])],
                    dim=-1,
                )
                scores = model(batch["x"], batch["missing"], cand_feat, batch["cand_mask"])
                # convert to logprob via log_softmax
                neg = torch.finfo(scores.dtype).min / 4
                final = torch.log_softmax(scores.masked_fill(~batch["cand_mask"], neg), dim=-1)
                gate = torch.ones(scores.size(0), 1, device=device)  # unused
            else:
                gate = model(batch["x"], batch["missing"])
                # soft force unavailable toward 1 in forward for loss stability; hard force only at infer
                ha = batch["history_available"].view(-1, 1).float()
                gate_eff = gate * ha + (1.0 - ha)
                w = wave_scores(batch["prob"], batch["prominence"], prominence_weight=pw)
                h = history_scores(batch["sample"], batch["expected"].expand_as(batch["sample"]), batch["sigma"].expand_as(batch["sample"]))
                final = gated_candidate_logprobs(gate_eff, w, h, batch["cand_mask"])
                scores = final

            tgt = soft_target_from_errors(batch["errors"], batch["cand_mask"], temp)
            loss = listwise_ce(final, tgt, batch["cand_mask"])
            loss = loss + float(cfg.get("lambda_pairwise", 0.2)) * pairwise_margin_loss(
                scores, batch["errors"], batch["cand_mask"], margin=float(cfg.get("pairwise_margin", 0.5)), good_thresh_s=float(cfg.get("good_thresh_s", 0.5))
            )
            if model_type != "candidate_ranker":
                loss = loss + float(cfg.get("lambda_fallback", 0.5)) * fallback_bce(gate, batch["bad_history"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.item()))

        # val
        model.eval()
        vlosses = []
        preds, trues, srs, gates = [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dl_va):
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(device)
                if model_type == "candidate_ranker":
                    z = (batch["sample"] - batch["expected"]) / batch["sigma"].clamp_min(1e-6)
                    cand_feat = torch.stack(
                        [batch["prob"], batch["prominence"], batch["width"], batch["rank"], z, torch.zeros_like(batch["prob"])],
                        dim=-1,
                    )
                    scores = model(batch["x"], batch["missing"], cand_feat, batch["cand_mask"])
                    neg = torch.finfo(scores.dtype).min / 4
                    final = torch.log_softmax(scores.masked_fill(~batch["cand_mask"], neg), dim=-1)
                    idx = torch.argmax(scores, dim=-1)
                    b = torch.arange(scores.size(0), device=device)
                    pick = batch["sample"][b, idx]
                    gate = torch.full((scores.size(0),), float("nan"), device=device)
                    tgt = soft_target_from_errors(batch["errors"], batch["cand_mask"], temp)
                    vloss = listwise_ce(final, tgt, batch["cand_mask"])
                else:
                    pout = predict_gate_pick(
                        model,
                        batch["x"],
                        batch["missing"],
                        batch["prob"],
                        batch["prominence"],
                        batch["sample"],
                        batch["cand_mask"],
                        batch["expected"],
                        batch["sigma"],
                        batch["history_available"],
                        prominence_weight=pw,
                    )
                    pick = pout["pick_sample"]
                    gate = pout["gate"]
                    tgt = soft_target_from_errors(batch["errors"], batch["cand_mask"], temp)
                    vloss = listwise_ce(pout["final_logprob"], tgt, batch["cand_mask"])
                vlosses.append(float(vloss.item()))
                # map true_s=-1 to nan
                ts = batch["true_s"].view(-1).cpu().numpy()
                ts = np.where(ts < 0, np.nan, ts)
                preds.extend(pick.cpu().numpy().tolist())
                trues.extend(ts.tolist())
                srs.extend(batch["sampling_rate"].view(-1).cpu().numpy().tolist())
                gates.extend(gate.cpu().numpy().tolist())

        m = match_picks(np.asarray(preds), np.asarray(trues), np.asarray(srs))
        f1 = float(m["f1@0.5s"])
        garr = np.asarray(gates, dtype=np.float64)
        garr = garr[np.isfinite(garr)]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": float(np.mean(vlosses)) if vlosses else float("nan"),
            "val_f1_0.5": f1,
            "gate_mean": float(np.mean(garr)) if garr.size else float("nan"),
            "gate_std": float(np.std(garr)) if garr.size else float("nan"),
        }
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in ["epoch", "train_loss", "val_loss", "val_f1_0.5", "gate_mean", "gate_std"]])
        print(row, flush=True)

        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "n_features": n_feat, "model_type": model_type}, out / "last.pt")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "cfg": cfg, "n_features": n_feat, "model_type": model_type, "val_f1_0.5": best_f1}, out / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print({"early_stop": epoch, "best_f1": best_f1}, flush=True)
                break

    save_json(
        {
            "seed": seed,
            "model_type": model_type,
            "n_params": n_params,
            "best_val_f1_0.5": best_f1,
            "n_train": len(train_rec),
            "n_val": len(val_rec),
            "out": str(out),
        },
        out / "train_summary.json",
    )
    # full config dump
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return {"seed": seed, "best_val_f1_0.5": best_f1, "n_params": n_params, "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_debug.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-type", default=None, choices=["scalar_gate", "candidate_ranker"])
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    model_type = args.model_type or cfg.get("model_type", "scalar_gate")
    summary = train_one(cfg, seed, model_type=model_type)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
