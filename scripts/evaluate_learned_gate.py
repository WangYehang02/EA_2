#!/usr/bin/env python
"""Evaluate learned gate vs PhaseNet / fixed rescore / ablations / oracles."""

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
from earthquake.gating.cache_io import load_gate_records
from earthquake.gating.candidate_ranker import CandidateRanker
from earthquake.gating.dataset import GateTraceDataset, collate_gate
from earthquake.gating.inference import predict_gate_pick
from earthquake.gating.scalar_gate import ScalarGate, gated_candidate_logprobs, history_scores, pick_from_final, wave_scores
from earthquake.metrics import audit_pick_errors, match_picks
from earthquake.utils import ensure_dir


def _summarize_s(pred, true, sr) -> dict:
    m = match_picks(np.asarray(pred), np.asarray(true), np.asarray(sr))
    a = audit_pick_errors(np.asarray(pred), np.asarray(true), np.asarray(sr), window_s=0.5)
    n_lab = max(int(a["n_labeled"]), 1)
    return {
        "precision@0.1": m["precision@0.1s"],
        "recall@0.1": m["recall@0.1s"],
        "f1@0.1": m["f1@0.1s"],
        "precision@0.5": m["precision@0.5s"],
        "recall@0.5": m["recall@0.5s"],
        "f1@0.5": m["f1@0.5s"],
        "e2e_mae": m["mae"],
        "e2e_median_ae": m["median_ae"],
        "e2e_p95": m["p95_ae"],
        "matched_timing_mae": a["matched_timing_mae"],
        "wrong_peak_rate": float(a["n_wrong_peak_beyond_tol"] / n_lab),
        "miss_rate": float(a["n_missed_pick"] / n_lab),
        "n": int(np.isfinite(true).sum()),
    }


def _load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mt = ckpt.get("model_type", "scalar_gate")
    n_feat = int(ckpt["n_features"])
    cfg = ckpt.get("cfg", {})
    if mt == "candidate_ranker":
        model = CandidateRanker(n_feat, hidden_dim=int(cfg.get("hidden_dim", 64)), dropout=float(cfg.get("dropout", 0.1)))
    else:
        model = ScalarGate(n_feat, hidden_dim=int(cfg.get("hidden_dim", 64)), dropout=float(cfg.get("dropout", 0.1)), init_gate=float(cfg.get("init_gate", 0.8)))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, mt, cfg


@torch.no_grad()
def predict_methods(records, model, model_type, cfg, device, feature_ablation=None, history_mode="normal"):
    """Return dict of method -> list of pred_s samples, plus gate values."""
    max_k = int(cfg.get("candidate_k", 5))
    pw = float(cfg.get("prominence_weight", 0.1))
    ds = GateTraceDataset(records, max_k=max_k, train=False, corrupt_prob=0.0, seed=0, prominence_weight=pw)
    # optional history corruption for eval controls
    if history_mode == "shuffled":
        rng = np.random.default_rng(0)
        exp = np.array([r["expected_s_sample"] for r in records], dtype=np.float64)
        perm = rng.permutation(len(exp))
        for i, r in enumerate(records):
            r = dict(r)
            r["expected_s_sample"] = float(exp[perm[i]])
            records[i] = r
        ds = GateTraceDataset(records, max_k=max_k, train=False, corrupt_prob=0.0, seed=0, prominence_weight=pw)
    elif history_mode == "biased":
        for i, r in enumerate(records):
            r = dict(r)
            r["expected_s_sample"] = float(r["expected_s_sample"]) + 5.0 * float(r["sampling_rate"])
            records[i] = r
        ds = GateTraceDataset(records, max_k=max_k, train=False, corrupt_prob=0.0, seed=0, prominence_weight=pw)

    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_gate)
    out = {
        "phasenet": [],
        "fixed_rescore": [],
        "learned_gate": [],
        "gate": [],
        "oracle_candidate": [],
        "distance_only": [],
        "true_s": [],
        "pred_p": [],
        "sr": [],
        "trace_name": [],
        "event_id": [],
        "history_available": [],
    }
    # fixed_rescore / phasenet from record fields
    rec_map = {r["trace_name"]: r for r in records}

    for batch in dl:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        x = batch["x"]
        miss = batch["missing"]
        if feature_ablation == "no_history_quality":
            # zero columns that look like history quality — approximate by masking second half? use missing=1 on hist cols
            # simpler: set history-related dims via schema positions unknown — skip precise; zero x where |z| from hist
            pass
        if feature_ablation == "no_waveform_uncertainty":
            pass

        if model_type == "candidate_ranker":
            z = (batch["sample"] - batch["expected"]) / batch["sigma"].clamp_min(1e-6)
            cand_feat = torch.stack(
                [batch["prob"], batch["prominence"], batch["width"], batch["rank"], z, torch.zeros_like(batch["prob"])],
                dim=-1,
            )
            scores = model(x, miss, cand_feat, batch["cand_mask"])
            idx = torch.argmax(scores, dim=-1)
            b = torch.arange(scores.size(0), device=device)
            pick = batch["sample"][b, idx]
            gate = torch.full((scores.size(0),), float("nan"), device=device)
        else:
            pout = predict_gate_pick(
                model,
                x,
                miss,
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

        # wave-only / hist-only
        w = wave_scores(batch["prob"], batch["prominence"], prominence_weight=pw)
        h = history_scores(batch["sample"], batch["expected"].expand_as(batch["sample"]), batch["sigma"].expand_as(batch["sample"]))
        ones = torch.ones(batch["prob"].size(0), 1, device=device)
        zeros = torch.zeros_like(ones)
        pn_idx = pick_from_final(gated_candidate_logprobs(ones, w, h, batch["cand_mask"]), batch["cand_mask"])
        # oracle candidate among batch
        err = batch["errors"]
        ora_idx = torch.argmin(err.masked_fill(~batch["cand_mask"], 1e9), dim=-1)
        b = torch.arange(batch["prob"].size(0), device=device)
        for j, name in enumerate(batch["trace_name"]):
            r = rec_map[name]
            ts = float(r["true_s_sample"])
            if ts < 0:
                ts = float("nan")
            out["phasenet"].append(float(r["pred_s_phasenet"]))
            out["fixed_rescore"].append(float(r.get("pred_s_fixed_rescore", r["pred_s_phasenet"])))
            out["learned_gate"].append(float(pick[j].item()))
            out["gate"].append(float(gate[j].item()) if torch.isfinite(gate[j]) else float("nan"))
            out["oracle_candidate"].append(float(batch["sample"][j, ora_idx[j]].item()))
            out["distance_only"].append(float(r["expected_s_sample"]))  # travel-time expected as proxy
            out["true_s"].append(ts)
            out["pred_p"].append(float(r["pred_p_sample"]))
            out["sr"].append(float(r["sampling_rate"]))
            out["trace_name"].append(name)
            out["event_id"].append(r["event_id"])
            out["history_available"].append(bool(r["history_available"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_debug.yaml")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-type", default="scalar_gate")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out = ensure_dir(artifacts_dir() / "results" / "stage3")
    records = load_gate_records(
        artifacts_dir() / "candidates" / f"stage3_{args.split}.zarr",
        min_history=float(cfg.get("min_history", 5)),
        mad_threshold=float(cfg.get("mad_threshold", 1.0)),
    )
    # restore fixed_rescore from baselines parquet if present
    tag = "debug" if "debug" in Path(args.config).stem else str(cfg.get("mode", "catalog"))
    base_path = out / f"gate_baselines_{tag}.parquet"
    if base_path.exists():
        base = pd.read_parquet(base_path)
        base = base[base["split"] == args.split]
        bmap = base.set_index("trace_name")["pred_s_fixed_rescore"].to_dict()
        for r in records:
            if r["trace_name"] in bmap:
                r["pred_s_fixed_rescore"] = float(bmap[r["trace_name"]])

    seeds = list(cfg.get("seeds", [cfg.get("seed", 42)])) if args.all_seeds else [int(args.seed if args.seed is not None else cfg.get("seed", 42))]
    seed_metrics = []
    all_picks = []

    for seed in seeds:
        ckpt = artifacts_dir() / "models" / "learned_gate" / f"{args.model_type}_seed{seed}" / "best.pt"
        if not ckpt.exists():
            print({"missing_ckpt": str(ckpt)}, flush=True)
            continue
        model, mt, mcfg = _load_model(ckpt, device)
        # merge train cfg
        use_cfg = dict(cfg)
        use_cfg.update({k: v for k, v in mcfg.items() if k in cfg or k in {"hidden_dim", "dropout", "prominence_weight", "candidate_k"}})

        pred = predict_methods(records, model, mt, use_cfg, device)
        # oracle selector
        ora_sel = []
        for i in range(len(pred["trace_name"])):
            pn = pred["phasenet"][i]
            fr = pred["fixed_rescore"][i]
            true = pred["true_s"][i]
            sr = pred["sr"][i]
            e_pn = abs(pn - true) / sr if np.isfinite(pn) and np.isfinite(true) else np.inf
            e_fr = abs(fr - true) / sr if np.isfinite(fr) and np.isfinite(true) else np.inf
            ora_sel.append(pn if e_pn <= e_fr else fr)

        methods = {
            "phasenet": pred["phasenet"],
            "distance_only": pred["distance_only"],
            "fixed_catalog_rescore": pred["fixed_rescore"],
            "learned_gate": pred["learned_gate"],
            "oracle_candidate": pred["oracle_candidate"],
            "oracle_selector": ora_sel,
        }
        # shuffled / biased history eval
        for hmode, key in [("shuffled", "learned_gate_shuffled_history"), ("biased", "learned_gate_biased_history")]:
            # copy records fresh
            rec2 = load_gate_records(
                artifacts_dir() / "candidates" / f"stage3_{args.split}.zarr",
                min_history=float(cfg.get("min_history", 5)),
                mad_threshold=float(cfg.get("mad_threshold", 1.0)),
            )
            if base_path.exists():
                for r in rec2:
                    if r["trace_name"] in bmap:
                        r["pred_s_fixed_rescore"] = float(bmap[r["trace_name"]])
            p2 = predict_methods(rec2, model, mt, use_cfg, device, history_mode=hmode)
            methods[key] = p2["learned_gate"]

        row = {"seed": seed}
        for name, arr in methods.items():
            s = _summarize_s(arr, pred["true_s"], pred["sr"])
            for k, v in s.items():
                row[f"{name}__{k}"] = v
            print(seed, name, {k: s[k] for k in ["f1@0.1", "f1@0.5", "e2e_mae", "e2e_p95", "wrong_peak_rate"]}, flush=True)
        g = np.asarray(pred["gate"], dtype=np.float64)
        g = g[np.isfinite(g)]
        row["gate_mean"] = float(np.mean(g)) if g.size else float("nan")
        row["gate_std"] = float(np.std(g)) if g.size else float("nan")
        row["gate_collapse"] = bool(g.size and float(np.std(g)) < 0.02)
        seed_metrics.append(row)

        all_picks.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "trace_name": pred["trace_name"],
                    "event_id": pred["event_id"],
                    "pred_p_sample": pred["pred_p"],
                    "pred_s_phasenet": pred["phasenet"],
                    "pred_s_fixed_rescore": pred["fixed_rescore"],
                    "pred_s_learned_gate": pred["learned_gate"],
                    "pred_s_oracle_candidate": pred["oracle_candidate"],
                    "pred_s_oracle_selector": ora_sel,
                    "true_s_sample": pred["true_s"],
                    "sampling_rate_hz": pred["sr"],
                    "gate": pred["gate"],
                    "history_available": pred["history_available"],
                }
            )
        )

    if not seed_metrics:
        raise SystemExit("No seeds evaluated")

    metrics_df = pd.DataFrame(seed_metrics)
    metrics_df.to_csv(out / f"learned_gate_metrics_{args.split}.csv", index=False)
    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df.to_parquet(out / f"learned_gate_picks_{args.split}.parquet", index=False)

    # mean/std across seeds for learned gate
    summary = {"n_seeds": len(seed_metrics), "per_seed": seed_metrics, "aggregate": {}}
    for col in metrics_df.columns:
        if col == "seed":
            continue
        if metrics_df[col].dtype == bool:
            continue
        if pd.api.types.is_numeric_dtype(metrics_df[col]):
            summary["aggregate"][col] = {
                "mean": float(metrics_df[col].mean()),
                "std": float(metrics_df[col].std(ddof=0)) if len(metrics_df) > 1 else 0.0,
            }
    # decision vs fixed rescore
    if "learned_gate__f1@0.5" in metrics_df.columns and "fixed_catalog_rescore__f1@0.5" in metrics_df.columns:
        d = metrics_df["learned_gate__f1@0.5"] - metrics_df["fixed_catalog_rescore__f1@0.5"]
        summary["learned_vs_fixed_f1_0.5"] = {"mean_delta": float(d.mean()), "per_seed": d.tolist()}
        summary["claim_beats_fixed_rescore"] = False  # bootstrap decides
        summary["note"] = "Do not claim superiority without bootstrap CI (scripts/bootstrap_gate.py)."
    save_json(summary, out / f"learned_gate_eval_{args.split}.json")
    print(json.dumps({"aggregate_keys": list(summary["aggregate"])[:8], "n_seeds": summary["n_seeds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
