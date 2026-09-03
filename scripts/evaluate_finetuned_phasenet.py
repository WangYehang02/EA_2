#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import match_picks, noise_false_positive_rate
from earthquake.models.phasenet_wrapper import PhaseNetWrapper
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir


def load_finetuned(weight_init: str, ckpt: Path, device: str):
    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained(weight_init)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phasenet_finetune_debug.yaml")
    parser.add_argument("--ckpt", default="artifacts/phasenet_finetune/checkpoints/best.pt")
    parser.add_argument("--eval-list", default="artifacts/diagnostics/fixed_eval_traces.txt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    device = args.device or cfg.get("device", "cpu")

    eval_list = Path(args.eval_list)
    if not eval_list.exists():
        raise SystemExit(f"Missing fixed eval list: {eval_list}. Build it first.")
    names = [ln.strip() for ln in eval_list.read_text().splitlines() if ln.strip()]
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index" / "noise.parquet")
    ev = events[events["trace_name"].astype(str).isin(names)].copy()
    nz = noise[noise["trace_name"].astype(str).isin(names)].copy() if len(noise) else pd.DataFrame()

    # Baseline stead via annotate reference
    base = SeisBenchPhaseNetReference(weight=cfg.get("init_weight", "stead"), device=device)
    # Finetuned: reuse wrapper plumbing by swapping model
    ft_wrap = PhaseNetWrapper(weight=cfg.get("init_weight", "stead"), device=device)
    ft_model = load_finetuned(cfg.get("init_weight", "stead"), ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt), device)
    ft_wrap.model = ft_model
    ft_wrap._ref.model = ft_model

    h5e = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    h5n = resolve_instance_root() / "noise" / "Instance_noise.hdf5"

    def eval_events(model_ref, label: str):
        preds_p, true_p, preds_s, true_s, srs = [], [], [], [], []
        with InstanceHDF5Reader(h5e) as reader:
            for _, row in tqdm(ev.iterrows(), total=len(ev), desc=f"eval-events:{label}"):
                wave = reader.read_waveform(str(row.trace_name))
                if hasattr(model_ref, "predict_row"):
                    out = model_ref.predict_row(wave, row)
                    preds_p.append(float(out["p_pred_sample_on_waveform"]))
                    preds_s.append(float(out["s_pred_sample_on_waveform"]))
                else:
                    out = model_ref.predict_row(wave, row)
                    preds_p.append(float(out["p_pred_sample_on_waveform"]))
                    preds_s.append(float(out["s_pred_sample_on_waveform"]))
                true_p.append(float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan)
                true_s.append(float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan)
                srs.append(float(row.sampling_rate_hz))
        mp = match_picks(np.array(preds_p), np.array(true_p), np.array(srs))
        ms = match_picks(np.array(preds_s), np.array(true_s), np.array(srs))
        return {"P": mp, "S": ms, "n": len(ev)}

    def eval_noise(model_ref, label: str):
        if len(nz) == 0:
            return {"p_fpr": None, "s_fpr": None, "n": 0}
        det_p, det_s = [], []
        with InstanceHDF5Reader(h5n) as reader:
            for _, row in tqdm(nz.iterrows(), total=len(nz), desc=f"eval-noise:{label}"):
                wave = reader.read_waveform(str(row.trace_name))
                # synthesize minimal row fields for stream
                r = row.copy()
                if "network" not in r or pd.isna(r.get("network")):
                    # noise schema already normalized
                    pass
                out = model_ref.predict_row(wave, r)
                det_p.append(float(out["p_peak_probability"]) >= float(cfg.get("pick_threshold", 0.3)))
                det_s.append(float(out["s_peak_probability"]) >= float(cfg.get("pick_threshold", 0.3)))
        return {
            "p_fpr": noise_false_positive_rate(np.array(det_p)),
            "s_fpr": noise_false_positive_rate(np.array(det_s)),
            "n": len(nz),
        }

    out_dir = ensure_dir(artifacts_dir() / "results")
    results = {
        "baseline_stead": {**eval_events(base, "stead"), "noise": eval_noise(base, "stead")},
        "finetuned": {**eval_events(ft_wrap, "finetuned"), "noise": eval_noise(ft_wrap, "finetuned")},
        "ckpt": str(args.ckpt),
        "n_eval_events": len(ev),
        "n_eval_noise": len(nz),
    }
    save_json(results, out_dir / "finetuned_phasenet_metrics.json")
    print(json_dumps := __import__("json").dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
