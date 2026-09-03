#!/usr/bin/env python
from __future__ import annotations

"""Train a lightweight learned gate on top of frozen PhaseNet outputs.

This is implemented for completeness; stage-1 acceptance focuses on fixed alpha.
Run only after fixed-alpha shows gains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, save_json
from earthquake.models.learned_gate import LearnedGate
from earthquake.utils import ensure_dir


FEATURE_COLS = [
    "p_peak_probability",
    "s_peak_probability",
    "p_entropy",
    "s_entropy",
    "p_peak_width",
    "s_peak_width",
    "history_count",
    "tau_p_mad",
    "tau_s_mad",
    "delta_sp_mad",
    "time_since_last_history",
    "nearest_historical_source_distance_km",
]


class GateFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        feats = []
        for c in FEATURE_COLS:
            if c == "history_count":
                feats.append(np.log1p(df[c].fillna(0).to_numpy(dtype=np.float32)))
            else:
                feats.append(df[c].fillna(0).to_numpy(dtype=np.float32))
        self.x = np.stack(feats, axis=1)
        self.hist = df["history_available"].astype(bool).to_numpy()
        # Supervision proxy: prefer PhaseNet when history sparse / high MAD
        # Gate target around 0.9 initially; residual learning via loss on picks is approximate.
        self.y = np.stack(
            [
                df.get("true_p_sample", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32),
                df.get("true_s_sample", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        self.pn = np.stack(
            [
                df["pred_p_sample"].to_numpy(dtype=np.float32),
                df["pred_s_sample"].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "hist": torch.tensor(self.hist[idx], dtype=torch.bool),
            "y": torch.from_numpy(self.y[idx]),
            "pn": torch.from_numpy(self.pn[idx]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_learned.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)

    pn = pd.read_parquet(artifacts_dir() / "phasenet" / "phasenet_picks.parquet")
    hist = pd.read_parquet(artifacts_dir() / "history" / "history_features_frozen.parquet")
    df = pn.merge(hist, on=["trace_name", "event_id", "split"], how="inner", suffixes=("", "_h"))
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    if len(train) == 0 or len(val) == 0:
        # In debug, PhaseNet may only have been run on val/test. Fall back to val for smoke.
        train = val if len(val) else df
        print({"warning": "insufficient train picks; using available rows for smoke training", "n": len(train)})

    ds = GateFeatureDataset(train)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=True)
    model = LearnedGate(in_dim=len(FEATURE_COLS), hidden=int(cfg.get("hidden", 64)), init_gate=float(cfg.get("init_gate", 0.9)))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    w = cfg.get("loss_weights", {})

    model.train()
    for epoch in range(int(cfg.get("epochs", 5))):
        total = 0.0
        for batch in loader:
            gates = model(batch["x"], history_available=batch["hist"].float())
            # Encourage trusting PhaseNet when history weak
            hist_count = batch["x"][:, FEATURE_COLS.index("history_count")]
            mad = batch["x"][:, FEATURE_COLS.index("tau_p_mad")] + batch["x"][:, FEATURE_COLS.index("tau_s_mad")]
            dist = batch["x"][:, FEATURE_COLS.index("nearest_historical_source_distance_km")]
            trust_pn = torch.sigmoid(1.0 - hist_count + 0.5 * mad + 0.01 * dist).unsqueeze(1).expand_as(gates)
            sparse_reg = F.mse_loss(gates, trust_pn)
            # Smooth L1 between PhaseNet picks and labels when available
            y = batch["y"]
            pn_pick = batch["pn"]
            mask = torch.isfinite(y)
            arrival = F.smooth_l1_loss(pn_pick[mask], y[mask]) if mask.any() else torch.tensor(0.0)
            # Keep gates near init unless evidence
            gate_prior = F.mse_loss(gates, torch.full_like(gates, float(cfg.get("init_gate", 0.9))))
            loss = (
                float(w.get("final_phase", 1.0)) * gate_prior
                + float(w.get("phasenet_auxiliary", 0.3)) * gate_prior
                + float(w.get("arrival_smooth_l1", 0.1)) * arrival
                + float(w.get("sparse_history_gate", 0.02)) * sparse_reg
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print({"epoch": epoch, "loss": total / max(len(loader), 1)})

    out = ensure_dir(artifacts_dir() / "learned")
    torch.save({"state_dict": model.state_dict(), "feature_cols": FEATURE_COLS, "cfg": cfg}, out / "learned_gate.pt")
    save_json({"status": "trained_smoke", "n_train": len(ds)}, out / "train_summary.json")
    # Placeholder metrics file for pipeline completeness
    save_json({"note": "Run fixed-alpha first; learned gate metrics filled after full eval."}, artifacts_dir() / "results" / "learned_fusion_metrics.json")
    print({"saved": str(out / "learned_gate.pt")})


if __name__ == "__main__":
    main()
