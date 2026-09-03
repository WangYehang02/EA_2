"""DKPN diagnostic adapter — INSTANCE-trained weights; not for main Table A."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DKPN_ROOT = Path("/home/yehang/EARTHQUAKE/baseline/DKPN")
DEFAULT_CKPT = (
    DKPN_ROOT
    / "models_v0412_paper_sb4/MEDIUM/DKPN_TrainDataset_INSTANCE_Size_MEDIUM_Rnd_50_Epochs_10_LR_0.0010_Batch_64.pt"
)


@dataclass
class DKPNConfig:
    checkpoint: Path = DEFAULT_CKPT
    device: str = "cpu"
    role: str = "diagnostic_only_possible_leakage"
    official_S_threshold: float = 0.2
    component_order: str = "ZNE"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class DKPNWeightAudit:
    """Load checkpoint tensors only — does not claim valid comparator predictions."""

    def __init__(self, cfg: DKPNConfig | None = None):
        self.cfg = cfg or DKPNConfig()
        if not self.cfg.checkpoint.exists():
            raise FileNotFoundError(self.cfg.checkpoint)
        self.sha256 = sha256_file(self.cfg.checkpoint)
        obj = torch.load(self.cfg.checkpoint, map_location="cpu", weights_only=False)
        self.raw = obj
        if isinstance(obj, dict):
            # common patterns
            if "state_dict" in obj:
                self.state = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                self.state = obj["model"]
            else:
                # may already be state_dict
                self.state = {k: v for k, v in obj.items() if torch.is_tensor(v)}
                if not self.state:
                    self.state = obj
        else:
            self.state = obj

    def summary(self) -> dict[str, Any]:
        n_tensors = 0
        n_params = 0
        keys = []
        if isinstance(self.state, dict):
            for k, v in self.state.items():
                if torch.is_tensor(v):
                    n_tensors += 1
                    n_params += int(v.numel())
                    keys.append(k)
        return {
            "checkpoint": str(self.cfg.checkpoint),
            "sha256": self.sha256,
            "role": self.cfg.role,
            "n_tensors": n_tensors,
            "n_params": n_params,
            "example_keys": keys[:8],
            "contains_INSTANCE_in_filename": "INSTANCE" in self.cfg.checkpoint.name,
            "enter_main_table": False,
            "random_init": False,
            "loaded": n_tensors > 0 or self.state is not None,
        }
