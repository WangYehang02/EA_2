"""Pack per-trace candidate groups into fixed-K tensors for ranker training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TraceCandidateDataset(Dataset):
    def __init__(
        self,
        feat_df: pd.DataFrame,
        feature_names: list[str],
        *,
        max_k: int = 10,
        hard_boost: bool = False,
        seed: int = 0,
    ):
        self.feature_names = list(feature_names)
        self.max_k = int(max_k)
        self.groups = []
        rng = np.random.default_rng(seed)
        for tn, g in feat_df.groupby("trace_name", sort=False):
            g = g.sort_values("candidate_index").reset_index(drop=True)
            n = min(len(g), self.max_k)
            x = np.zeros((self.max_k, len(self.feature_names)), dtype=np.float32)
            mask = np.zeros(self.max_k, dtype=bool)
            samples = np.full(self.max_k, np.nan, dtype=np.float64)
            for i in range(n):
                x[i] = g.iloc[i][self.feature_names].to_numpy(dtype=np.float32)
                mask[i] = True
                samples[i] = float(g.iloc[i]["candidate_sample"])
            none = bool(g.iloc[0]["label_none_of_k"])
            pos = int(g.iloc[0]["positive_index"])
            target = self.max_k if none or pos < 0 else int(pos)
            if target < self.max_k and not mask[target]:
                target = self.max_k
            classes = str(g.iloc[0].get("analysis_classes", ""))
            weight = 1.0
            if hard_boost:
                if "ida_only_recoverable" in classes or "recoverable_stead_wrong" in classes or "multi_peak" in classes:
                    weight = 2.0
                if "none_of_k" in classes:
                    weight = 1.5
            self.groups.append(
                {
                    "trace_name": str(tn),
                    "event_id": str(g.iloc[0]["event_id"]),
                    "x": x,
                    "mask": mask,
                    "target": target,
                    "samples": samples,
                    "true_s": float(g.iloc[0]["true_s_sample"]),
                    "sr": float(g.iloc[0]["sampling_rate_hz"]),
                    "weight": weight,
                    "classes": classes,
                }
            )
        # optional resample indices for hard-heavy training
        self.indices = list(range(len(self.groups)))
        if hard_boost and self.groups:
            hard = [i for i, g in enumerate(self.groups) if g["weight"] >= 2.0]
            none = [i for i, g in enumerate(self.groups) if "none_of_k" in g["classes"]]
            ordinary = [i for i, g in enumerate(self.groups) if i not in hard and i not in none]
            n = len(self.groups)
            n_hard = int(0.5 * n)
            n_none = int(0.25 * n)
            n_ord = n - n_hard - n_none
            pick = []
            if hard:
                pick.extend(rng.choice(hard, size=n_hard, replace=True).tolist())
            if none:
                pick.extend(rng.choice(none, size=n_none, replace=True).tolist())
            if ordinary:
                pick.extend(rng.choice(ordinary, size=max(n_ord, 0), replace=True).tolist())
            if pick:
                self.indices = pick

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        g = self.groups[self.indices[idx]]
        return {
            "x": torch.from_numpy(g["x"]),
            "mask": torch.from_numpy(g["mask"]),
            "target": torch.tensor(g["target"], dtype=torch.long),
            "samples": torch.from_numpy(g["samples"]),
            "true_s": torch.tensor(g["true_s"], dtype=torch.float64),
            "sr": torch.tensor(g["sr"], dtype=torch.float64),
            "trace_name": g["trace_name"],
            "event_id": g["event_id"],
        }


def collate_traces(batch):
    out = {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "samples": torch.stack([b["samples"] for b in batch], dim=0),
        "true_s": torch.stack([b["true_s"] for b in batch], dim=0),
        "sr": torch.stack([b["sr"] for b in batch], dim=0),
        "trace_name": [b["trace_name"] for b in batch],
        "event_id": [b["event_id"] for b in batch],
    }
    return out
