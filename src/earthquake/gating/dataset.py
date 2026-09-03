"""Gate training dataset with optional history corruption (train only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class GateTraceDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        max_k: int = 5,
        train: bool = False,
        corrupt_prob: float = 0.0,
        seed: int = 0,
        prominence_weight: float = 0.1,
    ):
        self.records = records
        self.max_k = int(max_k)
        self.train = train
        self.corrupt_prob = float(corrupt_prob)
        self.rng = np.random.default_rng(seed)
        self.prominence_weight = float(prominence_weight)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        x = torch.tensor(r["x"], dtype=torch.float32)
        m = torch.tensor(r["missing"], dtype=torch.float32)
        k = self.max_k
        prob = np.zeros(k, dtype=np.float32)
        prom = np.zeros(k, dtype=np.float32)
        width = np.zeros(k, dtype=np.float32)
        rank = np.arange(k, dtype=np.float32)
        samp = np.full(k, np.nan, dtype=np.float32)
        mask = np.zeros(k, dtype=bool)
        cands = r["s_cands"][:k]
        for j, c in enumerate(cands):
            prob[j] = float(c["peak_probability"])
            prom[j] = float(c.get("prominence", c["peak_probability"]))
            width[j] = float(c.get("peak_width", np.nan)) if np.isfinite(float(c.get("peak_width", np.nan))) else 0.0
            samp[j] = float(c["sample_index"])
            mask[j] = True

        expected = float(r["expected_s_sample"])
        sigma = float(r["history_sigma_samples"])
        hist_avail = bool(r["history_available"])

        # train-only history corruption
        if self.train and hist_avail and self.rng.random() < self.corrupt_prob:
            kind = int(self.rng.integers(0, 4))
            if kind == 0:
                # swap expected with another record
                j = int(self.rng.integers(0, len(self.records)))
                expected = float(self.records[j]["expected_s_sample"])
            elif kind == 1:
                expected = expected + float(self.rng.uniform(2, 10) * self.rng.choice([-1, 1])) * float(r["sampling_rate"])
            elif kind == 2:
                expected = expected + float(self.rng.normal(0, 5.0)) * float(r["sampling_rate"])
            else:
                expected = expected + float(self.rng.uniform(-8, 8)) * float(r["sampling_rate"])

        true_s = float(r["true_s_sample"]) if r["true_s_sample"] is not None and np.isfinite(r["true_s_sample"]) else np.nan
        errors = np.full(k, 1e6, dtype=np.float32)
        if np.isfinite(true_s):
            for j in range(k):
                if mask[j]:
                    errors[j] = abs(samp[j] - true_s) / float(r["sampling_rate"])

        bad_hist = (not hist_avail) or (float(r.get("history_count", 0)) < float(r.get("min_history", 5))) or (
            np.isfinite(float(r.get("history_mad", np.nan))) and float(r.get("history_mad", 0)) > float(r.get("mad_threshold", 1.0))
        )

        return {
            "x": x,
            "missing": m,
            "prob": torch.tensor(prob),
            "prominence": torch.tensor(prom),
            "width": torch.tensor(width),
            "rank": torch.tensor(rank),
            "sample": torch.tensor(samp),
            "cand_mask": torch.tensor(mask),
            "expected": torch.tensor([expected], dtype=torch.float32),
            "sigma": torch.tensor([sigma if np.isfinite(sigma) and sigma > 1e-6 else 20.0], dtype=torch.float32),
            "errors": torch.tensor(errors),
            "history_available": torch.tensor([hist_avail], dtype=torch.bool),
            "bad_history": torch.tensor([bad_hist], dtype=torch.bool),
            "true_s": torch.tensor([true_s if np.isfinite(true_s) else -1.0], dtype=torch.float32),
            "sampling_rate": torch.tensor([float(r["sampling_rate"])], dtype=torch.float32),
            "trace_name": r["trace_name"],
            "event_id": r["event_id"],
        }


def collate_gate(batch: list[dict]) -> dict[str, Any]:
    out = {}
    for k in batch[0]:
        if k in {"trace_name", "event_id"}:
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
