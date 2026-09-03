"""Optional temperature scaling for gate calibration (val-fit only)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp_min(1e-3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
