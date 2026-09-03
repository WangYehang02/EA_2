"""Frozen-cycle sampler so A/B share batch order."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class FrozenCycleSampler(Sampler[int]):
    """Infinite cycle of a seed-frozen permutation. A and B must use the same seed."""

    def __init__(self, n: int, seed: int = 42):
        self.n = int(n)
        self.seed = int(seed)
        self.order = np.random.default_rng(self.seed).permutation(self.n).astype(int).tolist()

    def __iter__(self):
        yield from self.order

    def __len__(self) -> int:
        return self.n
