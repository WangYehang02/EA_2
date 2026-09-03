from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from earthquake.data.hdf5_reader import InstanceHDF5Reader


class InstanceWaveformDataset(Dataset):
    """Lazy waveform dataset backed by parquet index + HDF5."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        hdf5_path: Path | str,
        transform=None,
    ):
        self.df = index_df.reset_index(drop=True)
        self.reader = InstanceHDF5Reader(hdf5_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        trace_name = str(row["trace_name"])
        wave = self.reader.read_waveform(trace_name)  # (3, T)
        sample = {
            "waveform": torch.from_numpy(np.asarray(wave, dtype=np.float32)),
            "trace_name": trace_name,
            "event_id": str(row.get("event_id", "")),
            "sampling_rate_hz": float(row.get("sampling_rate_hz", 100.0)),
            "p_arrival_sample": float(row["p_arrival_sample"]) if pd.notna(row.get("p_arrival_sample")) else float("nan"),
            "s_arrival_sample": float(row["s_arrival_sample"]) if pd.notna(row.get("s_arrival_sample")) else float("nan"),
            "index": int(idx),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        self.reader.close()
