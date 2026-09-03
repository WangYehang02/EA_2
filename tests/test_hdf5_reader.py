from __future__ import annotations

import numpy as np
import pytest

from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.config import resolve_instance_root


def test_hdf5_reader_reads_trace():
    root = resolve_instance_root()
    h5 = root / "events" / "Instance_events_counts.hdf5"
    import pandas as pd
    from earthquake.data.metadata import find_events_csv

    df = pd.read_csv(find_events_csv(root), nrows=5)
    tname = str(df.iloc[0]["trace_name"])
    with InstanceHDF5Reader(h5) as r:
        wave = r.read_waveform(tname)
    assert wave.ndim == 2
    assert wave.shape[0] == 3
    assert wave.shape[1] > 100
    assert wave.dtype == np.float32
