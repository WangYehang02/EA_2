from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earthquake.config import artifacts_dir, resolve_instance_root
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.phasenet_wrapper import PhaseNetWrapper
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference


@pytest.mark.parametrize("_unused", [None])
def test_reference_wrapper_agreement_on_real_traces(_unused):
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    sample = events[events["split"] == "val"].head(5)
    wrap = PhaseNetWrapper(weight="stead", device="cpu")
    ref = SeisBenchPhaseNetReference(weight="stead", device="cpu")
    ref.model = wrap.model
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    with InstanceHDF5Reader(h5) as reader:
        for _, row in sample.iterrows():
            wave = reader.read_waveform(str(row.trace_name))
            a = wrap.predict_row(wave, row)
            b = ref.predict_row(wave, row)
            for ph in ("p", "s"):
                da = float(a[f"{ph}_pred_sample_on_waveform"])
                db = float(b[f"{ph}_pred_sample_on_waveform"])
                assert abs(da - db) <= 1.0 + 1e-6
                # remapped probs nearly identical
                assert np.max(np.abs(a[ph] - b[ph])) < 1e-5
