from __future__ import annotations

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from earthquake.models.seisbench_reference import (
    channel_code,
    remap_trace_to_waveform_grid,
    sample_index_on_waveform,
    true_pick_utc,
)
from obspy import Trace


def test_lazy_dataset_converts_enz_to_zne(monkeypatch):
    """HDF5 ENZ must be remapped to PhaseNet ZNE before the crop tensor."""
    import earthquake.stage6.lazy_dataset as ld

    monkeypatch.setattr(ld, "random_crop_window", lambda *a, **k: 0)

    class _FakeSrc:
        def read(self, name, *, is_noise=False):
            T = 5000
            t = np.arange(T, dtype=np.float32)
            return np.stack([t, 2 * t, np.sin(t / 50.0)], axis=0).astype(np.float32)

    meta = pd.DataFrame(
        [
            {
                "trace_name": "dummy",
                "event_id": "1",
                "is_noise": False,
                "p_arrival_sample": 1000.0,
                "s_arrival_sample": 1500.0,
                "sampling_rate_hz": 100.0,
            }
        ]
    )
    ds = ld.LazyPhaseNetCropDataset(meta, _FakeSrc(), in_samples=3001, augment=False, allow_p_only=False)
    x = ds[0]["x"].numpy()
    assert x.shape == (3, 3001)
    z_raw = np.sin(np.arange(3001, dtype=np.float32) / 50.0)
    z_raw = (z_raw - z_raw.mean()) / max(float(z_raw.std()), 1e-6)
    corr_z = float(np.corrcoef(x[0], z_raw)[0, 1])
    e_raw = np.arange(3001, dtype=np.float32)
    e_raw = (e_raw - e_raw.mean()) / max(float(e_raw.std()), 1e-6)
    corr_e_if_wrong = float(np.corrcoef(x[0], e_raw)[0, 1])
    assert corr_z > 0.99
    assert abs(corr_e_if_wrong) < 0.2



def test_peak_utc_all_nan_returns_none():
    from earthquake.models.seisbench_reference import peak_utc_from_trace

    tr = Trace(data=np.array([np.nan, np.nan], dtype=np.float64))
    tr.stats.sampling_rate = 100.0
    t, p, i = peak_utc_from_trace(tr)
    assert t is None and i == -1



def test_remap_respects_output_start_offset():
    # prediction starts 2.5s later, length 100
    w0 = UTCDateTime("2016-01-01T00:00:00Z")
    tr = Trace(data=np.zeros(100, dtype=np.float32))
    tr.data[10] = 1.0
    tr.stats.starttime = w0 + 2.5
    tr.stats.sampling_rate = 100.0
    out = remap_trace_to_waveform_grid(tr, w0, n_samples=1200, sampling_rate=100.0)
    # peak at pred sample 10 => absolute time = w0+2.5+0.10 = w0+2.6s => waveform sample 260
    assert int(np.argmax(out)) == 260


def test_channel_code():
    assert channel_code("HH", "Z") == "HHZ"
    assert channel_code("EH", "E") == "EHE"
