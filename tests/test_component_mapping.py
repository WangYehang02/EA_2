from __future__ import annotations

import numpy as np
import pandas as pd

from earthquake.models.seisbench_reference import enz_array_to_stream


def test_component_mapping_enz_channels():
    wave = np.zeros((3, 1000), dtype=np.float32)
    wave[0] = 1  # E
    wave[1] = 2
    wave[2] = 3
    st = enz_array_to_stream(
        wave,
        starttime="2020-01-01T00:00:00Z",
        sampling_rate=100.0,
        network="IV",
        station="AAAA",
        location="",
        channel_prefix="HH",
    )
    chans = {tr.stats.channel: float(tr.data[0]) for tr in st}
    assert chans["HHE"] == 1.0
    assert chans["HHN"] == 2.0
    assert chans["HHZ"] == 3.0
    assert len({tr.stats.npts for tr in st}) == 1
