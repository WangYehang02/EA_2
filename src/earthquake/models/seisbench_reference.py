from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime


def to_utc(value) -> UTCDateTime:
    """Robust conversion to ObsPy UTCDateTime from pandas/str/datetime/UTCDateTime."""
    if isinstance(value, UTCDateTime):
        return value
    if isinstance(value, pd.Timestamp):
        # Avoid pandas string formats that ObsPy fails to parse (e.g. +00:00)
        if pd.isna(value):
            raise ValueError("Cannot convert NaT to UTCDateTime")
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        return UTCDateTime(ts.to_pydatetime())
    if isinstance(value, str):
        # Normalize Z/+00:00 styles
        s = value.strip().replace("+00:00", "Z")
        if s.endswith("Z") or "T" in s:
            return UTCDateTime(s)
        return UTCDateTime(pd.Timestamp(s, tz="UTC").to_pydatetime())
    # datetime-like
    try:
        return UTCDateTime(value)
    except Exception:
        return UTCDateTime(pd.Timestamp(value).to_pydatetime())


def channel_code(channel_prefix: str, component: str) -> str:
    """Build SEED channel code from INSTANCE station_channels prefix + component."""
    prefix = str(channel_prefix).strip()
    comp = str(component).upper()
    if len(prefix) >= 3 and prefix[-1].upper() in {"E", "N", "Z", "1", "2", "3"}:
        # already a full channel
        return prefix[:-1] + comp
    # INSTANCE uses HH/EH/HN style prefixes
    return f"{prefix}{comp}"


def enz_array_to_stream(
    waveform_enz: np.ndarray,
    *,
    starttime: pd.Timestamp | str | UTCDateTime,
    sampling_rate: float,
    network: str,
    station: str,
    location: str | None,
    channel_prefix: str,
) -> Stream:
    """Create ObsPy Stream from ENZ array with correct channel component suffixes."""
    if waveform_enz.ndim != 2 or waveform_enz.shape[0] != 3:
        raise ValueError(f"Expected ENZ waveform (3, T), got {waveform_enz.shape}")
    t0 = to_utc(starttime)
    loc = "" if location is None or (isinstance(location, float) and np.isnan(location)) else str(location)
    st = Stream()
    for comp, arr in zip(("E", "N", "Z"), waveform_enz):
        tr = Trace(data=np.asarray(arr, dtype=np.float32).copy())
        tr.stats.network = str(network)
        tr.stats.station = str(station)
        tr.stats.location = loc
        tr.stats.channel = channel_code(channel_prefix, comp)
        tr.stats.sampling_rate = float(sampling_rate)
        tr.stats.starttime = t0
        st.append(tr)
    lengths = {tr.stats.npts for tr in st}
    if len(lengths) != 1:
        raise ValueError(f"Component length mismatch: {[(tr.stats.channel, tr.stats.npts) for tr in st]}")
    return st


def stream_from_row(waveform_enz: np.ndarray, row: pd.Series) -> Stream:
    return enz_array_to_stream(
        waveform_enz,
        starttime=row["trace_start_time"],
        sampling_rate=float(row.get("sampling_rate_hz", 100.0)),
        network=str(row["network"]),
        station=str(row["station"]),
        location=row.get("location", ""),
        channel_prefix=str(row["channel_prefix"]),
    )


def extract_phase_traces(annotated: Stream, model_name: str = "PhaseNet") -> dict[str, Trace]:
    out = {}
    for phase, key in (("p", "P"), ("s", "S"), ("noise", "N")):
        cands = [tr for tr in annotated if tr.stats.channel.endswith(f"_{key}") or tr.stats.channel.endswith(key)]
        if not cands:
            # SeisBench uses ModelName_P etc.
            cands = [tr for tr in annotated if tr.stats.channel.upper().endswith(f"_{key}")]
        if not cands:
            raise KeyError(f"Missing annotated phase channel for {phase} in {[tr.stats.channel for tr in annotated]}")
        out[phase] = cands[0]
    return out


def remap_trace_to_waveform_grid(
    trace: Trace,
    waveform_start: pd.Timestamp | str | UTCDateTime,
    n_samples: int,
    sampling_rate: float,
) -> np.ndarray:
    """Interpolate a prediction Trace onto the original waveform sample grid via UTC time."""
    w0 = to_utc(waveform_start)
    sr = float(sampling_rate)
    pred = np.asarray(trace.data, dtype=np.float64).reshape(-1)
    if pred.size == 0 or n_samples <= 0:
        return np.zeros(max(int(n_samples), 0), dtype=np.float32)
    p0 = trace.stats.starttime
    psr = float(trace.stats.sampling_rate)
    if not np.isfinite(psr) or psr <= 0:
        return np.zeros(int(n_samples), dtype=np.float32)
    pred_times = np.array([float(p0 + i / psr - w0) for i in range(len(pred))], dtype=np.float64)
    target_times = np.arange(n_samples, dtype=np.float64) / sr
    out = np.interp(target_times, pred_times, pred, left=0.0, right=0.0).astype(np.float32)
    return out


def peak_utc_from_trace(trace: Trace, threshold: float = 0.0) -> tuple[UTCDateTime | None, float, int]:
    """Return (peak_utc, peak_prob, peak_index_on_trace).

    Empty / length-0 prediction traces return (None, nan, -1) instead of raising.
    """
    data = np.asarray(trace.data, dtype=np.float64).reshape(-1)
    if data.size == 0 or not np.isfinite(data).any():
        return None, float("nan"), -1
    # Ignore non-finite samples for peak search
    finite = np.isfinite(data)
    if not finite.any():
        return None, float("nan"), -1
    idx = int(np.argmax(np.where(finite, data, -np.inf)))
    prob = float(data[idx])
    if not np.isfinite(prob) or prob < threshold:
        return None, prob, idx
    t = trace.stats.starttime + idx / float(trace.stats.sampling_rate)
    return t, prob, idx


def sample_index_on_waveform(peak_utc: UTCDateTime | None, waveform_start, sampling_rate: float) -> float:
    if peak_utc is None:
        return float("nan")
    w0 = to_utc(waveform_start)
    return float(peak_utc - w0) * float(sampling_rate)


def true_pick_utc(waveform_start, arrival_sample: float, sampling_rate: float) -> UTCDateTime | None:
    if arrival_sample is None or not np.isfinite(float(arrival_sample)):
        return None
    w0 = to_utc(waveform_start)
    return w0 + float(arrival_sample) / float(sampling_rate)


class SeisBenchPhaseNetReference:
    """Official SeisBench annotate()-based PhaseNet inference with UTC alignment."""

    def __init__(self, weight: str = "stead", device: str | None = None, allow_instance: bool = False):
        import seisbench.models as sbm
        import torch

        if (not allow_instance) and "instance" in weight.lower():
            raise ValueError("Refusing INSTANCE weight unless allow_instance=True (diagnostic only)")
        self.weight = weight
        self.allow_instance = allow_instance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sbm.PhaseNet.from_pretrained(weight)
        self.model.to(self.device)
        self.model.eval()

    def annotate_stream(self, stream: Stream, **kwargs) -> Stream:
        return self.model.annotate(stream, **kwargs)

    def predict_row(
        self,
        waveform_enz: np.ndarray,
        row: pd.Series,
        *,
        remap_to_waveform: bool = True,
        **annotate_kwargs,
    ) -> dict[str, Any]:
        st = stream_from_row(waveform_enz, row)
        ann = self.annotate_stream(st, **annotate_kwargs)
        phases = extract_phase_traces(ann)
        sr = float(row.get("sampling_rate_hz", 100.0))
        n = int(waveform_enz.shape[-1])
        w_start = row["trace_start_time"]

        result: dict[str, Any] = {
            "weight": self.weight,
            "input_starttime": str(to_utc(w_start)),
            "input_npts": n,
            "annotated_channels": [tr.stats.channel for tr in ann],
            "diagnostic_only_data_leakage": bool(self.allow_instance and "instance" in self.weight.lower()),
        }
        for phase, tr in phases.items():
            peak_t, peak_p, peak_i = peak_utc_from_trace(tr)
            result[f"{phase}_output_starttime"] = str(tr.stats.starttime)
            result[f"{phase}_output_npts"] = int(tr.stats.npts)
            result[f"{phase}_output_sampling_rate"] = float(tr.stats.sampling_rate)
            result[f"{phase}_peak_trace_sample"] = int(peak_i)
            result[f"{phase}_peak_probability"] = peak_p
            result[f"{phase}_peak_utc"] = str(peak_t) if peak_t is not None else None
            result[f"{phase}_pred_sample_on_waveform"] = sample_index_on_waveform(peak_t, w_start, sr)
            if remap_to_waveform:
                result[f"{phase}_proba_on_waveform"] = remap_trace_to_waveform_grid(tr, w_start, n, sr)
            result[f"{phase}_trace"] = tr
        # Convenience aliases matching wrapper API
        if remap_to_waveform:
            result["noise"] = result["noise_proba_on_waveform"]
            result["p"] = result["p_proba_on_waveform"]
            result["s"] = result["s_proba_on_waveform"]
        return result
