from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class InstanceHDF5Reader:
    """Lazy HDF5 waveform reader for INSTANCE counts files.

    Supports common layouts:
    - /data/<trace_name> dataset with shape (3, T) or (T, 3)
    - /<trace_name> dataset
    - nested groups ending in a dataset
    """

    def __init__(self, hdf5_path: Path | str):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file: h5py.File | None = None
        self._data_group_name: str | None = None
        self._probed = False

    def open(self) -> "InstanceHDF5Reader":
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._probe()
        return self

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "InstanceHDF5Reader":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self.open()
        assert self._file is not None
        return self._file

    def _probe(self) -> None:
        if self._probed:
            return
        keys = list(self.file.keys())
        if "data" in keys and isinstance(self.file["data"], h5py.Group):
            self._data_group_name = "data"
        else:
            self._data_group_name = None
        self._probed = True

    def structure_summary(self, max_items: int = 20) -> dict[str, Any]:
        self.open()
        top = list(self.file.keys())
        summary: dict[str, Any] = {
            "path": str(self.path),
            "top_keys": top[:max_items],
            "n_top_keys": len(top),
            "data_group": self._data_group_name,
        }
        if self._data_group_name:
            g = self.file[self._data_group_name]
            kids = list(g.keys())
            summary["data_n_keys"] = len(kids)
            summary["data_sample_keys"] = kids[:5]
            if kids:
                ds = g[kids[0]]
                if isinstance(ds, h5py.Dataset):
                    summary["sample_dataset"] = {
                        "name": kids[0],
                        "shape": list(ds.shape),
                        "dtype": str(ds.dtype),
                    }
        elif top:
            first = self.file[top[0]]
            if isinstance(first, h5py.Dataset):
                summary["sample_dataset"] = {
                    "name": top[0],
                    "shape": list(first.shape),
                    "dtype": str(first.dtype),
                }
            elif isinstance(first, h5py.Group):
                summary["sample_group_children"] = list(first.keys())[:10]
        return summary

    def _lookup(self, trace_name: str) -> h5py.Dataset:
        f = self.file
        candidates = [trace_name]
        # INSTANCE sometimes stores with/without trailing channel details
        if trace_name.endswith(".HH") or trace_name.endswith(".EH") or trace_name.endswith(".HN"):
            pass
        if self._data_group_name:
            g = f[self._data_group_name]
            if trace_name in g:
                obj = g[trace_name]
                if isinstance(obj, h5py.Dataset):
                    return obj
            # try component group layout: data/trace/{E,N,Z}
            if trace_name in g and isinstance(g[trace_name], h5py.Group):
                raise KeyError(f"Trace {trace_name} is a group; use read_waveform for components")
        if trace_name in f and isinstance(f[trace_name], h5py.Dataset):
            return f[trace_name]
        raise KeyError(f"Trace not found in HDF5: {trace_name}")

    def has_trace(self, trace_name: str) -> bool:
        try:
            self._lookup(trace_name)
            return True
        except KeyError:
            # component-wise fallback
            if self._data_group_name:
                g = self.file[self._data_group_name]
                if trace_name in g and isinstance(g[trace_name], h5py.Group):
                    return True
                for comp in ("E", "N", "Z", "1", "2", "Z"):
                    key = f"{trace_name}{comp}" if not trace_name.endswith(comp) else None
                # SeisBench INSTANCE style: datasets named exactly as trace_name
            return False

    def read_waveform(self, trace_name: str) -> np.ndarray:
        """Return waveform as float32 array with shape (3, T), order ENZ if possible."""
        self.open()
        if self._data_group_name:
            g = self.file[self._data_group_name]
            if trace_name in g and isinstance(g[trace_name], h5py.Group):
                grp = g[trace_name]
                comps = []
                for c in ("E", "N", "Z"):
                    if c not in grp:
                        # try channel suffix forms
                        matches = [k for k in grp.keys() if str(k).endswith(c)]
                        if not matches:
                            raise KeyError(f"Missing component {c} for {trace_name}")
                        comps.append(np.asarray(grp[matches[0]][...]))
                    else:
                        comps.append(np.asarray(grp[c][...]))
                arr = np.stack(comps, axis=0)
                return arr.astype(np.float32, copy=False)

        ds = self._lookup(trace_name)
        arr = np.asarray(ds[...])
        if arr.ndim == 1:
            raise ValueError(f"Expected multi-component waveform for {trace_name}, got shape {arr.shape}")
        if arr.ndim == 2:
            if arr.shape[0] == 3:
                out = arr
            elif arr.shape[1] == 3:
                out = arr.T
            else:
                raise ValueError(f"Unrecognized waveform shape {arr.shape} for {trace_name}")
        else:
            raise ValueError(f"Unrecognized waveform ndim={arr.ndim} for {trace_name}")
        return out.astype(np.float32, copy=False)
