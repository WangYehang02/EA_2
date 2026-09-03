#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.data.metadata import find_events_csv, find_noise_csv, read_csv_header
from earthquake.data.schema import LOGICAL_FIELDS, SchemaMapping, map_columns
from earthquake.utils import ensure_dir


def main() -> None:
    instance_root = resolve_instance_root()
    out = ensure_dir(artifacts_dir() / "schema")

    events_csv = find_events_csv(instance_root)
    noise_csv = find_noise_csv(instance_root)
    events_cols = read_csv_header(events_csv)
    noise_cols = read_csv_header(noise_csv)

    events_map = map_columns(events_cols)
    noise_map = map_columns(noise_cols)

    derived = {}
    if events_map.get("sampling_rate_hz") is None and "trace_dt_s" in events_cols:
        derived["sampling_rate_hz"] = "1.0 / trace_dt_s"

    schema = SchemaMapping(
        mapping=events_map,
        derived=derived,
        events_columns=events_cols,
        noise_columns=noise_cols,
    )
    save_json(schema.to_dict(), out / "schema_mapping.json")
    pd.DataFrame({"column": events_cols}).to_csv(out / "events_columns.csv", index=False)
    pd.DataFrame({"column": noise_cols}).to_csv(out / "noise_columns.csv", index=False)

    events_h5 = instance_root / "events" / "Instance_events_counts.hdf5"
    noise_h5 = instance_root / "noise" / "Instance_noise.hdf5"

    audit: dict = {
        "instance_root": str(instance_root),
        "events_csv": str(events_csv),
        "noise_csv": str(noise_csv),
        "events_hdf5": str(events_h5),
        "noise_hdf5": str(noise_h5),
        "logical_fields_mapped": {k: events_map.get(k) for k in LOGICAL_FIELDS},
        "unmapped_logical_fields": [k for k in LOGICAL_FIELDS if events_map.get(k) is None and k not in derived],
        "derived": derived,
        "waveform_checks": [],
    }

    # Sample a few rows for waveform validation
    sample_df = pd.read_csv(events_csv, nrows=50, low_memory=False)
    rename = {v: k for k, v in events_map.items() if v is not None}
    sample_df = sample_df.rename(columns=rename)
    if "sampling_rate_hz" not in sample_df.columns and "trace_dt_s" in sample_df.columns:
        sample_df["sampling_rate_hz"] = 1.0 / pd.to_numeric(sample_df["trace_dt_s"], errors="coerce")

    with InstanceHDF5Reader(events_h5) as reader:
        audit["events_hdf5_structure"] = reader.structure_summary()
        n_ok = 0
        for _, row in sample_df.head(10).iterrows():
            tname = str(row.get("trace_name", ""))
            check = {"trace_name": tname}
            try:
                wave = reader.read_waveform(tname)
                check.update(
                    {
                        "ok": True,
                        "shape": list(wave.shape),
                        "dtype": str(wave.dtype),
                        "n_samples_meta": int(row.get("n_samples", -1)) if pd.notna(row.get("n_samples", np.nan)) else None,
                        "sampling_rate_hz": float(row.get("sampling_rate_hz", np.nan)),
                    }
                )
                npts = wave.shape[-1]
                for phase in ("p", "s"):
                    scol = f"{phase}_arrival_sample"
                    if scol in row and pd.notna(row[scol]):
                        s = float(row[scol])
                        check[f"{phase}_in_range"] = 0 <= s < npts
                n_ok += 1
            except Exception as e:
                check.update({"ok": False, "error": str(e)})
            audit["waveform_checks"].append(check)
        audit["n_waveform_ok"] = n_ok

    if noise_h5.exists():
        noise_sample = pd.read_csv(noise_csv, nrows=5, low_memory=False)
        noise_rename = {v: k for k, v in noise_map.items() if v is not None}
        noise_sample = noise_sample.rename(columns=noise_rename)
        with InstanceHDF5Reader(noise_h5) as reader:
            audit["noise_hdf5_structure"] = reader.structure_summary()
            for _, row in noise_sample.iterrows():
                tname = str(row.get("trace_name", ""))
                try:
                    wave = reader.read_waveform(tname)
                    audit.setdefault("noise_waveform_checks", []).append(
                        {"trace_name": tname, "ok": True, "shape": list(wave.shape), "dtype": str(wave.dtype)}
                    )
                except Exception as e:
                    audit.setdefault("noise_waveform_checks", []).append(
                        {"trace_name": tname, "ok": False, "error": str(e)}
                    )

    save_json(audit, out / "audit_report.json")
    print(json.dumps({"instance_root": str(instance_root), "artifacts": str(out), "n_waveform_ok": audit.get("n_waveform_ok")}, indent=2))
    print("Wrote schema_mapping.json / events_columns.csv / noise_columns.csv / audit_report.json")


if __name__ == "__main__":
    main()
