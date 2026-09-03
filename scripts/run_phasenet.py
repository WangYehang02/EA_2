#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.phasenet_wrapper import PhaseNetWrapper, list_pretrained_weights
from earthquake.picking import extract_pick_features
from earthquake.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "test"])
    args = parser.parse_args()

    cfg = load_yaml_config(ROOT / args.config)
    instance_root = resolve_instance_root()
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")

    available = list_pretrained_weights()
    print("PhaseNet pretrained weights:", available)
    weight = cfg.get("phasenet_weight") or "stead"
    wrapper = PhaseNetWrapper(weight=weight, device=cfg.get("device", "cpu"))
    out_dir = ensure_dir(artifacts_dir() / "phasenet")
    wrapper.save_info(out_dir / "model_info.json")
    save_json({"available_weights": available, "selected": wrapper.weight}, out_dir / "weight_selection.json")

    max_traces = int(cfg.get("max_traces_per_split", cfg.get("max_eval_traces", 10**9)))
    rows = []
    h5_path = instance_root / "events" / "Instance_events_counts.hdf5"

    # Determine output length from first waveform
    with InstanceHDF5Reader(h5_path) as reader:
        for split in args.splits:
            sub = events[events["split"] == split].copy()
            if "max_events" in cfg:
                # already indexed
                pass
            sub = sub.head(max_traces)
            if len(sub) == 0:
                continue
            # probe length
            w0 = reader.read_waveform(str(sub.iloc[0]["trace_name"]))
            tlen = int(w0.shape[-1])
            zpath = out_dir / f"probs_{split}.zarr"
            if zpath.exists():
                import shutil

                shutil.rmtree(zpath)
            root = zarr.open_group(str(zpath), mode="w")
            chunk = int(cfg.get("zarr_chunk_traces", 32))
            arr_p = root.create_array("p", shape=(len(sub), tlen), chunks=(min(chunk, len(sub)), tlen), dtype="f4")
            arr_s = root.create_array("s", shape=(len(sub), tlen), chunks=(min(chunk, len(sub)), tlen), dtype="f4")
            arr_n = root.create_array("noise", shape=(len(sub), tlen), chunks=(min(chunk, len(sub)), tlen), dtype="f4")
            names = []

            for i, (_, row) in enumerate(tqdm(sub.iterrows(), total=len(sub), desc=f"PhaseNet:{split}")):
                tname = str(row["trace_name"])
                wave = reader.read_waveform(tname)
                if wave.shape[-1] != tlen:
                    # pad/crop for storage uniformity in debug
                    outw = np.zeros((3, tlen), dtype=np.float32)
                    m = min(tlen, wave.shape[-1])
                    outw[:, :m] = wave[:, :m]
                    wave = outw
                proba = wrapper.predict_proba(wave, row=row)
                arr_p[i] = proba["p"][:tlen]
                arr_s[i] = proba["s"][:tlen]
                arr_n[i] = proba["noise"][:tlen]
                # Prefer UTC-aligned peak samples from annotate path
                full = wrapper.predict_row(wave, row)
                pf = extract_pick_features(proba["p"])
                sf = extract_pick_features(proba["s"])
                names.append(tname)
                rows.append(
                    {
                        "trace_name": tname,
                        "event_id": str(row["event_id"]),
                        "split": split,
                        "sampling_rate_hz": float(row["sampling_rate_hz"]),
                        "true_p_sample": float(row["p_arrival_sample"]) if pd.notna(row["p_arrival_sample"]) else np.nan,
                        "true_s_sample": float(row["s_arrival_sample"]) if pd.notna(row["s_arrival_sample"]) else np.nan,
                        "pred_p_sample": float(full["p_pred_sample_on_waveform"]),
                        "pred_s_sample": float(full["s_pred_sample_on_waveform"]),
                        "p_peak_probability": float(full["p_peak_probability"]),
                        "s_peak_probability": float(full["s_peak_probability"]),
                        "p_entropy": pf["entropy"],
                        "s_entropy": sf["entropy"],
                        "p_peak_width": pf["peak_width"],
                        "s_peak_width": sf["peak_width"],
                        "output_starttime": full["p_output_starttime"],
                        "output_npts": full["p_output_npts"],
                        "snr_db": row.get("snr_db"),
                        "distance_km": row.get("distance_km"),
                    }
                )
            # Store names as JSON sidecar (zarr object dtype varies by version)
            (out_dir / f"probs_{split}_trace_names.json").write_text(
                __import__("json").dumps(names), encoding="utf-8"
            )

    picks = pd.DataFrame(rows)
    picks.to_parquet(out_dir / "phasenet_picks.parquet", index=False)
    print({"n_picks": len(picks), "weight": wrapper.weight, "out": str(out_dir)})


if __name__ == "__main__":
    main()
