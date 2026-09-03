#!/usr/bin/env python
"""Per-GPU worker for Stage 7A runtime benchmark (single e2e path)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import resolve_instance_root
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference, stream_from_row


def gpu_stats(device_index: int) -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    util, mem_used = [float(x.strip()) for x in out.split(",")]
    return {"util": util, "mem_used_mib": mem_used}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--n-timed", type=int, required=True)
    ap.add_argument("--physical-gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=["e2e", "sections"],
        default="e2e",
        help="e2e=predict_row only; sections=separate read/pre/annotate timings (extra work)",
    )
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    need = args.warmup + args.n_timed
    if len(man) < need:
        raise SystemExit(f"manifest too short: {len(man)} < {need}")

    ref = SeisBenchPhaseNetReference(weight=args.weight, device="cuda:0")
    reader = InstanceHDF5Reader(resolve_instance_root() / "events" / "Instance_events_counts.hdf5").open()
    try:
        for i in range(args.warmup):
            row = man.iloc[i]
            _ = ref.predict_row(reader.read_waveform(str(row.trace_name)), row, remap_to_waveform=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        util_samples = []
        mem_peak = 0.0
        t_read = t_pre = t_ann = t_cand = t_ser = 0.0
        ser_rows: list[dict] = []
        t0_all = time.perf_counter()
        for j in range(args.n_timed):
            row = man.iloc[args.warmup + j]
            if args.mode == "sections":
                t0 = time.perf_counter()
                wave = reader.read_waveform(str(row.trace_name))
                t_read += time.perf_counter() - t0
                t0 = time.perf_counter()
                st = stream_from_row(wave, row)
                t_pre += time.perf_counter() - t0
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = ref.annotate_stream(st)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_ann += time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = ref.predict_row(wave, row, remap_to_waveform=True)
                t_cand += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                wave = reader.read_waveform(str(row.trace_name))
                t_read += time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = ref.predict_row(wave, row, remap_to_waveform=True)
                t_cand += time.perf_counter() - t0
                t_ann = t_cand  # annotate embedded in predict_row
                t_pre = float("nan")

            t0 = time.perf_counter()
            ser_rows.append(
                {
                    "trace_name": str(row.trace_name),
                    "pred_s_sample": float(pred["s_pred_sample_on_waveform"]),
                    "s_peak_probability": float(pred["s_peak_probability"]),
                }
            )
            if len(ser_rows) % 50 == 0:
                _ = pd.DataFrame(ser_rows[-50:]).to_json()
            t_ser += time.perf_counter() - t0

            if j % 25 == 0:
                stt = gpu_stats(args.physical_gpu)
                util_samples.append(stt["util"])
                mem_peak = max(mem_peak, stt["mem_used_mib"])
        wall = time.perf_counter() - t0_all
    finally:
        reader.close()

    util_arr = np.asarray(util_samples, float) if util_samples else np.array([float("nan")])
    n = float(args.n_timed)
    out = {
        "physical_gpu": args.physical_gpu,
        "weight": args.weight,
        "mode": args.mode,
        "warmup": args.warmup,
        "n_timed": args.n_timed,
        "wall_clock_s": wall,
        "traces_per_s_e2e": n / wall,
        "sections": {
            "hdf5_read_s": t_read,
            "preprocess_s": t_pre,
            "model_annotate_s": t_ann,
            "candidate_extraction_s": t_cand,
            "cache_serialization_s": t_ser,
            "hdf5_read_tps": n / max(t_read, 1e-9),
            "candidate_extraction_tps": n / max(t_cand, 1e-9),
        },
        "gpu_util_median": float(np.nanmedian(util_arr)),
        "gpu_util_p95": float(np.nanpercentile(util_arr, 95)),
        "gpu_mem_peak_mib_smi": mem_peak,
        "gpu_mem_peak_mib_torch": float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else float("nan"),
        "batch_size_note": "SeisBench annotate default (single-stream per call; official path)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"gpu": args.physical_gpu, "tps": out["traces_per_s_e2e"]}))


if __name__ == "__main__":
    main()
