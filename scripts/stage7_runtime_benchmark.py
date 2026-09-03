#!/usr/bin/env python
"""Stage 7A runtime benchmark: 1/4/(8 if free) GPUs on fixed 10k-dev manifest.

Does not use confirm for throughput tuning. Free GPUs only; no process killing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir


def free_gpus(max_used_mib: float = 500.0) -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in out.strip().splitlines():
        idx_s, used_s = [x.strip() for x in line.split(",")]
        if float(used_s) < max_used_mib:
            free.append(int(idx_s))
    return free


def build_runtime_manifest(dev_csv: Path, n: int = 10000) -> pd.DataFrame:
    man = pd.read_csv(dev_csv)
    man = man.copy()
    man["_h"] = man.trace_name.astype(str).map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    man = man.sort_values("_h").reset_index(drop=True).head(n).drop(columns=["_h"])
    return man


def gpu_stats(device_index: int) -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    util, mem_used, mem_tot = [float(x.strip()) for x in out.split(",")]
    return {"util": util, "mem_used_mib": mem_used, "mem_total_mib": mem_tot}


def timed_sections(
    *,
    weight: str,
    manifest: pd.DataFrame,
    device: str,
    physical_gpu: int,
    warmup: int,
    n_timed: int,
    batch_note: str,
) -> dict:
    ref = SeisBenchPhaseNetReference(weight=weight, device=device)
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    rows = manifest.iloc[: warmup + n_timed]
    # warmup
    for i in range(warmup):
        row = rows.iloc[i]
        wave = reader.read_waveform(str(row.trace_name))
        _ = ref.predict_row(wave, row, remap_to_waveform=True)

    util_samples = []
    mem_peak = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_read = t_pre = t_ann = t_cand = t_ser = 0.0
    preds = []
    t_e2e0 = time.perf_counter()
    for i in range(warmup, warmup + n_timed):
        row = rows.iloc[i]
        t0 = time.perf_counter()
        wave = reader.read_waveform(str(row.trace_name))
        t_read += time.perf_counter() - t0

        t0 = time.perf_counter()
        # preprocess is inside stream_from_row / annotate; approximate as stream build only
        from earthquake.models.seisbench_reference import stream_from_row

        st = stream_from_row(wave, row)
        t_pre += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.inference_mode():
            ann = ref.annotate_stream(st)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ann += time.perf_counter() - t0

        t0 = time.perf_counter()
        # candidate extraction = peak from annotate result via predict_row path
        pred = ref.predict_row(wave, row, remap_to_waveform=True)
        t_cand += time.perf_counter() - t0

        t0 = time.perf_counter()
        preds.append(
            {
                "trace_name": str(row.trace_name),
                "pred_s_sample": float(pred["s_pred_sample_on_waveform"]),
                "s_peak_probability": float(pred["s_peak_probability"]),
            }
        )
        # serialize every 50 to amortize
        if len(preds) % 50 == 0:
            _ = pd.DataFrame(preds).to_parquet  # noqa: touch attribute
            buf = pd.DataFrame(preds[-50:]).to_json()
            del buf
        t_ser += time.perf_counter() - t0

        if (i - warmup) % 20 == 0:
            stt = gpu_stats(physical_gpu)
            util_samples.append(stt["util"])
            mem_peak = max(mem_peak, stt["mem_used_mib"])

    t_e2e = time.perf_counter() - t_e2e0
    reader.close()
    cuda_peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else float("nan")
    n = float(n_timed)
    util_arr = np.asarray(util_samples, float) if util_samples else np.array([float("nan")])
    return {
        "weight": weight,
        "physical_gpu": physical_gpu,
        "n_timed": n_timed,
        "warmup": warmup,
        "batch_size_note": batch_note,
        "wall_clock_s": t_e2e,
        "traces_per_s_e2e": n / t_e2e,
        "sections_s": {
            "hdf5_read": t_read,
            "preprocess_stream": t_pre,
            "model_annotate": t_ann,
            "candidate_extraction_full_predict_row": t_cand,
            "cache_serialization_amortized": t_ser,
        },
        "section_traces_per_s": {
            "hdf5_read": n / max(t_read, 1e-9),
            "preprocess_stream": n / max(t_pre, 1e-9),
            "model_annotate": n / max(t_ann, 1e-9),
            "candidate_extraction_full_predict_row": n / max(t_cand, 1e-9),
        },
        "gpu_util_median": float(np.nanmedian(util_arr)),
        "gpu_util_p95": float(np.nanpercentile(util_arr, 95)),
        "gpu_mem_peak_mib_smi": mem_peak,
        "gpu_mem_peak_mib_torch": cuda_peak,
    }


def run_multi_gpu_e2e(
    *,
    weight: str,
    manifest: pd.DataFrame,
    gpus: list[int],
    warmup: int,
    n_timed: int,
) -> dict:
    """Approximate multi-GPU by sharding timed traces across free GPUs (separate processes would be better;
    here we run sequential per-GPU shards of equal size and report aggregate throughput as sum of
    per-GPU rates measured concurrently via subprocesses."""
    # Launch one subprocess per GPU measuring its shard concurrently
    out_dir = ensure_dir(artifacts_dir() / "results" / "stage7" / "runtime_tmp")
    procs = []
    per = n_timed // len(gpus)
    rem = n_timed % len(gpus)
    offset = 0
    for i, g in enumerate(gpus):
        n_i = per + (1 if i < rem else 0)
        man_i = manifest.iloc[offset : offset + warmup + n_i].reset_index(drop=True)
        offset += n_i  # note: warmup shared conceptually; each process warms on first warmup of its slice
        man_path = out_dir / f"man_g{g}.csv"
        man_i.to_csv(man_path, index=False)
        result_path = out_dir / f"res_g{g}.json"
        log_path = out_dir / f"log_g{g}.txt"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "stage7_runtime_worker.py"),
            "--weight",
            weight,
            "--manifest",
            str(man_path),
            "--warmup",
            str(warmup),
            "--n-timed",
            str(n_i),
            "--physical-gpu",
            str(g),
            "--out",
            str(result_path),
        ]
        procs.append(
            (
                g,
                n_i,
                result_path,
                subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT),
            )
        )
    t0 = time.perf_counter()
    results = []
    for g, n_i, rp, p in procs:
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"worker gpu={g} failed rc={rc}")
        results.append(json.loads(rp.read_text()))
    wall = time.perf_counter() - t0
    total_n = sum(r["n_timed"] for r in results)
    # Per-worker traces_per_s_e2e already excludes warmup; sum ≈ sustained multi-GPU rate under contention.
    sustained = float(sum(float(r["traces_per_s_e2e"]) for r in results))
    return {
        "n_gpus": len(gpus),
        "gpus": gpus,
        "wall_clock_s_includes_warmup": wall,
        "n_timed_total": total_n,
        "traces_per_s_aggregate": sustained,
        "traces_per_s_wall_includes_warmup": total_n / wall,
        "per_gpu": results,
        "events_per_s_approx": sustained / max(manifest.event_id.nunique() / max(len(manifest), 1), 1e-9)
        if "event_id" in manifest.columns
        else None,
        "hdf5_read_fraction_median": float(
            np.median(
                [
                    r["sections"]["hdf5_read_s"]
                    / max(r["sections"]["hdf5_read_s"] + r["sections"]["candidate_extraction_s"], 1e-9)
                    for r in results
                ]
            )
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="ethz", choices=["ethz", "scedc", "stead"])
    ap.add_argument("--n-bench", type=int, default=1000, help="timed traces per setting (after warmup)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage7")
    reports = ensure_dir(ROOT / "reports" / "stage7")
    free = free_gpus()
    # exclude GPUs currently used by our long cache jobs if memory high — re-query after cache done preferred
    print({"free_gpus": free}, flush=True)

    man = build_runtime_manifest(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv", 10000)
    man_path = out / "runtime_manifest_10k.csv"
    man.to_csv(man_path, index=False)
    man_hash = sha256_file(man_path)
    save_json(
        {
            "n": len(man),
            "sha256": man_hash,
            "selection": "sha256(trace_name) sort, first 10000 of Stage6 dev",
            "n_events": int(man.event_id.nunique()),
            "n_stations": int(man.station_code.nunique()) if "station_code" in man.columns else None,
            "confirm_overlap": 0,
        },
        out / "runtime_manifest_meta.json",
    )

    # Need worker script
    configs = []
    if len(free) >= 1:
        configs.append(("1gpu", free[:1]))
    if len(free) >= 4:
        configs.append(("4gpu", free[:4]))
    if len(free) >= 8:
        configs.append(("8gpu", free[:8]))
    else:
        configs.append(("8gpu_unavailable", free))  # record actual

    all_runs = []
    scaling_rows = []
    for label, gpus in configs:
        if label.endswith("_unavailable") and len(gpus) < 8:
            scaling_rows.append(
                {
                    "setting": "8gpu",
                    "requested": 8,
                    "actual_gpus": len(gpus),
                    "status": "insufficient_free_gpus",
                    "gpus": gpus,
                }
            )
            continue
        rep_rates = []
        for rep in range(args.reps):
            print({"run": label, "rep": rep, "gpus": gpus}, flush=True)
            # use subset of 10k for timed; full warmup from same manifest
            # take first warmup+n_bench after hash sort (deterministic)
            sub = man.iloc[: args.warmup + args.n_bench].reset_index(drop=True)
            r = run_multi_gpu_e2e(
                weight=args.weight,
                manifest=sub,
                gpus=gpus,
                warmup=args.warmup,
                n_timed=args.n_bench,
            )
            r["label"] = label
            r["rep"] = rep
            all_runs.append(r)
            rep_rates.append(r["traces_per_s_aggregate"])
            print(r, flush=True)
        rates = np.asarray(rep_rates, float)
        scaling_rows.append(
            {
                "setting": label,
                "n_gpus": len(gpus),
                "gpus": gpus,
                "traces_per_s_median": float(np.median(rates)),
                "traces_per_s_min": float(rates.min()),
                "traces_per_s_max": float(rates.max()),
                "weight": args.weight,
                "n_timed_per_rep": args.n_bench,
                "warmup": args.warmup,
                "reps": args.reps,
            }
        )

    # scaling ratios
    by = {r["setting"]: r for r in scaling_rows if "traces_per_s_median" in r}
    ratios = {}
    if "1gpu" in by and "4gpu" in by:
        ratios["scale_1_to_4"] = by["4gpu"]["traces_per_s_median"] / max(by["1gpu"]["traces_per_s_median"], 1e-9)
    if "4gpu" in by and "8gpu" in by:
        ratios["scale_4_to_8"] = by["8gpu"]["traces_per_s_median"] / max(by["4gpu"]["traces_per_s_median"], 1e-9)
    elif "4gpu" in by:
        ratios["scale_4_to_8"] = None
        ratios["scale_4_to_8_note"] = "8 free GPUs unavailable"

    # projections from best available rate (prefer 4gpu)
    rate = None
    for k in ("4gpu", "1gpu", "8gpu"):
        if k in by:
            rate = by[k]["traces_per_s_median"]
            rate_setting = k
            break
    projections = {}
    if rate:
        projections = {
            "reference_setting": rate_setting,
            "traces_per_s": rate,
            "eta_100k_traces_h": (100000 / rate) / 3600,
            "eta_confirm_74753_traces_h": (74753 / rate) / 3600,
            "eta_full_INSTANCE_1159249_traces_h": (1159249 / rate) / 3600,
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_manifest_sha256": man_hash,
        "free_gpus_at_start": free,
        "requested_gpu_counts": [1, 4, 8],
        "actual_settings": scaling_rows,
        "scaling_ratios": ratios,
        "projections": projections,
        "hdf5_io_bottleneck_likely": bool(
            ratios.get("scale_4_to_8") is not None and ratios.get("scale_4_to_8", 2) < 1.5
        )
        if ratios.get("scale_4_to_8") is not None
        else "unknown_8gpu_unavailable",
        "runs": all_runs,
        "sota_claim_allowed": False,
    }
    save_json(summary, out / "runtime_benchmark.json")
    pd.DataFrame(scaling_rows).to_csv(out / "runtime_scaling.csv", index=False)

    md = f"""# Stage 7A — Runtime Benchmark

**Weight under test:** PhaseNet-`{args.weight}` (representative SeisBench annotate path)  
**Manifest:** first 10000 of sha256(trace_name)-sorted Stage-6 dev (`{man_hash[:16]}…`)  
**Warmup:** {args.warmup} · **Timed/rep:** {args.n_bench} · **Reps:** {args.reps}  
**Free GPUs at start:** {free}

## Throughput (median of {args.reps} reps)

| Setting | GPUs | traces/s (median) | range |
|--|--:|--:|--|
"""
    for r in scaling_rows:
        if "traces_per_s_median" in r:
            md += f"| {r['setting']} | {r['n_gpus']} | {r['traces_per_s_median']:.3f} | [{r['traces_per_s_min']:.3f}, {r['traces_per_s_max']:.3f}] |\n"
        else:
            md += f"| {r['setting']} | {r.get('actual_gpus')} | n/a | {r.get('status')} |\n"
    md += f"""
## Scaling

- 1→4: `{ratios.get('scale_1_to_4')}`
- 4→8: `{ratios.get('scale_4_to_8')}` ({ratios.get('scale_4_to_8_note', '')})

## Projections (from `{projections.get('reference_setting')}` @ {projections.get('traces_per_s')} traces/s)

- 100k traces: **{projections.get('eta_100k_traces_h'):.2f} h**
- confirm 74753 traces: **{projections.get('eta_confirm_74753_traces_h'):.2f} h**
- full INSTANCE 1,159,249: **{projections.get('eta_full_INSTANCE_1159249_traces_h'):.2f} h**

## HDF5 I/O

Bottleneck flag: `{summary['hdf5_io_bottleneck_likely']}`  
Single shared HDF5; 8-GPU saturation expected if scaling << 2× from 4→8.
"""
    (reports / "stage7A_runtime_benchmark.md").write_text(md)
    print(json.dumps({"ok": True, "ratios": ratios, "projections": projections}, indent=2))


if __name__ == "__main__":
    main()
