#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.phasenet_wrapper import PhaseNetWrapper
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference, true_pick_utc
from earthquake.utils import ensure_dir


def select_diagnostic_traces(events: pd.DataFrame, n: int = 512, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    val = events[events["split"] == "val"].copy()
    # stratify by event: at most a few stations per event
    rows = []
    for eid, g in val.groupby("event_id"):
        take = min(len(g), 3)
        rows.append(g.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    pooled = pd.concat(rows, ignore_index=True)
    # diversify by distance/snr bins if present
    if len(pooled) > n:
        pooled = pooled.sample(n=n, random_state=seed)
    return pooled.reset_index(drop=True)


def plot_example(wave, row, custom, ref, out_path: Path, weight: str):
    t = np.arange(wave.shape[-1]) / float(row["sampling_rate_hz"])
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i, name in enumerate(["E", "N", "Z"]):
        axes[i].plot(t, wave[i], lw=0.6, color="k")
        axes[i].set_ylabel(name)
        for phase, col in (("p", "r"), ("s", "b")):
            s = row.get(f"{phase}_arrival_sample")
            if pd.notna(s):
                axes[i].axvline(float(s) / float(row["sampling_rate_hz"]), color=col, ls="--", alpha=0.8)
    axes[3].plot(t, custom["p"], label="custom P", color="r", alpha=0.8)
    axes[3].plot(t, ref["p"], label="ref P", color="orange", alpha=0.8)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_ylabel("P prob")
    axes[4].plot(t, custom["s"], label="custom S", color="b", alpha=0.8)
    axes[4].plot(t, ref["s"], label="ref S", color="c", alpha=0.8)
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].set_ylabel("S prob")
    axes[4].set_xlabel("Time from waveform start (s)")
    fig.suptitle(
        f"{row['trace_name']} | weight={weight}\n"
        f"in_start={custom.get('input_starttime')} out_start={custom.get('p_output_starttime')} "
        f"in_npts={custom.get('input_npts')} out_npts={custom.get('p_output_npts')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traces", type=int, default=512)
    parser.add_argument("--weight", type=str, default="stead")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "diagnostics")
    examples = ensure_dir(out / "examples")
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    diag = select_diagnostic_traces(events, n=args.n_traces, seed=args.seed)
    diag.to_parquet(out / "diagnostic_traces.parquet", index=False)

    wrapper = PhaseNetWrapper(weight=args.weight, device=args.device, allow_instance=("instance" in args.weight))
    reference = SeisBenchPhaseNetReference(
        weight=args.weight, device=args.device, allow_instance=("instance" in args.weight)
    )
    # share model for fairness on device memory; agreement test still compares APIs
    reference.model = wrapper.model

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    rows = []
    with InstanceHDF5Reader(h5) as reader:
        for i, (_, row) in enumerate(tqdm(diag.iterrows(), total=len(diag), desc="align-diagnose")):
            wave = reader.read_waveform(str(row["trace_name"]))
            custom = wrapper.predict_row(wave, row)
            ref = reference.predict_row(wave, row)
            rec = {
                "trace_name": str(row["trace_name"]),
                "event_id": str(row["event_id"]),
                "weight": args.weight,
                "input_starttime": custom["input_starttime"],
                "output_starttime": custom["p_output_starttime"],
                "input_npts": custom["input_npts"],
                "output_npts": custom["p_output_npts"],
                "sampling_rate_hz": float(row["sampling_rate_hz"]),
                "distance_km": row.get("distance_km"),
                "snr_db": row.get("snr_db"),
                "channel_prefix": row.get("channel_prefix"),
            }
            for phase in ("p", "s"):
                true_s = float(row[f"{phase}_arrival_sample"]) if pd.notna(row.get(f"{phase}_arrival_sample")) else np.nan
                true_utc = true_pick_utc(row["trace_start_time"], true_s, float(row["sampling_rate_hz"]))
                c_sample = float(custom[f"{phase}_pred_sample_on_waveform"])
                r_sample = float(ref[f"{phase}_pred_sample_on_waveform"])
                c_utc = custom[f"{phase}_peak_utc"]
                r_utc = ref[f"{phase}_peak_utc"]
                rec.update(
                    {
                        f"true_{phase}_sample": true_s,
                        f"custom_pred_{phase}_sample": c_sample,
                        f"reference_pred_{phase}_sample": r_sample,
                        f"custom_pred_{phase}_utc": c_utc,
                        f"reference_pred_{phase}_utc": r_utc,
                        f"true_{phase}_utc": str(true_utc) if true_utc is not None else None,
                        f"custom_minus_reference_{phase}_s": (c_sample - r_sample) / float(row["sampling_rate_hz"]),
                        f"custom_residual_{phase}_s": (c_sample - true_s) / float(row["sampling_rate_hz"])
                        if np.isfinite(true_s)
                        else np.nan,
                        f"reference_residual_{phase}_s": (r_sample - true_s) / float(row["sampling_rate_hz"])
                        if np.isfinite(true_s)
                        else np.nan,
                        f"custom_{phase}_prob": float(custom[f"{phase}_peak_probability"]),
                        f"reference_{phase}_prob": float(ref[f"{phase}_peak_probability"]),
                    }
                )
            # P before S check
            if np.isfinite(rec["custom_pred_p_sample"]) and np.isfinite(rec["custom_pred_s_sample"]):
                rec["custom_p_before_s"] = rec["custom_pred_p_sample"] < rec["custom_pred_s_sample"]
            rows.append(rec)

            # save a subset of figures
            if i < 30:
                plot_example(wave, row, custom, ref, examples / f"random_{i:03d}.png", args.weight)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "phasenet_alignment.parquet", index=False)

    summary = {}
    for phase in ("p", "s"):
        d = df[f"custom_minus_reference_{phase}_s"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        summary[phase] = {
            "n": int(np.isfinite(df[f"custom_minus_reference_{phase}_s"]).sum()),
            "max_abs_custom_minus_reference_s": float(np.nanmax(np.abs(d))) if d.size else None,
            "max_abs_custom_minus_reference_samples": float(np.nanmax(np.abs(d)) * 100) if d.size else None,
            "mean_custom_residual_s": float(np.nanmean(df[f"custom_residual_{phase}_s"])),
            "median_custom_residual_s": float(np.nanmedian(df[f"custom_residual_{phase}_s"])),
            "mean_reference_residual_s": float(np.nanmean(df[f"reference_residual_{phase}_s"])),
            "median_reference_residual_s": float(np.nanmedian(df[f"reference_residual_{phase}_s"])),
            "frac_output_start_offset_near_2_5s": float(
                np.mean(
                    np.abs(
                        (
                            pd.to_datetime(df["output_starttime"], utc=True)
                            - pd.to_datetime(df["input_starttime"], utc=True)
                        ).dt.total_seconds()
                        - 2.5
                    )
                    < 0.05
                )
            )
            if len(df)
            else None,
            "median_output_npts": float(np.nanmedian(df["output_npts"])),
            "median_input_npts": float(np.nanmedian(df["input_npts"])),
        }
    summary["agreement_within_1_sample"] = all(
        (
            summary[ph]["max_abs_custom_minus_reference_samples"] is not None
            and summary[ph]["max_abs_custom_minus_reference_samples"] <= 1.0 + 1e-6
        )
        for ph in ("p", "s")
    )
    summary["weight"] = args.weight
    save_json(summary, out / "alignment_summary.json")

    # histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, phase in zip(axes[0], ("p", "s")):
        ax.hist(df[f"reference_residual_{phase}_s"].dropna(), bins=50, alpha=0.8)
        ax.set_title(f"reference residual {phase.upper()}")
        ax.set_xlabel("seconds")
    for ax, phase in zip(axes[1], ("p", "s")):
        ax.hist(df[f"custom_minus_reference_{phase}_s"].dropna(), bins=50, alpha=0.8)
        ax.set_title(f"custom - reference {phase.upper()}")
        ax.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out / "residual_histogram.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, phase in zip(axes, ("p", "s")):
        ax.scatter(df[f"true_{phase}_sample"], df[f"reference_residual_{phase}_s"], s=8, alpha=0.5)
        ax.set_xlabel(f"true {phase} sample")
        ax.set_ylabel("reference residual (s)")
        ax.set_title(f"residual vs true sample ({phase})")
    fig.tight_layout()
    fig.savefig(out / "residual_vs_true_sample.png", dpi=140)
    plt.close(fig)

    # extreme examples
    for phase, tag in (("p", "p_worst"), ("s", "s_worst")):
        sub = df.dropna(subset=[f"reference_residual_{phase}_s"]).copy()
        sub["abs"] = sub[f"reference_residual_{phase}_s"].abs()
        worst = sub.nlargest(10, "abs")
        with InstanceHDF5Reader(h5) as reader:
            for j, (_, r) in enumerate(worst.iterrows()):
                row = diag[diag.trace_name == r.trace_name].iloc[0]
                wave = reader.read_waveform(str(row.trace_name))
                custom = wrapper.predict_row(wave, row)
                ref = reference.predict_row(wave, row)
                plot_example(wave, row, custom, ref, examples / f"{tag}_{j:02d}.png", args.weight)

    # low SNR
    if "snr_db" in diag.columns:
        low = diag.nsmallest(10, "snr_db")
        with InstanceHDF5Reader(h5) as reader:
            for j, (_, row) in enumerate(low.iterrows()):
                wave = reader.read_waveform(str(row.trace_name))
                custom = wrapper.predict_row(wave, row)
                ref = reference.predict_row(wave, row)
                plot_example(wave, row, custom, ref, examples / f"lowsnr_{j:02d}.png", args.weight)

    print(json.dumps(summary, indent=2))
    if not summary["agreement_within_1_sample"]:
        raise SystemExit("FAIL: custom wrapper vs annotate disagree by >1 sample; fix before finetune")


if __name__ == "__main__":
    main()
