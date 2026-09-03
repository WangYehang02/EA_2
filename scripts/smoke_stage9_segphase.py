#!/usr/bin/env python
"""SegPhase smoke + alignment sanity (Stage 9)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import resolve_instance_root
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.stage9.segphase_adapter import SegPhaseConfig, SegPhasePicker, enz_to_ud_ns_ew, window_starts


def main() -> int:
    out = ROOT / "artifacts/results/stage9"
    reports = ROOT / "reports/stage9"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # use free GPU 0
    cfg = SegPhaseConfig(device=device, window_scheme="A")
    picker = SegPhasePicker(cfg)

    # reject random: compare state_dict norms vs loaded
    loaded_norm = sum(p.detach().float().norm().item() for p in picker.model.parameters())
    rnd = type(picker.model)(in_length=3000, in_channels=3, class_num=3, strides=[3, 2, 2], kernel_size=3)
    rnd_norm = sum(p.detach().float().norm().item() for p in rnd.parameters())

    man = pd.read_csv(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv")
    # pick cases by S sample position
    sr = man.sampling_rate_hz.astype(float)
    s = man.s_arrival_sample.astype(float)
    t_s = s / sr
    cases = []
    for name, lo, hi in [
        ("S_in_first_30s", 0, 30),
        ("near_30s_boundary", 28, 32),
        ("near_60s", 55, 65),
        ("near_90s", 85, 95),
        ("near_end", 100, 119),
    ]:
        sub = man[(t_s >= lo) & (t_s < hi)]
        if len(sub):
            cases.append((name, sub.iloc[0]))

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    rows = []
    preds_a = []
    try:
        for name, row in cases:
            wave = reader.read_waveform(str(row.trace_name))
            assert wave.shape[0] == 3
            zne = enz_to_ud_ns_ew(wave)
            assert np.allclose(zne[0], wave[2])  # UD=Z
            assert np.allclose(zne[1], wave[1])  # NS=N
            assert np.allclose(zne[2], wave[0])  # EW=E
            out1 = picker.predict_s(wave, threshold=0.1, top_k=5)
            # batch consistency: stack same window twice
            from earthquake.stage9.segphase_adapter import zscore_channels

            chunk = zscore_channels(zne[:, :3000])
            x = torch.from_numpy(np.stack([chunk, chunk])).float().to(picker.device)
            with torch.inference_mode():
                pred = picker.model(x).detach().cpu().numpy()
            batch_ok = np.allclose(pred[0], pred[1], atol=1e-5, equal_nan=True)
            # repeat determinism
            out2 = picker.predict_s(wave, threshold=0.1, top_k=5)
            det_ok = (
                (np.isnan(out1["pred_s_sample"]) and np.isnan(out2["pred_s_sample"]))
                or abs(float(out1["pred_s_sample"]) - float(out2["pred_s_sample"])) < 1e-6
            )
            true_s = float(row.s_arrival_sample)
            pred_s = float(out1["pred_s_sample"])
            err = abs(pred_s - true_s) if np.isfinite(pred_s) else float("nan")
            rows.append(
                {
                    "case": name,
                    "trace_name": str(row.trace_name),
                    "true_s_sample": true_s,
                    "pred_s_sample": pred_s,
                    "prob": out1["s_peak_probability"],
                    "abs_err_samples": err,
                    "abs_err_s": err / float(row.sampling_rate_hz) if np.isfinite(err) else float("nan"),
                    "batch_ok": batch_ok,
                    "determinism_ok": det_ok,
                    "n_peaks": out1.get("n_peaks", 0),
                    "output_len": int(out1["s_proba"].shape[0]),
                }
            )
            preds_a.append(out1)
            # scheme B quick compare on same wave
            picker.cfg.window_scheme = "B"
            outb = picker.predict_s(wave, threshold=0.1, top_k=1)
            picker.cfg.window_scheme = "A"
            rows[-1]["pred_s_schemeB"] = outb["pred_s_sample"]
    finally:
        reader.close()

    # CPU vs GPU if cuda
    cpu_gpu_ok = True
    if torch.cuda.is_available():
        p_cpu = SegPhasePicker(SegPhaseConfig(device="cpu", window_scheme="A"))
        wave0 = InstanceHDF5Reader(h5).open()
        try:
            w = wave0.read_waveform(str(cases[0][1].trace_name))
            a = picker.predict_s(w, threshold=0.1)
            b = p_cpu.predict_s(w, threshold=0.1)
            if np.isfinite(a["pred_s_sample"]) and np.isfinite(b["pred_s_sample"]):
                cpu_gpu_ok = abs(a["pred_s_sample"] - b["pred_s_sample"]) <= 1.0  # ≤1 sample
            # also compare proba max
            if a["s_proba"].shape == b["s_proba"].shape:
                mad = float(np.max(np.abs(a["s_proba"] - b["s_proba"])))
                rows.append({"case": "cpu_gpu_proba_mad", "abs_err_samples": mad})
        finally:
            wave0.close()

    finite_errs = [r["abs_err_samples"] for r in rows if r.get("case", "").startswith("S_") or "near" in r.get("case", "")]
    finite_errs = [e for e in finite_errs if isinstance(e, (float, int)) and np.isfinite(e)]
    # alignment gate: systematic bias — use median of signed errors if available
    # For smoke we check whether any case has absurd offset (>100 samples unexplained)
    max_err = max(finite_errs) if finite_errs else float("nan")
    # Gate: if ALL predictions exist and median abs err huge AND consistent → fail. Soft for smoke.
    alignment_gate = "PASS"
    if not np.isfinite(loaded_norm) or loaded_norm < 1e-3:
        alignment_gate = "FAIL"
    if abs(loaded_norm - rnd_norm) < 1e-2:
        alignment_gate = "FAIL_RANDOM_LIKE"
    # if every finite err > 500 samples (~5s) likely offset bug
    if finite_errs and all(e > 500 for e in finite_errs):
        alignment_gate = "FAIL"

    smoke = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SegPhase-100Hz",
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_sha256": picker.checkpoint_sha256,
        "device": str(picker.device),
        "loaded_param_norm": loaded_norm,
        "random_param_norm": rnd_norm,
        "checkpoint_not_random": abs(loaded_norm - rnd_norm) > 1.0,
        "channel_map": "INSTANCE ENZ -> UD,NS,EW = Z,N,E",
        "window_scheme_default": "A",
        "window_starts_A": window_starts("A"),
        "window_starts_B": window_starts("B"),
        "official_peak_height": 0.1,
        "batch_consistency": all(r.get("batch_ok", True) for r in rows if "batch_ok" in r),
        "determinism": all(r.get("determinism_ok", True) for r in rows if "determinism_ok" in r),
        "cpu_gpu_ok": cpu_gpu_ok,
        "max_abs_err_samples_among_cases": max_err,
        "alignment_gate": alignment_gate,
        "cases": rows,
        "enter_full_eval": alignment_gate == "PASS",
    }
    (out / "segphase_smoke_metrics.json").write_text(json.dumps(smoke, indent=2, default=float) + "\n")
    pd.DataFrame(rows).to_csv(out / "segphase_alignment_cases.csv", index=False)

    md = f"""# Stage 9 — SegPhase Alignment Sanity

**alignment_gate:** `{alignment_gate}`  
**checkpoint:** `{cfg.checkpoint}`  
**SHA256:** `{picker.checkpoint_sha256}`  
**checkpoint_not_random:** `{smoke['checkpoint_not_random']}`

## Mapping

- Input shape: `(B,3,3000)` @ 100 Hz
- INSTANCE ENZ → **UD,NS,EW = Z,N,E**
- Normalize: per-channel z-score
- Peaks: official `find_peaks(distance=100, height=thr)`
- Window A starts: `{smoke['window_starts_A']}`
- Window B starts: `{smoke['window_starts_B']}`

## Smoke cases

See `artifacts/results/stage9/segphase_alignment_cases.csv`.

## Batch / determinism / CPU-GPU

- batch_consistency: `{smoke['batch_consistency']}`
- determinism: `{smoke['determinism']}`
- cpu_gpu_ok: `{cpu_gpu_ok}`
"""
    (reports / "segphase_alignment_sanity.md").write_text(md)
    print(json.dumps({"alignment_gate": alignment_gate, "enter_full_eval": smoke["enter_full_eval"], "max_err": max_err}, indent=2))
    return 0 if alignment_gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
