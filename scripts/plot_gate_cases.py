#!/usr/bin/env python
"""Plot representative Stage-3 gate cases (waveform + candidates + picks)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_yaml_config, resolve_instance_root
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.gating.cache_io import build_cand_index
from earthquake.utils import ensure_dir


def _pick_cases(picks: pd.DataFrame) -> dict[str, pd.Series]:
    sr = picks["sampling_rate_hz"].to_numpy()
    true = picks["true_s_sample"].to_numpy()
    picks = picks.copy()
    picks["e_pn"] = np.abs(picks["pred_s_phasenet"] - true) / sr
    picks["e_fr"] = np.abs(picks["pred_s_fixed_rescore"] - true) / sr
    picks["e_lg"] = np.abs(picks["pred_s_learned_gate"] - true) / sr
    cases = {}
    # 1 PhaseNet wrong, gate fixes
    m = (picks.e_pn > 0.5) & (picks.e_lg <= 0.5)
    if m.any():
        cases["pn_wrong_gate_fix"] = picks[m].sort_values("e_pn", ascending=False).iloc[0]
    # 2 PN correct, history/fixed wrong, gate falls back
    m = (picks.e_pn <= 0.5) & (picks.e_fr > 0.5) & (picks.e_lg <= 0.5)
    if m.any():
        cases["pn_ok_hist_wrong_gate_fallback"] = picks[m].iloc[0]
    # 3 fixed wrong, gate correct
    m = (picks.e_fr > 0.5) & (picks.e_lg <= 0.5)
    if m.any():
        cases["fixed_wrong_gate_ok"] = picks[m].iloc[0]
    # 4 gate wrong, fixed ok
    m = (picks.e_lg > 0.5) & (picks.e_fr <= 0.5)
    if m.any():
        cases["gate_wrong_fixed_ok"] = picks[m].iloc[0]
    # 5 oracle candidate also fails (no good peak in topK) — approximate: both gate and pn and fr bad and oracle cand bad
    if "pred_s_oracle_candidate" in picks.columns:
        picks["e_ora"] = np.abs(picks["pred_s_oracle_candidate"] - true) / sr
        m = picks.e_ora > 0.5
        if m.any():
            cases["no_correct_candidate"] = picks[m].sort_values("e_ora", ascending=False).iloc[0]
    return cases


def _plot_one(row, wave, cands, out_path: Path):
    sr = float(row.sampling_rate_hz)
    t = np.arange(wave.shape[-1]) / sr
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    labels = ["E", "N", "Z"]
    for i in range(min(3, wave.shape[0])):
        axes[i].plot(t, wave[i], lw=0.6, color="0.2")
        axes[i].set_ylabel(labels[i])
        for nm, col, sty in [
            ("P", row.get("true_p_sample", np.nan) if "true_p_sample" in row else np.nan, "g"),
            ("S", row.true_s_sample, "r"),
        ]:
            if np.isfinite(float(sty if False else col)):
                axes[i].axvline(float(col) / sr, color="C1" if nm == "P" else "C3", ls="--", lw=1, alpha=0.8)
    ax = axes[3]
    ax.set_ylabel("S cands")
    for c in cands:
        ax.axvline(c.sample_index / sr, color="0.5", alpha=0.4)
        ax.plot(c.sample_index / sr, c.peak_probability, "o", color="0.3")
    for name, val, color in [
        ("PN", row.pred_s_phasenet, "C0"),
        ("fixed", row.pred_s_fixed_rescore, "C1"),
        ("gate", row.pred_s_learned_gate, "C2"),
        ("trueS", row.true_s_sample, "C3"),
    ]:
        if np.isfinite(float(val)):
            ax.axvline(float(val) / sr, color=color, lw=1.5, label=name)
    ax.legend(loc="upper right", fontsize=8)
    g = float(row.gate) if np.isfinite(row.gate) else float("nan")
    fig.suptitle(f"{row.trace_name}  gate={g:.3f}  hist_n={row.get('history_count', '?')}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_debug.yaml")
    parser.add_argument("--picks", default="artifacts/results/stage3/learned_gate_picks_test.parquet")
    args = parser.parse_args()
    _ = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage3" / "gate_cases")
    picks = pd.read_parquet(ROOT / args.picks)
    if "seed" in picks.columns:
        picks = picks[picks.seed == int(picks.seed.iloc[0])].copy()
    # attach history meta if available
    for bp in [
        artifacts_dir() / "results" / "stage3" / "gate_baselines_debug.parquet",
        artifacts_dir() / "results" / "stage3" / "gate_baselines_catalog.parquet",
    ]:
        if bp.exists():
            meta = pd.read_parquet(bp)
            picks = picks.merge(meta[["trace_name", "history_count", "history_mad", "abcd_class"]], on="trace_name", how="left")
            break
    cases = _pick_cases(picks)
    cand_path = artifacts_dir() / "candidates" / "stage3_phasenet_cache_candidates.parquet"
    if not cand_path.exists():
        cand_path = artifacts_dir() / "results" / "stage2" / "phasenet_fixed_cache_candidates.parquet"
    cidx = build_cand_index(pd.read_parquet(cand_path)) if cand_path.exists() else {}
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet").set_index("trace_name")
    with InstanceHDF5Reader(h5) as reader:
        for name, row in cases.items():
            wave = reader.read_waveform(str(row.trace_name))
            # wave may be dict or array
            if isinstance(wave, dict):
                arr = np.stack([wave[k] for k in ("E", "N", "Z") if k in wave], axis=0)
            else:
                arr = np.asarray(wave)
            cands = cidx.get((str(row.trace_name), "s"), [])
            if "true_p_sample" not in row.index and str(row.trace_name) in events.index:
                row = row.copy()
                row["true_p_sample"] = events.loc[str(row.trace_name), "p_arrival_sample"]
            _plot_one(row, arr, cands, out / f"{name}.png")
            print({"saved": str(out / f"{name}.png")}, flush=True)
    print({"n_cases": len(cases), "out": str(out)}, flush=True)


if __name__ == "__main__":
    main()
