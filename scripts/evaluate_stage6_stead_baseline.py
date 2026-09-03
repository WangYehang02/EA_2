#!/usr/bin/env python
"""Reproduce STEAD PhaseNet baseline on Stage 6 dev (annotate + UTC remap + metrics + K oracle)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.fusion.peak_candidates import extract_candidates
from earthquake.metrics import audit_pick_errors, match_picks
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.splits import assert_confirm_sealed, load_stage6_trace_names
from earthquake.utils import ensure_dir


def oracle_from_candidates(cands_samples: list[np.ndarray], true: np.ndarray, sr: np.ndarray) -> dict:
    pred = []
    for i, t in enumerate(true):
        c = np.asarray(cands_samples[i], dtype=float)
        c = c[np.isfinite(c)]
        if not np.isfinite(t) or c.size == 0:
            pred.append(np.nan)
            continue
        ae = np.abs((c - t) / sr[i])
        pred.append(float(c[int(np.argmin(ae))]))
    m = match_picks(np.asarray(pred), true, sr, windows_s=(0.1, 0.5))
    # candidate recall @0.5
    hit = 0
    labeled = 0
    for i, t in enumerate(true):
        if not np.isfinite(t):
            continue
        labeled += 1
        c = np.asarray(cands_samples[i], dtype=float)
        c = c[np.isfinite(c)]
        if c.size and float(np.min(np.abs((c - t) / sr[i]))) <= 0.5:
            hit += 1
    return {
        "candidate_recall@0.5": float(hit / max(labeled, 1)),
        "oracle_f1@0.1": float(m["f1@0.1s"]),
        "oracle_f1@0.5": float(m["f1@0.5s"]),
        "oracle_e2e_p95": float(m["e2e_p95_ae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="stage6_dev")
    parser.add_argument("--weight", default="stead")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-traces", type=int, default=-1)
    parser.add_argument("--k-list", default="1,3,5,10")
    args = parser.parse_args()

    if args.subset == "stage6_internal_confirm":
        assert_confirm_sealed()

    if "instance" in str(args.weight).lower():
        raise SystemExit("Refusing SeisBench instance weights as formal Stage6 baseline")

    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "baseline_stead")
    traces = load_stage6_trace_names(args.subset)
    if args.max_traces > 0:
        traces = traces[: args.max_traces]

    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    meta = events[events["trace_name"].astype(str).isin(traces)].copy()
    meta = meta.set_index("trace_name").loc[traces].reset_index()

    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    ref = SeisBenchPhaseNetReference(weight=args.weight, device=args.device)

    rows = []
    cand_by_k = {int(k): [] for k in args.k_list.split(",")}
    true_s, sr_all = [], []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"stead:{args.subset}"):
        wave = reader.read_waveform(str(row.trace_name))
        pred = ref.predict_row(wave, row, remap_to_waveform=True)
        # probability curve for S if available
        s_proba = pred.get("s_proba_on_waveform")
        if s_proba is None:
            s_proba = pred.get("s")
        sr = float(row.sampling_rate_hz)
        true = float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan
        true_s.append(true)
        sr_all.append(sr)
        top1 = float(pred["s_pred_sample_on_waveform"])
        rows.append(
            {
                "trace_name": str(row.trace_name),
                "event_id": str(row.event_id),
                "pred_s_sample": top1,
                "pred_p_sample": float(pred["p_pred_sample_on_waveform"]),
                "true_s_sample": true,
                "true_p_sample": float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan,
                "sampling_rate_hz": sr,
            }
        )
        if s_proba is None:
            for k in cand_by_k:
                cand_by_k[k].append(np.array([top1], dtype=float))
        else:
            for k in cand_by_k:
                cands = extract_candidates(np.asarray(s_proba, dtype=float), phase="S", k=k)
                samp = np.array([c.sample_index for c in cands], dtype=float) if cands else np.array([top1])
                cand_by_k[k].append(samp)

    reader.close()
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{args.subset}_stead_picks.parquet", index=False)

    true = df["true_s_sample"].to_numpy()
    pred = df["pred_s_sample"].to_numpy()
    sr = df["sampling_rate_hz"].to_numpy()
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.2, 0.5))
    a = audit_pick_errors(pred, true, sr, window_s=0.5)
    # P
    mp = match_picks(df["pred_p_sample"].to_numpy(), df["true_p_sample"].to_numpy(), sr, windows_s=(0.1, 0.5))

    oracles = {f"K={k}": oracle_from_candidates(cand_by_k[k], true, sr) for k in cand_by_k}
    result = {
        "subset": args.subset,
        "weight": args.weight,
        "n_traces": len(df),
        "n_events": int(df.event_id.nunique()),
        "S": {
            "f1@0.1": m["f1@0.1s"],
            "f1@0.2": m.get("f1@0.2s"),
            "f1@0.5": m["f1@0.5s"],
            "e2e_mae": m["e2e_mae"],
            "e2e_p95": m["e2e_p95_ae"],
            "matched_p95": a["matched_timing"]["p95_ae"],
            "wrong_peak_rate": a["n_wrong_peak_beyond_tol"] / max(a["n_labeled"], 1),
            "miss_rate": a["n_missed_pick"] / max(a["n_labeled"], 1),
        },
        "P": {"f1@0.1": mp["f1@0.1s"], "f1@0.5": mp["f1@0.5s"], "e2e_p95": mp["e2e_p95_ae"]},
        "oracles": oracles,
        "component_order_note": "SeisBench annotate path with UTC remap (ENZ->model order handled in reference)",
        "label_order_expected": "PSN",
    }
    save_json(result, out / f"{args.subset}_stead_metrics.json")
    # completion marker
    (out / f"{args.subset}_stead.DONE").write_text("ok\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
