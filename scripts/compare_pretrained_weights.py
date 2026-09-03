#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import match_picks
from earthquake.models.phasenet_wrapper import list_pretrained_weights
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.utils import ensure_dir


def eval_weight(weight: str, diag: pd.DataFrame, device: str, allow_instance: bool) -> dict:
    ref = SeisBenchPhaseNetReference(weight=weight, device=device, allow_instance=allow_instance)
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    preds = {ph: [] for ph in ("p", "s")}
    trues = {ph: [] for ph in ("s", "p")}
    srs = []
    with InstanceHDF5Reader(h5) as reader:
        for _, row in tqdm(diag.iterrows(), total=len(diag), desc=f"weight:{weight}"):
            wave = reader.read_waveform(str(row["trace_name"]))
            out = ref.predict_row(wave, row)
            srs.append(float(row["sampling_rate_hz"]))
            for ph in ("p", "s"):
                preds[ph].append(float(out[f"{ph}_pred_sample_on_waveform"]))
                trues[ph].append(
                    float(row[f"{ph}_arrival_sample"]) if pd.notna(row.get(f"{ph}_arrival_sample")) else np.nan
                )
    metrics = {"weight": weight, "n": len(diag), "diagnostic_only_data_leakage": allow_instance and "instance" in weight.lower()}
    for ph in ("p", "s"):
        m = match_picks(np.asarray(preds[ph]), np.asarray(trues[ph]), np.asarray(srs))
        metrics[f"{ph}_f1@0.1s"] = m["f1@0.1s"]
        metrics[f"{ph}_f1@0.5s"] = m["f1@0.5s"]
        metrics[f"{ph}_mae"] = m["mae"]
        metrics[f"{ph}_median_ae"] = m["median_ae"]
        metrics[f"{ph}_p95_ae"] = m["p95_ae"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", default=["stead", "ethz", "scedc", "instance"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-traces", type=int, default=512)
    args = parser.parse_args()

    available = list_pretrained_weights()
    print("Available:", available)
    weights = [w for w in args.weights if w in available]
    missing = [w for w in args.weights if w not in available]
    if missing:
        print("Skipping missing weights:", missing)

    out = ensure_dir(artifacts_dir() / "diagnostics")
    diag_path = out / "diagnostic_traces.parquet"
    if diag_path.exists():
        diag = pd.read_parquet(diag_path).head(args.max_traces)
    else:
        events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
        diag = events[events.split == "val"].sample(n=min(args.max_traces, (events.split == "val").sum()), random_state=0)

    rows = []
    for w in weights:
        allow = "instance" in w.lower()
        rows.append(eval_weight(w, diag, args.device, allow_instance=allow))
    df = pd.DataFrame(rows)
    df.to_csv(out / "pretrained_weight_comparison.csv", index=False)

    # Decision logic
    inst = df[df["weight"].str.contains("instance", case=False)]
    ext = df[~df["weight"].str.contains("instance", case=False)]
    decision = {"status": "unknown", "message": ""}
    if len(inst):
        inst_mae = float(inst.iloc[0]["p_mae"])
        inst_f1 = float(inst.iloc[0]["p_f1@0.5s"])
        ext_mae = float(ext["p_mae"].min()) if len(ext) else np.nan
        if (inst_mae > 15 and inst_f1 < 0.05) or (not np.isfinite(inst_mae)):
            decision = {
                "status": "pipeline_bug",
                "message": "instance diagnostic also poor (~20s MAE / ~0 F1) => inference/alignment/eval bug; STOP before finetune",
                "instance_p_mae": inst_mae,
                "instance_p_f1_0.5": inst_f1,
            }
        elif inst_mae < 5 and (not np.isfinite(ext_mae) or ext_mae > 10):
            decision = {
                "status": "domain_shift",
                "message": "instance recovers; external weights poor => pipeline OK, main issue is domain shift",
                "instance_p_mae": inst_mae,
                "best_external_p_mae": ext_mae,
            }
        else:
            decision = {
                "status": "mixed",
                "message": "See comparison CSV; inspect plots before concluding",
                "instance_p_mae": inst_mae,
                "best_external_p_mae": ext_mae,
            }
    save_json(decision, out / "pretrained_weight_decision.json")
    print(df.to_string(index=False))
    print(decision)
    if decision["status"] == "pipeline_bug":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
