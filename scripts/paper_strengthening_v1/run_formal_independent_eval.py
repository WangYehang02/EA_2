#!/usr/bin/env python
"""One-shot A–E formal evaluation on frozen independent-period READY package."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.paper_strengthening import (
    LinearPairwiseModel,
    metrics_from_pred,
    paired_event_bootstrap_delta,
    pred_base_tau_c1c2,
    pred_fixed,
    pred_resid_control,
    switch_mechanism,
)
from earthquake.pairwise.fulldev import apply_switch, load_scalar_model, predict_p_switch

OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
REPORT = ROOT / "reports" / "paper_strengthening_v1"
C_SIGMA = 0.25
C_LAM = 4.0
SCALAR_CKPT = ROOT / "artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt"
SCALAR_TAU = 0.50
LIN_PATH = ROOT / "artifacts/models/paper_strengthening_v1/linear_pairwise_SELECTED.json"
N_BOOT = 5000


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_linear(path: Path) -> tuple[LinearPairwiseModel, float]:
    blob = json.loads(path.read_text())
    model = LinearPairwiseModel(
        coef=np.asarray(blob["coef"], dtype=np.float64),
        mean=np.asarray(blob["mean"], dtype=np.float64),
        scale=np.asarray(blob["scale"], dtype=np.float64),
        C=float(blob["C"]),
    )
    return model, float(blob["tau"])


def main() -> int:
    ready_path = OUT / "READY.json"
    if not ready_path.exists():
        raise SystemExit("READY.json missing")
    ready = json.loads(ready_path.read_text())
    if ready.get("status") != "READY":
        raise SystemExit("READY status not READY")
    # verify locks
    data_lock = json.loads(Path(ready["data_lock_json"]).read_text())
    if data_lock.get("lock_body_sha256") != Path(OUT / "locks" / "DATA.LOCK.sha256").read_text().strip():
        # recompute body without lock_body field
        body = {k: v for k, v in data_lock.items() if k != "lock_body_sha256"}
        digest = hashlib.sha256((json.dumps(body, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        if digest != Path(OUT / "locks" / "DATA.LOCK.sha256").read_text().strip():
            print("WARN: data lock sidecar mismatch; continuing with recorded READY hashes", flush=True)

    pairs = pd.read_parquet(ready["pairs_parquet"])
    if sha_file(Path(ready["pairs_parquet"])) != ready["pairs_sha256"]:
        raise SystemExit("pairs sha256 mismatch vs READY")

    # --- predictions WITHOUT looking at metrics first; save then score ---
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    preds = {}
    preds["A_fixed"] = pred_fixed(pairs)
    preds["B_resid"] = pred_resid_control(pairs, lam=1.0, sigma=0.5)
    preds["C_best"] = pred_base_tau_c1c2(pairs, sigma_s=C_SIGMA, lambda_history=C_LAM)
    lin, lin_tau = load_linear(LIN_PATH)
    preds["D_linear"], _ = lin.predict(pairs, lin_tau)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_scalar_model(SCALAR_CKPT, device)
    p_sw = predict_p_switch(model, pairs, device)
    preds["E_scalar"], _ = apply_switch(pairs, p_sw, SCALAR_TAU)

    # freeze prediction arrays before metric join
    pred_table = pairs[["trace_name", "event_id"]].copy()
    for name, arr in preds.items():
        pred_table[name] = arr
        np.save(pred_dir / f"{name}.npy", arr)
    pred_parquet = pred_dir / "predictions_A_to_E.parquet"
    pred_table.to_parquet(pred_parquet, index=False)
    pred_lock = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": int(len(pairs)),
        "methods": list(preds.keys()),
        "predictions_sha256": sha_file(pred_parquet),
        "files": {k: sha_file(pred_dir / f"{k}.npy") for k in preds},
        "C": {"sigma_s": C_SIGMA, "lambda_history": C_LAM},
        "scalar": {"ckpt": str(SCALAR_CKPT), "ckpt_sha256": sha_file(SCALAR_CKPT), "tau": SCALAR_TAU},
        "linear": {"path": str(LIN_PATH), "sha256": sha_file(LIN_PATH), "tau": lin_tau},
        "labels_joined_after_freeze": True,
    }
    (OUT / "locks" / "PREDICTIONS.LOCK.json").write_text(json.dumps(pred_lock, indent=2) + "\n")

    # --- join labels / metrics ---
    true = pairs["true_s_sample"].to_numpy(float)
    sr = pairs["sampling_rate_hz"].to_numpy(float)
    eids = pairs["event_id"].to_numpy(str)

    rows = []
    for name, arr in preds.items():
        # nan preds (outside window if included) -> treat as miss by setting far away
        pred = np.asarray(arr, dtype=float).copy()
        bad = ~np.isfinite(pred)
        if bad.any():
            pred[bad] = true[bad] + 1e9  # guaranteed miss
        met = metrics_from_pred(pred, true, sr)
        mech = switch_mechanism(pairs, pred)
        rows.append({"method": name, **met, **{f"mech_{k}": v for k, v in mech.items()}})
    metrics_df = pd.DataFrame(rows)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT / "results" / "metrics_A_to_E.csv", index=False)

    # primary: E - C
    boot_primary = paired_event_bootstrap_delta(
        eids, preds["E_scalar"], preds["C_best"], true, sr, n_boot=N_BOOT, seed=42
    )
    # also MAE / frac>5s deltas (tail secondary; not replacing primary)
    def metric_delta(name_a, name_b, key):
        pa = np.asarray(preds[name_a], float).copy()
        pb = np.asarray(preds[name_b], float).copy()
        for p in (pa, pb):
            bad = ~np.isfinite(p)
            p[bad] = true[bad] + 1e9
        ma = metrics_from_pred(pa, true, sr)
        mb = metrics_from_pred(pb, true, sr)
        return float(ma[key] - mb[key]), ma[key], mb[key]

    # bootstrap for tail metrics
    def boot_metric(pred_a, pred_b, key, n_boot=N_BOOT, seed=42):
        rng = np.random.default_rng(seed)
        events = np.unique(eids)
        ev_to_idx = {}
        for i, e in enumerate(eids):
            ev_to_idx.setdefault(e, []).append(i)
        pa = np.asarray(pred_a, float).copy()
        pb = np.asarray(pred_b, float).copy()
        for p in (pa, pb):
            bad = ~np.isfinite(p)
            p[bad] = true[bad] + 1e9
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            samp = rng.choice(events, size=len(events), replace=True)
            idx = np.concatenate([ev_to_idx[e] for e in samp])
            ma = metrics_from_pred(pa[idx], true[idx], sr[idx])[key]
            mb = metrics_from_pred(pb[idx], true[idx], sr[idx])[key]
            deltas[b] = ma - mb
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        return {"mean": float(deltas.mean()), "ci95": [float(lo), float(hi)], "n_boot": n_boot}

    d_f1, e_f1, c_f1 = metric_delta("E_scalar", "C_best", "f1@0.5")
    d_mae, e_mae, c_mae = metric_delta("E_scalar", "C_best", "detected_ae_mae")
    d_gt5, e_gt5, c_gt5 = metric_delta("E_scalar", "C_best", "frac_err_gt_5s")
    boot_mae = boot_metric(preds["E_scalar"], preds["C_best"], "detected_ae_mae")
    boot_gt5 = boot_metric(preds["E_scalar"], preds["C_best"], "frac_err_gt_5s")

    # secondary preregistered: C - fixed
    boot_sec = paired_event_bootstrap_delta(
        eids, preds["C_best"], preds["A_fixed"], true, sr, n_boot=N_BOOT, seed=42
    )
    d_cf, c_f1b, a_f1 = metric_delta("C_best", "A_fixed", "f1@0.5")

    comparisons = {
        "primary_scalar_minus_C": {
            "metric": "F1@0.5",
            "delta": d_f1,
            "E_f1@0.5": e_f1,
            "C_f1@0.5": c_f1,
            "bootstrap": boot_primary,
            "tail_secondary": {
                "detected_ae_mae": {"delta": d_mae, "E": e_mae, "C": c_mae, "bootstrap": boot_mae},
                "frac_err_gt_5s": {"delta": d_gt5, "E": e_gt5, "C": c_gt5, "bootstrap": boot_gt5},
            },
            "note": "Tail metrics cannot replace primary endpoint.",
        },
        "secondary_C_minus_fixed": {
            "metric": "F1@0.5",
            "delta": d_cf,
            "C_f1@0.5": c_f1b,
            "A_f1@0.5": a_f1,
            "bootstrap": boot_sec,
            "preregistered": True,
        },
    }
    (OUT / "results" / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    metrics_df.to_csv(REPORT / "independent_period_metrics_A_to_E.csv", index=False)
    (REPORT / "independent_period_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    # judgment draft numbers only
    summary = {
        "run_status": "COMPLETED",
        "n_pairs": int(len(pairs)),
        "n_events": int(pairs["event_id"].nunique()),
        "metrics_csv": str(OUT / "results" / "metrics_A_to_E.csv"),
        "comparisons": comparisons,
        "prediction_lock": pred_lock,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "results" / "EVAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"primary_delta_f1": d_f1, "boot": boot_primary, "secondary_delta_f1": d_cf, "boot_sec": boot_sec}, indent=2))
    return 0


if __name__ == "__main__":
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    raise SystemExit(main())
