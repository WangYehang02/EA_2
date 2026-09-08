#!/usr/bin/env python
"""Complete scalar vs C comparison from frozen paper_strengthening_v1 predictions.

Does NOT re-run models. Bootstrap on frozen preds is labeled exploratory/dev.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks
from earthquake.paper_strengthening import paired_event_bootstrap_delta, switch_mechanism
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "paper_strengthening_v1"
REPORT = ROOT / "reports" / "paper_strengthening_v1"
PRED = OUT / "predictions"
PRACTICAL = 0.003


def frac_tail(err: np.ndarray, thr: float) -> float:
    if err.size == 0:
        return float("nan")
    return float(np.mean(err > thr))


def full_metrics(pred: np.ndarray, true: np.ndarray, sr: np.ndarray, pairs: pd.DataFrame) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    both = np.isfinite(pred) & np.isfinite(true)
    err = np.abs((pred[both] - true[both]) / sr[both])
    mech = switch_mechanism(pairs, pred)
    return {
        "f1@0.5": float(m["f1@0.5s"]),
        "f1@0.1": float(m["f1@0.1s"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_mae": float(m["detected_ae_mae"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "frac_err_gt_1s": frac_tail(err, 1.0),
        "frac_err_gt_5s": frac_tail(err, 5.0),
        "frac_err_gt_10s": frac_tail(err, 10.0),
        "miss_rate": float(m["miss_rate"]),
        "n_eval": int(m["n_eval"]),
        "n_pred": int(m["n_pred"]),
        "n_miss": int(m["n_miss"]),
        "coverage_pred": float(m["n_pred"] / max(m["n_eval"], 1)),
        "fixes": mech["fixes"],
        "breaks": mech["breaks"],
        "net_fixes": mech["net_fixes"],
        "n_switch": mech["n_switch"],
        "n_ge2": mech["n_ge2"],
        "switch_precision_state_changing": mech["switch_precision_state_changing"],
        "p95_scope": m["p95_scope"],
    }


def boot_metric(
    event_ids: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    sr: np.ndarray,
    *,
    key: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Paired event bootstrap of metric_a - metric_b for selected keys."""
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    ev_to_idx = {}
    for i, e in enumerate(event_ids.astype(str)):
        ev_to_idx.setdefault(e, []).append(i)

    def one(pred, idx):
        m = match_picks(pred[idx], true[idx], sr[idx], windows_s=(0.1, 0.5))
        both = np.isfinite(pred[idx]) & np.isfinite(true[idx])
        err = np.abs((pred[idx][both] - true[idx][both]) / sr[idx][both])
        bundle = {
            "f1@0.5": float(m["f1@0.5s"]),
            "f1@0.1": float(m["f1@0.1s"]),
            "detected_ae_median": float(m["detected_ae_median"]),
            "detected_ae_mae": float(m["detected_ae_mae"]),
            "detected_ae_p95": float(m["detected_ae_p95"]),
            "frac_err_gt_1s": frac_tail(err, 1.0),
            "frac_err_gt_5s": frac_tail(err, 5.0),
            "frac_err_gt_10s": frac_tail(err, 10.0),
        }
        return bundle[key]

    deltas = np.empty(n_boot, float)
    for b in range(n_boot):
        samp = rng.choice(events, size=len(events), replace=True)
        idx = np.concatenate([ev_to_idx[e] for e in samp])
        deltas[b] = one(pred_a, idx) - one(pred_b, idx)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"mean": float(deltas.mean()), "ci95": [float(lo), float(hi)], "n_boot": n_boot, "seed": seed}


def main() -> int:
    ensure_dir(OUT / "scalar_vs_c")
    ensure_dir(REPORT)

    split_frames = {
        "calibration": pd.read_parquet(artifacts_dir() / "results/pairwise_pilot/pairs_ranker_train.parquet").query(
            "split == 'calibration'"
        ).reset_index(drop=True),
        "heldout_eval": pd.read_parquet(artifacts_dir() / "results/pairwise_pilot/pairs_ranker_train.parquet").query(
            "split == 'heldout_eval'"
        ).reset_index(drop=True),
        "fulldev_phaseB": pd.read_parquet(artifacts_dir() / "results/pairwise_fulldev/pairs_phaseB.parquet"),
    }

    rows = []
    boot_rows = []
    for split, pairs in split_frames.items():
        pred_e = np.load(PRED / f"{split}__E_scalar.npy")
        pred_c = np.load(PRED / f"{split}__C_best.npy")
        # verify hashes
        for name, arr in [("E_scalar", pred_e), ("C_best", pred_c)]:
            h = hashlib.sha256(arr.astype(np.float64).tobytes()).hexdigest()
            expect = (PRED / f"{split}__{name}.sha256").read_text().strip()
            if h != expect:
                raise RuntimeError(f"hash mismatch {split} {name}")
        true = pairs["true_s_sample"].to_numpy(float)
        sr = pairs["sampling_rate_hz"].to_numpy(float)
        assert len(pred_e) == len(pairs) == len(pred_c)

        me = full_metrics(pred_e, true, sr, pairs)
        mc = full_metrics(pred_c, true, sr, pairs)
        for method, met in [("E_scalar", me), ("C_best", mc)]:
            rows.append({"split": split, "method": method, "role": "historical_dev" if split != "calibration" else "selection_split", **met})
        delta = {k: me[k] - mc[k] for k in [
            "f1@0.5", "f1@0.1", "detected_ae_median", "detected_ae_mae", "detected_ae_p95",
            "frac_err_gt_1s", "frac_err_gt_5s", "frac_err_gt_10s", "miss_rate",
            "fixes", "breaks", "net_fixes", "n_switch",
        ]}
        rows.append({"split": split, "method": "scalar_minus_C", "role": rows[-1]["role"], **{**me, **{f"delta_{k}": v for k, v in delta.items()}}})
        # overwrite with clean delta row
        rows[-1] = {
            "split": split,
            "method": "scalar_minus_C",
            "role": "historical_dev" if split != "calibration" else "selection_split_not_independent",
            **{f"delta_{k}": v for k, v in delta.items()},
            "n_eval": me["n_eval"],
            "practical_scale_f1": PRACTICAL,
            "delta_f1@0.5_vs_practical": float(delta["f1@0.5"] - PRACTICAL),
        }

        # exploratory bootstrap on frozen preds (subset of metrics; labeled exploratory)
        for key in ["f1@0.5", "f1@0.1", "detected_ae_p95", "detected_ae_mae", "frac_err_gt_1s", "frac_err_gt_5s"]:
            b = boot_metric(
                pairs["event_id"].to_numpy(),
                pred_e,
                pred_c,
                true,
                sr,
                key=key,
                n_boot=2000 if split != "fulldev_phaseB" else 1000,
                seed=42,
            )
            boot_rows.append({"split": split, "metric": key, "analysis": "exploratory_dev_on_frozen_preds", **b})

    df = pd.DataFrame(rows)
    # write tidy complete table: one row per split×method for E and C, plus delta rows
    wide_rows = []
    for split, pairs in split_frames.items():
        pred_e = np.load(PRED / f"{split}__E_scalar.npy")
        pred_c = np.load(PRED / f"{split}__C_best.npy")
        true = pairs["true_s_sample"].to_numpy(float)
        sr = pairs["sampling_rate_hz"].to_numpy(float)
        me = full_metrics(pred_e, true, sr, pairs)
        mc = full_metrics(pred_c, true, sr, pairs)
        wide_rows.append({"split": split, "method": "E_scalar", **me})
        wide_rows.append({"split": split, "method": "C_best", **mc})
        wide_rows.append({
            "split": split,
            "method": "scalar_minus_C",
            "f1@0.5": me["f1@0.5"] - mc["f1@0.5"],
            "f1@0.1": me["f1@0.1"] - mc["f1@0.1"],
            "detected_ae_median": me["detected_ae_median"] - mc["detected_ae_median"],
            "detected_ae_mae": me["detected_ae_mae"] - mc["detected_ae_mae"],
            "detected_ae_p95": me["detected_ae_p95"] - mc["detected_ae_p95"],
            "frac_err_gt_1s": me["frac_err_gt_1s"] - mc["frac_err_gt_1s"],
            "frac_err_gt_5s": me["frac_err_gt_5s"] - mc["frac_err_gt_5s"],
            "frac_err_gt_10s": me["frac_err_gt_10s"] - mc["frac_err_gt_10s"],
            "miss_rate": me["miss_rate"] - mc["miss_rate"],
            "n_eval": me["n_eval"],
            "n_pred": me["n_pred"],
            "coverage_pred": me["coverage_pred"],
            "fixes": me["fixes"] - mc["fixes"],
            "breaks": me["breaks"] - mc["breaks"],
            "net_fixes": me["net_fixes"] - mc["net_fixes"],
            "n_switch": me["n_switch"] - mc["n_switch"],
            "n_ge2": me["n_ge2"],
        })
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(OUT / "scalar_vs_c" / "scalar_vs_C_complete_metrics.csv", index=False)
    wide.to_csv(REPORT / "scalar_vs_C_complete_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "scalar_vs_c" / "scalar_vs_C_bootstrap_exploratory.csv", index=False)
    save_json({"rows": boot_rows, "note": "exploratory_dev_on_frozen_preds; not independent validation"}, OUT / "scalar_vs_c" / "scalar_vs_C_bootstrap_exploratory.json")

    # judgment snippet numbers
    fd = wide[(wide.split == "fulldev_phaseB") & (wide.method == "scalar_minus_C")].iloc[0]
    ho = wide[(wide.split == "heldout_eval") & (wide.method == "scalar_minus_C")].iloc[0]
    summary = {
        "practical_scale_abs_f1@0.5": PRACTICAL,
        "fulldev_delta_f1@0.5": float(fd["f1@0.5"]),
        "heldout_delta_f1@0.5": float(ho["f1@0.5"]),
        "fulldev_delta_p95": float(fd["detected_ae_p95"]),
        "fulldev_delta_mae": float(fd["detected_ae_mae"]),
        "fulldev_delta_frac_gt_1s": float(fd["frac_err_gt_1s"]),
        "fulldev_delta_frac_gt_5s": float(fd["frac_err_gt_5s"]),
        "interpretation_flags": {
            "only_tiny_f1_increment": abs(float(fd["f1@0.5"])) < PRACTICAL,
            "timing_tail_clearly_better_for_scalar": float(fd["detected_ae_p95"]) < -0.05 or float(fd["frac_err_gt_5s"]) < -0.002,
            "do_not_claim_equivalence_from_nonsignificance_alone": True,
            "do_not_claim_material_gain_from_ci_barely_above_zero": True,
        },
    }
    save_json(summary, OUT / "scalar_vs_c" / "comparison_summary.json")
    print(wide.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
