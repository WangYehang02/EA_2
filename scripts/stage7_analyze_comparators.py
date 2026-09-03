#!/usr/bin/env python
"""Stage 7A analysis: threshold locks, confirm metrics, bootstrap, final verdict.

Assumes peak caches exist for ethz/scedc on dev+confirm. Does not modify Stage 6.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics
from earthquake.utils import ensure_dir

THRESH_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
BOOT_N = 5000
BOOT_SEED = 20260817


def apply_threshold(peaks: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = peaks.copy()
    prob = pd.to_numeric(out["s_peak_probability"], errors="coerce")
    pred = pd.to_numeric(out["pred_s_sample"], errors="coerce")
    mask = prob >= float(thr)
    pred2 = pred.where(mask, np.nan)
    out["pred_s_sample"] = pred2
    out["none_of_k"] = ~np.isfinite(pred2.to_numpy(float))
    return out


def select_threshold(dev_meta: pd.DataFrame, peaks: pd.DataFrame) -> tuple[float, dict, pd.DataFrame]:
    rows = []
    best = None
    for thr in THRESH_GRID:
        pred_df = apply_threshold(peaks, thr)
        aligned = keyed_align_predictions(dev_meta, pred_df)
        m = metrics_from_aligned(aligned)
        row = {"threshold": thr, **{k: m[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "detected_ae_p95", "precision@0.5", "recall@0.5", "prediction_coverage"]}}
        rows.append(row)
        key = (m["f1@0.5"], m["f1@0.1"], -m["miss_rate"], -m["detected_ae_p95"] if np.isfinite(m["detected_ae_p95"]) else -1e9, thr)
        if best is None or key > best[0]:
            best = (key, thr, m)
    assert best is not None
    return float(best[1]), best[2], pd.DataFrame(rows)


def event_bootstrap(pred_a, pred_b, true, sr, event_ids, n_boot=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids.astype(str))
    idx_by = {}
    for i, e in enumerate(event_ids.astype(str)):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(idx_by[e], int) for e in events]
    keys = ["f1@0.5", "f1@0.1", "precision@0.5", "recall@0.5", "miss_rate", "wrong_peak_rate", "detected_ae_p95"]

    def pack(pred, ix=None):
        if ix is None:
            return comprehensive_pick_metrics(pred, true, sr)
        return comprehensive_pick_metrics(pred[ix], true[ix], sr[ix])

    ba, bb = pack(pred_a), pack(pred_b)
    deltas = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma, mb = pack(pred_a, ix), pack(pred_b, ix)
        for k in keys:
            deltas[k][b] = float(ma[k] - mb[k])
    out = {"n_boot": n_boot, "seed": seed, "definition": "delta = fixed_UNION - comparator", "point_delta": {k: float(ba[k] - bb[k]) for k in keys}}
    for k in keys:
        arr = deltas[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {"mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)]}
    return out


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage7")
    locks = ensure_dir(out / "locks")
    # verify stage6 lock unchanged
    lock6 = artifacts_dir() / "results" / "stage6" / "final_confirm" / "method_lock.json"
    h6 = (artifacts_dir() / "results" / "stage6" / "final_confirm" / "method_lock.sha256").read_text().strip()
    assert sha256_file(lock6) == h6
    assert (artifacts_dir() / "results" / "stage6" / "final_confirm" / "CONFIRM.CONSUMED").exists()

    dev_meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    conf_meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_s_eval_manifest.csv")
    cohort = {
        "dev_n_events": int(dev_meta.event_id.nunique()),
        "dev_n_traces": int(len(dev_meta)),
        "confirm_n_events": int(conf_meta.event_id.nunique()),
        "confirm_n_traces": int(len(conf_meta)),
        "dev_manifest_sha256": sha256_file(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv"),
        "confirm_manifest_sha256": sha256_file(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_s_eval_manifest.csv"),
        "stage6_method_lock_sha256": h6,
        "confirm_consumed": True,
        "post_confirm_external_comparator_evaluation": True,
    }
    save_json(cohort, out / "cohort_hashes.json")

    # frozen Stage6 confirm preds
    frozen = pd.read_parquet(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_predictions.parquet")
    true = conf_meta.s_arrival_sample.to_numpy(float)
    sr = conf_meta.sampling_rate_hz.to_numpy(float)
    events = conf_meta.event_id.astype(str).to_numpy()
    names = conf_meta.trace_name.astype(str).to_numpy()
    fixed_union = frozen.set_index("trace_name").reindex(names)["fixed_rescore_UNION"].to_numpy(float)
    stead = frozen.set_index("trace_name").reindex(names)["STEAD_top1"].to_numpy(float)

    sweep_rows = []
    confirm_rows = []
    boot = {}
    selected = {}

    for weight in ["ethz", "scedc"]:
        dev_peaks = pd.read_parquet(out / "cache" / f"dev_{weight}" / f"phasenet_{weight}_peaks.parquet")
        conf_peaks = pd.read_parquet(out / "cache" / f"confirm_{weight}" / f"phasenet_{weight}_peaks.parquet")
        thr, dev_m, sweep = select_threshold(dev_meta, dev_peaks)
        sweep["model"] = f"PhaseNet-{weight.upper()}"
        sweep_rows.append(sweep)
        selected[weight] = thr

        lock = {
            "model": f"PhaseNet-{weight.upper()}",
            "weight": weight,
            "weight_sha256": load_json(out / "comparator_registry.json")["models"][f"PhaseNet-{weight.upper()}"]["weight_sha256"],
            "threshold": thr,
            "selected_before_confirm": True,
            "threshold_grid": THRESH_GRID,
            "selection_metric": "S F1@0.5",
            "dev_metrics": {k: dev_m[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "detected_ae_p95", "prediction_coverage"]},
            "preprocessing": "SeisBench annotate + UTC remap ENZ->ZNE PSN",
            "evaluation_code_sha256": sha256_file(ROOT / "src/earthquake/stage6/keyed_align.py"),
            "locked_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(lock, locks / f"PhaseNet-{weight.upper()}_comparator_lock.json")

        pred_df = apply_threshold(conf_peaks, thr)
        aligned = keyed_align_predictions(conf_meta, pred_df)
        cm = metrics_from_aligned(aligned)
        pred = aligned.pred_s_sample.to_numpy(float)
        confirm_rows.append({"model": f"PhaseNet-{weight.upper()}", "threshold": thr, **cm})
        # save preds
        aligned.to_parquet(out / f"confirm_preds_PhaseNet-{weight.upper()}.parquet", index=False)
        boot[f"fixed_UNION_vs_PhaseNet-{weight.upper()}"] = event_bootstrap(fixed_union, pred, true, sr, events)
        print({"locked": weight, "thr": thr, "dev_f1": dev_m["f1@0.5"], "confirm_f1": cm["f1@0.5"]}, flush=True)

    # also report frozen methods on confirm (from Stage6 metrics file)
    s6 = load_json(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_method_metrics.json")
    keep = [
        "f1@0.1", "f1@0.5", "precision@0.1", "precision@0.5", "recall@0.1", "recall@0.5",
        "miss_rate", "wrong_peak_rate", "prediction_coverage", "detected_ae_p95", "detected_ae_median",
        "detected_ae_mae", "detected_ae_p90", "detected_ae_p99", "matched_count_0.5",
        "detected_ae_p95_denominator", "n_traces", "n_finite_prediction", "tp@0.1", "fp@0.1", "fn@0.1",
        "tp@0.5", "fp@0.5", "fn@0.5",
    ]
    for name in ["STEAD_top1", "fixed_rescore_STEAD", "fixed_rescore_UNION", "oracle_UNION"]:
        m = s6[name]
        confirm_rows.append({"model": name, "threshold": None, **{k: m.get(k) for k in keep}})

    sweep_all = pd.concat(sweep_rows, ignore_index=True)
    sweep_all.to_csv(out / "dev_threshold_sweep.csv", index=False)
    pd.DataFrame(confirm_rows).to_csv(out / "comparator_metrics_confirm.csv", index=False)
    # add frozen bootstrap vs STEAD from stage6
    s6b = load_json(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_bootstrap.json")
    boot["fixed_UNION_vs_STEAD_top1_frozen"] = s6b.get("fixed_UNION_vs_STEAD_top1")
    save_json(boot, out / "comparator_bootstrap.json")

    verdict = {
        "stage": "7A",
        "sota_claim_allowed": False,
        "post_confirm_external_comparator_evaluation": True,
        "stage6_main_method_unchanged": "fixed_rescore_UNION",
        "stage6_method_lock_sha256": h6,
        "new_comparators": {
            "PhaseNet-ETHZ": {"threshold": selected["ethz"]},
            "PhaseNet-SCEDC": {"threshold": selected["scedc"]},
        },
        "excluded": ["EQTransformer-* (download failed)", "LFTNet (not reproducible)", "instance weights (leakage)"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # summarize whether fixed UNION beats comparators on F1@0.5
    for w in ["ethz", "scedc"]:
        key = f"fixed_UNION_vs_PhaseNet-{w.upper()}"
        d = boot[key]["f1@0.5"]
        verdict[key] = {"mean_delta": d["mean_delta"], "ci95": d["ci95"], "fixed_better_stable": d["ci95"][0] > 0}
    save_json(verdict, out / "stage7A_final_verdict.json")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
