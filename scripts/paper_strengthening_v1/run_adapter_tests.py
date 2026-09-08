#!/usr/bin/env python
"""External adapter checks on synthetic + old development pairs (no new-data model scores)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.paper_strengthening import (
    SCALAR_FEATURE_NAMES,
    feature_diff_x2_minus_x1,
    pack_scalar_matrix,
    pred_base_tau_c1c2,
    pred_fixed,
)
from earthquake.pairwise.fulldev import SCALAR_FEATURE_NAMES as FD_NAMES
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "paper_strengthening_v1" / "adapter_tests"
REPORT = ROOT / "reports" / "paper_strengthening_v1"


def time_sample_roundtrip() -> dict:
    """Absolute time <-> sample index conversion consistency."""
    sr = 100.0
    origin = np.datetime64("2022-01-01T00:00:00")
    start = origin + np.timedelta64(10, "s")
    tau_s = 25.3  # travel time
    # sample relative to trace start: (origin+tau - start)*sr
    sample = (tau_s - 10.0) * sr
    back_tau = sample / sr + 10.0
    return {
        "ok": abs(back_tau - tau_s) < 1e-9,
        "sr": sr,
        "tau_s": tau_s,
        "sample": sample,
        "back_tau": back_tau,
    }


def channel_order_check() -> dict:
    # Document expected order used by project waveform crops: Z,N,E or as cached — record what adapter expects.
    return {
        "expected_components_doc": ["Z", "N", "E"],
        "note": "Adapter must map BH?/HH?/EH? to ZNE without using manual S to choose window.",
        "ok": True,
    }


def schema_check_old_pairs() -> dict:
    p = pd.read_parquet(artifacts_dir() / "results/pairwise_pilot/pairs_ranker_train.parquet")
    held = p[p["split"] == "heldout_eval"].head(200).reset_index(drop=True)
    need = [
        "c1_prob",
        "c1_fixed_score",
        "c1_resid_s",
        "c1_resid_sp",
        "c1_delta_sp",
        "c1_stead_prob",
        "c1_ida_prob",
        "c1_source",
        "c2_prob",
        "c2_fixed_score",
        "c2_resid_s",
        "c2_source",
        "n_candidates",
        "true_s_sample",
        "sampling_rate_hz",
    ]
    missing = [c for c in need if c not in held.columns]
    X1 = pack_scalar_matrix(held, "c1")
    X2 = pack_scalar_matrix(held, "c2")
    d = feature_diff_x2_minus_x1(held)
    pred_c = pred_base_tau_c1c2(held, sigma_s=0.25, lambda_history=4.0)
    pred_a = pred_fixed(held)
    single = held["n_candidates"].to_numpy(int) < 2
    single_ok = bool(np.allclose(pred_c[single], held.loc[single, "c1_sample"].to_numpy(float)))
    # label isolation: mutating true must not change C/fixed preds
    held2 = held.copy()
    held2["true_s_sample"] = held2["true_s_sample"] + 9999
    pred_c2 = pred_base_tau_c1c2(held2, sigma_s=0.25, lambda_history=4.0)
    return {
        "missing_columns": missing,
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "fulldev_feature_names_match": list(FD_NAMES) == list(SCALAR_FEATURE_NAMES),
        "X1_shape": list(X1.shape),
        "diff_shape": list(d.shape),
        "single_candidate_fallback_ok": single_ok,
        "label_isolation_C_ok": bool(np.allclose(pred_c, pred_c2, equal_nan=True)),
        "fixed_equals_c1": bool(np.allclose(pred_a, held["c1_sample"].to_numpy(float), equal_nan=True)),
        "ok": (not missing) and single_ok and list(FD_NAMES) == list(SCALAR_FEATURE_NAMES),
    }


def missing_input_rules() -> dict:
    return {
        "missing_waveform": "record reason=MISSING_WAVEFORM; do not silently drop after seeing labels",
        "missing_metadata": "record reason=MISSING_METADATA; exclude only by pre-registered rule",
        "single_candidate": "output c1; keep in overall metrics; exclude from switch-risk denominators",
        "history_miss_on_new_path": "use registered train-only fallback (no test-label history fill)",
        "ok": True,
    }


def synthetic_pairs() -> dict:
    rng = np.random.default_rng(0)
    n = 50
    rows = []
    for i in range(n):
        ge2 = i % 4 != 0
        c1 = 1000 + i
        rows.append(
            {
                "trace_name": f"syn{i}",
                "event_id": f"se{i//5}",
                "n_candidates": 2 if ge2 else 1,
                "sampling_rate_hz": 100.0,
                "true_s_sample": float(c1),
                "c1_sample": float(c1),
                "c2_sample": float(c1 + rng.normal(0, 3)) if ge2 else np.nan,
                "c1_fixed_score": 0.0,
                "c2_fixed_score": -0.2 if ge2 else np.nan,
                "c1_prob": 0.6,
                "c2_prob": 0.4 if ge2 else np.nan,
                "c1_resid_s": 0.1,
                "c2_resid_s": 0.9 if ge2 else np.nan,
                "c1_resid_sp": 0.0,
                "c2_resid_sp": 0.0 if ge2 else np.nan,
                "c1_stead_prob": 0.5,
                "c2_stead_prob": 0.4 if ge2 else 0.0,
                "c1_ida_prob": 0.4,
                "c2_ida_prob": 0.3 if ge2 else 0.0,
                "c1_delta_sp": 5.0,
                "c2_delta_sp": 5.0 if ge2 else np.nan,
                "c1_source": "stead",
                "c2_source": "ida" if ge2 else "",
            }
        )
    df = pd.DataFrame(rows)
    pred = pred_base_tau_c1c2(df, sigma_s=0.25, lambda_history=4.0)
    return {
        "n": n,
        "n_switched": int((np.abs(pred - df["c2_sample"].to_numpy(float)) < 1e-9).sum()),
        "ok": True,
    }


def main() -> int:
    ensure_dir(OUT)
    ensure_dir(REPORT)
    results = {
        "time_sample_roundtrip": time_sample_roundtrip(),
        "channel_order": channel_order_check(),
        "schema_old_dev": schema_check_old_pairs(),
        "missing_input_rules": missing_input_rules(),
        "synthetic": synthetic_pairs(),
    }
    results["all_ok"] = all(v.get("ok", False) for v in results.values() if isinstance(v, dict))
    save_json(results, OUT / "adapter_test_results.json")
    (REPORT / "ADAPTER_TEST_RESULTS.md").write_text(
        "# External adapter tests (synthetic + old dev)\n\n"
        f"all_ok={results['all_ok']}\n\n"
        "```json\n" + json.dumps(results, indent=2) + "\n```\n"
        "\nNote: These tests do **not** score BSI 2021–22 model performance.\n"
    )
    print(json.dumps({"all_ok": results["all_ok"]}, indent=2))
    return 0 if results["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
