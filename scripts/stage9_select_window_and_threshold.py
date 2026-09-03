#!/usr/bin/env python
"""Select SegPhase window A/B on 8k subset + threshold on full peaks if present, else on 8k.

Writes segphase_method_lock.json and COMPARATORS_LOCKED when ready.
Threshold grid only on Stage-6?dev (here: 8k window subset first; refreshed on full?dev peaks when available).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned
from earthquake.stage6.phaseB import sha256_file

THRESH = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
OFFICIAL = 0.10


def apply_thr(peaks: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = peaks.copy()
    prob = pd.to_numeric(out["s_peak_probability"], errors="coerce")
    pred = pd.to_numeric(out["pred_s_sample"], errors="coerce")
    pred2 = pred.where(prob >= float(thr), np.nan)
    out["pred_s_sample"] = pred2
    out["none_of_k"] = ~np.isfinite(pred2.to_numpy(float))
    return out


def best_threshold(meta: pd.DataFrame, peaks: pd.DataFrame) -> tuple[float, dict, pd.DataFrame]:
    rows = []
    best = None
    for thr in THRESH:
        m = metrics_from_aligned(keyed_align_predictions(meta, apply_thr(peaks, thr)))
        rows.append({"threshold": thr, **{k: m[k] for k in ["f1@0.5", "f1@0.1", "recall@0.5", "miss_rate", "detected_ae_p95", "prediction_coverage"]}})
        key = (
            m["f1@0.5"],
            m["f1@0.1"],
            m["recall@0.5"],
            -m["miss_rate"],
            -(m["detected_ae_p95"] if np.isfinite(m["detected_ae_p95"]) else 1e9),
            thr,
        )
        if best is None or key > best[0]:
            best = (key, thr, m)
    assert best is not None
    return float(best[1]), best[2], pd.DataFrame(rows)


def eval_scheme(meta: pd.DataFrame, peaks: pd.DataFrame, thr: float = OFFICIAL) -> dict:
    return metrics_from_aligned(keyed_align_predictions(meta, apply_thr(peaks, thr)))


def main() -> None:
    out = artifacts_dir() / "results" / "stage9"
    meta8 = pd.read_csv(out / "window_select_8k_manifest.csv")
    peaks_a = pd.read_parquet(out / "cache" / "winA_8k" / "segphase_A_peaks.parquet")
    peaks_b = pd.read_parquet(out / "cache" / "winB_8k" / "segphase_B_peaks.parquet")

    ma = eval_scheme(meta8, peaks_a, OFFICIAL)
    mb = eval_scheme(meta8, peaks_b, OFFICIAL)
    # selection rules
    if mb["f1@0.5"] > ma["f1@0.5"] + 0.002:
        scheme = "B"
        reason = "higher F1@0.5 by >0.002"
    elif ma["f1@0.5"] > mb["f1@0.5"] + 0.002:
        scheme = "A"
        reason = "higher F1@0.5 by >0.002"
    else:
        # F1@0.1 tie-break
        if mb["f1@0.1"] > ma["f1@0.1"]:
            scheme = "B"
            reason = "F1@0.5 within 0.002; higher F1@0.1"
        elif ma["f1@0.1"] > mb["f1@0.1"]:
            scheme = "A"
            reason = "F1@0.5 within 0.002; higher F1@0.1"
        else:
            scheme = "A"
            reason = "tie → cheaper non-overlap scheme A"

    peaks_win = peaks_a if scheme == "A" else peaks_b
    # Prefer full?dev peaks for threshold if available
    full_path = out / "cache" / f"dev_{scheme}" / f"segphase_{scheme}_peaks.parquet"
    if full_path.exists():
        meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
        peaks = pd.read_parquet(full_path)
        thr_source = "full_stage6_dev"
    else:
        meta = meta8
        peaks = peaks_win
        thr_source = "window_select_8k_dev_subset"

    thr, mdev, sweep = best_threshold(meta, peaks)
    sweep.to_csv(out / "segphase_dev_threshold_sweep.csv", index=False)

    # official threshold metrics on same peaks
    m_off = eval_scheme(meta, peaks, OFFICIAL)

    ckpt = Path("/home/yehang/EARTHQUAKE/baseline/SegPhase/model/model_100Hz.pth")
    lock = {
        "model": "SegPhase-100Hz",
        "selected_before_confirm": True,
        "post_confirm_external_comparator_evaluation": True,
        "repo_head": "27e7e5d9ce02fbc2aea5ec646a569a381a5e7b6e",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "channel_order": "UD_NS_EW==Z_N_E from INSTANCE ENZ",
        "normalize": "per-channel zscore",
        "peak": "scipy.find_peaks distance=100 height=threshold",
        "window_scheme": scheme,
        "window_selection": {
            "subset_n": len(meta8),
            "subset_manifest": "artifacts/results/stage9/window_select_8k_manifest.csv",
            "official_thr_for_window_compare": OFFICIAL,
            "metrics_A": {k: ma[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "prediction_coverage"]},
            "metrics_B": {k: mb[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "prediction_coverage"]},
            "reason": reason,
        },
        "threshold": thr,
        "official_threshold": OFFICIAL,
        "threshold_grid": THRESH,
        "threshold_source_cohort": thr_source,
        "dev_metrics_at_locked_thr": {k: mdev[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "detected_ae_p95", "prediction_coverage", "precision@0.5", "recall@0.5"]},
        "dev_metrics_at_official_thr": {k: m_off[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "detected_ae_p95", "prediction_coverage"]},
        "adapter_sha256": sha256_file(ROOT / "src/earthquake/stage9/segphase_adapter.py"),
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "sota_claim_allowed": False,
    }
    save_json(lock, out / "segphase_method_lock.json")

    # DKPN lock explicitly not for main
    save_json(
        {
            "model": "DKPN",
            "locked": False,
            "role": "diagnostic_only_possible_leakage",
            "enter_main_table": False,
            "reason": "INSTANCE-trained official weights; confirm leakage not excludable",
        },
        out / "dkpn_method_lock.json",
    )

    # COMPARATORS_LOCKED only when threshold from full?dev OR we allow provisional then refresh
    if thr_source == "full_stage6_dev":
        (out / "COMPARATORS_LOCKED").write_text(
            json.dumps({"SegPhase": True, "DKPN": False, "utc": datetime.now(timezone.utc).isoformat()}) + "\n"
        )
    else:
        (out / "COMPARATORS_LOCKED").write_text(
            json.dumps(
                {
                    "SegPhase_provisional_window_locked": True,
                    "SegPhase_threshold_provisional_on_8k": True,
                    "refresh_threshold_after_full_dev_cache": True,
                    "window_scheme": scheme,
                    "utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )

    report = f"""# Stage 9 — Dev threshold / window selection

## Window scheme (8k sha256-sorted Stage-6?dev subset)

| Scheme | F1@0.5 | F1@0.1 | miss | coverage |
|--|--:|--:|--:|--:|
| A non-overlap | {ma['f1@0.5']:.4f} | {ma['f1@0.1']:.4f} | {ma['miss_rate']:.4f} | {ma['prediction_coverage']:.4f} |
| B 50% overlap | {mb['f1@0.5']:.4f} | {mb['f1@0.1']:.4f} | {mb['miss_rate']:.4f} | {mb['prediction_coverage']:.4f} |

**Selected:** `{scheme}` — {reason}

## Threshold (grid on `{thr_source}`)

- Official height: `{OFFICIAL}`
- **Locked (dev):** `{thr}`
- Dev F1@0.5 @ locked: `{mdev['f1@0.5']:.4f}`
- Dev F1@0.5 @ official: `{m_off['f1@0.5']:.4f}`

Confirm must not re-select.
"""
    (ROOT / "reports/stage9/dev_threshold_selection.md").write_text(report)
    print(json.dumps({"scheme": scheme, "threshold": thr, "thr_source": thr_source, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
