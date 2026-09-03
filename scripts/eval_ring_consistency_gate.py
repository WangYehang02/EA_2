#!/usr/bin/env python
"""Evaluate same-ring multi-station consistency gate on Stage-6 phaseB full-dev.

Does not touch locked confirm artifacts. Writes under artifacts/results/multistation/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks
from earthquake.multistation.ring_consistency import RingGateConfig, apply_ring_consistency_gate
from earthquake.utils import ensure_dir


def _load_phaseB():
    root = artifacts_dir() / "results" / "stage6"
    meta = pd.read_csv(root / "phaseB_eval_manifest.csv")
    pred_dir = root / "phaseC" / "baseline_preds"
    preds = {
        "STEAD_top1": np.load(pred_dir / "STEAD_top1.npy").astype(np.float64),
        "fixed_rescore_UNION": np.load(pred_dir / "fixed_rescore_UNION.npy").astype(np.float64),
    }
    for k, v in preds.items():
        if len(v) != len(meta):
            raise RuntimeError(f"{k} length {len(v)} != manifest {len(meta)}")
    return meta, preds


def _metrics(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    return {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.5": float(m["f1@0.5s"]),
        "precision@0.1": float(m["precision@0.1s"]),
        "recall@0.1": float(m["recall@0.1s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(m["miss_rate"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "n_pred": int(m["n_pred"]),
        "n_eval": int(m["n_eval"]),
    }


def main() -> None:
    meta, preds = _load_phaseB()
    true = meta["s_arrival_sample"].to_numpy(dtype=np.float64)
    sr = meta["sampling_rate_hz"].to_numpy(dtype=np.float64)

    configs: list[tuple[str, RingGateConfig]] = [
        (
            "user_missing10pct_w1s_ring5",
            RingGateConfig(
                ring_km=5.0,
                time_window_s=1.0,
                min_neighbors=1,
                rule="max_missing_frac",
                max_missing_frac=0.1,
                neighbor_mode="pred",
                no_neighbor_policy="keep",
            ),
        ),
        (
            "rec_min_support2_w1s_ring5",
            RingGateConfig(
                ring_km=5.0,
                time_window_s=1.0,
                min_neighbors=2,
                rule="min_support",
                min_support=2,
                neighbor_mode="pred",
                no_neighbor_policy="keep",
            ),
        ),
        (
            "rec_min_support2_w1p5s_ring5",
            RingGateConfig(
                ring_km=5.0,
                time_window_s=1.5,
                min_neighbors=2,
                rule="min_support",
                min_support=2,
                neighbor_mode="pred",
                no_neighbor_policy="keep",
            ),
        ),
        (
            "majority_w1s_ring5",
            RingGateConfig(
                ring_km=5.0,
                time_window_s=1.0,
                min_neighbors=2,
                rule="majority",
                neighbor_mode="pred",
                no_neighbor_policy="keep",
            ),
        ),
        (
            "oracle_catalog_min_support2_w1s_ring5",
            RingGateConfig(
                ring_km=5.0,
                time_window_s=1.0,
                min_neighbors=2,
                rule="min_support",
                min_support=2,
                neighbor_mode="catalog",
                no_neighbor_policy="keep",
            ),
        ),
        (
            "tight_ring2km_min_support2_w1s",
            RingGateConfig(
                ring_km=2.0,
                time_window_s=1.0,
                min_neighbors=2,
                rule="min_support",
                min_support=2,
                neighbor_mode="pred",
                no_neighbor_policy="keep",
            ),
        ),
    ]

    out_dir = ensure_dir(artifacts_dir() / "results" / "multistation")
    rows = []
    detail = {}

    for method, arr in preds.items():
        base = _metrics(arr, true, sr)
        rows.append({"method": method, "gate": "none", **base, "abstain_rate": 0.0, "n_gate_applied": 0})
        detail[f"{method}__none"] = {"metrics": base, "stats": {"abstain_rate": 0.0}}

        for gate_name, cfg in configs:
            out = apply_ring_consistency_gate(meta, arr, cfg=cfg)
            met = _metrics(out["gated_pred_samples"], true, sr)
            rows.append(
                {
                    "method": method,
                    "gate": gate_name,
                    **met,
                    "abstain_rate": out["stats"]["abstain_rate"],
                    "n_gate_applied": out["stats"]["n_gate_applied"],
                    "mean_neighbors_when_applied": out["stats"]["mean_neighbors_when_applied"],
                    "mean_support_when_applied": out["stats"]["mean_support_when_applied"],
                    "delta_f1@0.5": met["f1@0.5"] - base["f1@0.5"],
                    "delta_f1@0.1": met["f1@0.1"] - base["f1@0.1"],
                    "delta_p95": met["detected_ae_p95"] - base["detected_ae_p95"],
                    "delta_miss": met["miss_rate"] - base["miss_rate"],
                }
            )
            detail[f"{method}__{gate_name}"] = {
                "config": out["config"],
                "metrics": met,
                "stats": out["stats"],
                "delta_vs_ungated": {
                    "f1@0.5": met["f1@0.5"] - base["f1@0.5"],
                    "f1@0.1": met["f1@0.1"] - base["f1@0.1"],
                    "miss_rate": met["miss_rate"] - base["miss_rate"],
                    "detected_ae_p95": met["detected_ae_p95"] - base["detected_ae_p95"],
                    "precision@0.5": met["precision@0.5"] - base["precision@0.5"],
                    "recall@0.5": met["recall@0.5"] - base["recall@0.5"],
                },
            }
            # save gated preds for recommended setting on UNION
            if method == "fixed_rescore_UNION" and gate_name == "rec_min_support2_w1s_ring5":
                np.save(out_dir / "gated_fixed_rescore_UNION_rec_min_support2.npy", out["gated_pred_samples"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ring_gate_phaseB_metrics.csv", index=False)
    save_json(
        {
            "split": "stage6_phaseB_full_dev",
            "n_traces": int(len(meta)),
            "n_events": int(meta["event_id"].nunique()),
            "backup": "/mnt/yehang/Earthquake_backup_20260903_pre_ringgate",
            "note": "Experimental ring gate on existing preds; confirm locks untouched.",
            "results": detail,
        },
        out_dir / "ring_gate_phaseB_summary.json",
    )

    # Markdown report
    lines = [
        "# Same-ring multi-station gate — phaseB full-dev",
        "",
        f"- traces: **{len(meta)}**, events: **{meta['event_id'].nunique()}**",
        "- backup: `/mnt/yehang/Earthquake_backup_20260903_pre_ringgate`",
        "- confirm locks: **not modified**",
        "",
        "## Metrics",
        "",
        "| method | gate | F1@0.1 | F1@0.5 | P@0.5 | R@0.5 | miss | P95 | abstain | ΔF1@0.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['gate']} | {r['f1@0.1']:.4f} | {r['f1@0.5']:.4f} | "
            f"{r['precision@0.5']:.4f} | {r['recall@0.5']:.4f} | {r['miss_rate']:.4f} | "
            f"{r['detected_ae_p95']:.3f} | {r.get('abstain_rate', 0):.3f} | {r.get('delta_f1@0.5', 0):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `user_missing10pct_*`: original draft (abstain if >10% same-ring neighbors lack a near peak).",
            "- `rec_min_support2_*`: recommended (need ≥2 supporting neighbors when ≥2 exist).",
            "- `oracle_catalog_*`: neighbors use labeled S times (upper-bound / leakage diagnostic).",
            "",
        ]
    )
    (out_dir / "ring_gate_phaseB_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
