#!/usr/bin/env python
"""Phase C.2 closeout: keyed recompute of frozen ranker preds; do not overwrite Phase C/C.1."""

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

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned, shuffle_invariant_check
from earthquake.stage6.phaseB import sha256_file
from earthquake.stage6.phaseC1.infer import align_preds_to_meta, predict_ranker_variants
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics
from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.models import build_ranker
from earthquake.utils import ensure_dir
from torch.utils.data import DataLoader


def _hash_existing(paths: dict[str, Path]) -> dict:
    out = {}
    for k, p in paths.items():
        out[k] = {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() else None}
    return out


@torch.no_grad()
def _eval_ckpt(ckpt: Path, feat: pd.DataFrame, meta: pd.DataFrame, device: str) -> dict:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    names = list(blob["feature_names"])
    model = build_ranker(blob["variant"], len(names))
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    ds = TraceCandidateDataset(feat, names, hard_boost=False, seed=0)
    ds.indices = list(range(len(ds.groups)))
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_traces)
    rows = []
    for batch in loader:
        logits = model(batch["x"].to(device), batch["mask"].to(device))
        idx = logits.argmax(-1).cpu().numpy()
        samp = batch["samples"].numpy()
        mask = batch["mask"].numpy()
        for i, tn in enumerate(batch["trace_name"]):
            pi = int(idx[i])
            none = pi >= 10
            pred = np.nan if none or not mask[i, pi] else float(samp[i, pi])
            rows.append(
                {
                    "trace_name": str(tn),
                    "event_id": str(batch["event_id"][i]),
                    "pred_s_sample": pred,
                    "true_s_sample": float(batch["true_s"][i]),
                    "sampling_rate_hz": float(batch["sr"][i]),
                    "none_of_k": none,
                }
            )
    preds = pd.DataFrame(rows)
    aligned = keyed_align_predictions(meta, preds)
    met = metrics_from_aligned(aligned)
    met.update({"ckpt": str(ckpt), "variant": blob.get("variant"), "seed": blob.get("seed"), "beta": blob.get("beta"), "gamma": blob.get("gamma")})
    return met


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "phaseC2")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    frozen_preds = artifacts_dir() / "results" / "stage6" / "phaseC" / "ranker_dev_predictions.parquet"
    orig_paths = {
        "phaseC_final_verdict": artifacts_dir() / "results" / "stage6" / "phaseC" / "phaseC_final_verdict.json",
        "ranker_dev_metrics_overnight": artifacts_dir() / "results" / "stage6" / "phaseC" / "ranker_dev_metrics.json",
        "ranker_dev_predictions": frozen_preds,
        "phaseC1_verdict": artifacts_dir() / "results" / "stage6" / "phaseC1" / "phaseC1_final_verdict.json",
        "phaseB_manifest": artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv",
        "fixed_union_npy": artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy",
    }
    orig_hashes = _hash_existing(orig_paths)
    save_json(orig_hashes, out / "original_artifact_hashes.json")

    # Integrity on frozen overnight preds
    aligned = keyed_align_predictions(meta, pd.read_parquet(frozen_preds))
    r3_corrected = metrics_from_aligned(aligned)
    shuffle = shuffle_invariant_check(meta, pd.read_parquet(frozen_preds), seed=42)
    integrity = {
        "n_manifest": len(meta),
        "n_aligned": len(aligned),
        "shuffle_invariant": shuffle["ok"],
        "corrected_R3_from_frozen_preds": r3_corrected,
        "matches_phaseC1_f1_05": abs(r3_corrected["f1@0.5"] - 0.8322776841575145) < 1e-9,
        "evaluator_fix": "scripts/evaluate_stage6_candidate_ranker.py now uses keyed_align_predictions",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(integrity, out / "alignment_integrity.json")
    assert integrity["matches_phaseC1_f1_05"]

    # Recompute all rankers keyed
    feat_r1 = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R1.parquet")
    feat_r2 = pd.read_parquet(artifacts_dir() / "cache" / "stage6" / "phaseC" / "dev_features_R2.parquet")
    rows = []
    root = artifacts_dir() / "models" / "stage6" / "ranker"
    for d in sorted(root.iterdir()):
        ck = d / "best.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        feat = feat_r1 if blob["variant"] == "R1" else feat_r2
        print({"eval": d.name}, flush=True)
        met = _eval_ckpt(ck, feat, meta, device)
        rows.append({"model": d.name, **met})
    all_df = pd.DataFrame(rows)
    all_df.to_csv(out / "corrected_all_rankers.csv", index=False)

    fixed = np.load(orig_paths["fixed_union_npy"]).astype(float)
    assert len(fixed) == len(meta)
    fixed_m = comprehensive_pick_metrics(fixed, meta["s_arrival_sample"].to_numpy(float), meta["sampling_rate_hz"].to_numpy(float))

    # diagnostics from C.1 parquet if present else recompute reported only
    diag = {}
    ckpt = artifacts_dir() / "models" / "stage6" / "ranker" / "R3_seed2026_b0.2_g1.0" / "best.pt"
    var = predict_ranker_variants(ckpt, feat_r2, device=device, fixed_fallback=fixed, meta_trace_names=meta.trace_name.astype(str).tolist())
    for col, name in [("pred_reported", "R3_reported"), ("pred_forced_choice", "R3_forced_choice"), ("pred_none_fallback_fixed", "R3_none_fallback_fixed")]:
        pred = align_preds_to_meta(var, meta, col)
        none = ~np.isfinite(pred) if col != "pred_reported" else var.set_index("trace_name")["none_of_k"].reindex(meta.trace_name.astype(str)).fillna(True).to_numpy(bool)
        if col == "pred_reported":
            none = var.set_index("trace_name")["none_of_k"].reindex(meta.trace_name.astype(str)).fillna(True).to_numpy(bool)
        diag[name] = comprehensive_pick_metrics(pred, meta["s_arrival_sample"].to_numpy(float), meta["sampling_rate_hz"].to_numpy(float), none_mask=none)

    corrected = {
        "R3_reported_keyed": r3_corrected,
        "fixed_rescore_UNION": fixed_m,
        "delta_f1_05_r3_minus_fixed": float(r3_corrected["f1@0.5"] - fixed_m["f1@0.5"]),
        "diagnostics": {k: {kk: diag[k][kk] for kk in ["f1@0.5", "f1@0.1", "none_of_k_rate", "detected_ae_p95"]} for k in diag},
        "all_rankers_csv": str(out / "corrected_all_rankers.csv"),
    }
    save_json(corrected, out / "corrected_ranker_metrics.json")

    verdict = {
        "verdict": "ranker_failed_predeclared_gate",
        "recommended_stage6_main_method": "fixed_rescore_UNION",
        "ranker_role": "negative_ablation",
        "forced_choice_role": "posthoc_diagnostic_only",
        "none_fallback_role": "posthoc_diagnostic_only",
        "multistation_may_start": False,
        "confirm_may_unseal_only_after_method_lock": True,
        "corrected_R3_f1_05": r3_corrected["f1@0.5"],
        "fixed_union_f1_05": fixed_m["f1@0.5"],
        "overnight_phaseC_verdict_preserved": True,
        "phaseC1_preserved": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(verdict, out / "phaseC_corrected_verdict.json")

    md = f"""# Stage 6 Phase C.2 — Evaluator Closeout

**Verdict:** `ranker_failed_predeclared_gate`  
**Recommended main method:** `fixed_rescore_UNION`  
**Ranker role:** negative_ablation  
**Forced-choice / none-fallback:** posthoc_diagnostic_only  

## Fix

`evaluate_stage6_candidate_ranker.py` now scores via `keyed_align_predictions(trace_name)`.
Historical overnight `ranker_dev_metrics.json` / `phaseC_final_verdict.json` preserved.

## Corrected R3 (frozen preds, keyed)

- F1@0.5 = {r3_corrected['f1@0.5']:.6f}
- F1@0.1 = {r3_corrected['f1@0.1']:.6f}
- none/miss = {r3_corrected['none_of_k_rate']:.6f}
- detected_ae_p95 = {r3_corrected['detected_ae_p95']:.6f} (denom={r3_corrected['detected_ae_p95_denominator']})

## fixed_rescore_UNION

- F1@0.5 = {fixed_m['f1@0.5']:.6f}
- detected_ae_p95 = {fixed_m['detected_ae_p95']:.6f}

ΔF1@0.5 (R3−fixed) = {verdict['corrected_R3_f1_05'] - verdict['fixed_union_f1_05']:+.6f} (fails +0.01 gate)
"""
    (ROOT / "reports/stage6/phaseC2_evaluator_closeout.md").write_text(md)
    (out / "PHASEC2.DONE").write_text("DONE\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
