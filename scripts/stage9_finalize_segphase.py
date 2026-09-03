#!/usr/bin/env python
"""Finalize Stage-9 SegPhase confirm metrics + bootstrap + complementarity + reports.

DKPN remains diagnostic-only (no confirm main metrics).
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

BOOT_N = 5000
BOOT_SEED = 20260818
THRESH = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]


def apply_thr(peaks: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = peaks.copy()
    prob = pd.to_numeric(out["s_peak_probability"], errors="coerce")
    pred = pd.to_numeric(out["pred_s_sample"], errors="coerce")
    pred2 = pred.where(prob >= float(thr), np.nan)
    out["pred_s_sample"] = pred2
    out["none_of_k"] = ~np.isfinite(pred2.to_numpy(float))
    return out


def best_threshold(meta, peaks):
    best = None
    rows = []
    for thr in THRESH:
        m = metrics_from_aligned(keyed_align_predictions(meta, apply_thr(peaks, thr)))
        rows.append({"threshold": thr, **{k: m[k] for k in ["f1@0.5", "f1@0.1", "recall@0.5", "miss_rate", "detected_ae_p95"]}})
        key = (m["f1@0.5"], m["f1@0.1"], m["recall@0.5"], -m["miss_rate"], thr)
        if best is None or key > best[0]:
            best = (key, thr, m)
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
    out = {"n_boot": n_boot, "seed": seed, "definition": "delta = pred_a - pred_b", "point_delta": {k: float(ba[k] - bb[k]) for k in keys}}
    for k in keys:
        arr = deltas[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {"mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)]}
    return out


def oracle_from_candidates(meta: pd.DataFrame, cand_frames: list[pd.DataFrame], tol_s: float = 0.5) -> dict:
    """Pick among candidates the one closest to true within any source; else miss."""
    # Build per-trace list of (sample, prob, source)
    by = {str(t): [] for t in meta.trace_name.astype(str)}
    for src_name, df in cand_frames:
        for _, r in df.iterrows():
            tn = str(r.trace_name)
            if tn not in by:
                continue
            # top1 from peaks file
            if np.isfinite(r.pred_s_sample):
                by[tn].append((float(r.pred_s_sample), float(r.s_peak_probability) if np.isfinite(r.s_peak_probability) else 0.0, src_name))
            # also parse cand_json if present
            if "cand_json" in df.columns and isinstance(r.cand_json, str) and r.cand_json.startswith("["):
                try:
                    for c in json.loads(r.cand_json):
                        by[tn].append((float(c["sample"]), float(c["prob"]), src_name))
                except Exception:  # noqa: BLE001
                    pass
    preds = []
    true = meta.s_arrival_sample.to_numpy(float)
    sr = meta.sampling_rate_hz.to_numpy(float)
    names = meta.trace_name.astype(str).to_numpy()
    for i, tn in enumerate(names):
        cands = by.get(tn, [])
        if not cands or not np.isfinite(true[i]):
            preds.append(np.nan)
            continue
        # choose candidate minimizing |err| 
        best = None
        for samp, prob, src in cands:
            err = abs(samp - true[i]) / sr[i]
            if best is None or err < best[0]:
                best = (err, samp)
        preds.append(best[1] if best is not None else np.nan)
    pred = np.asarray(preds, float)
    return comprehensive_pick_metrics(pred, true, sr)


def main() -> None:
    out = artifacts_dir() / "results" / "stage9"
    reports = ROOT / "reports" / "stage9"
    lock = load_json(out / "segphase_method_lock.json")
    scheme = lock["window_scheme"]

    # refresh threshold on full?dev
    meta_dev = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    peaks_dev = pd.read_parquet(out / "cache" / f"dev_{scheme}" / f"segphase_{scheme}_peaks.parquet")
    thr, mdev, sweep = best_threshold(meta_dev, peaks_dev)
    sweep.to_csv(out / "segphase_dev_threshold_sweep.csv", index=False)
    lock["threshold"] = thr
    lock["threshold_source_cohort"] = "full_stage6_dev"
    lock["dev_metrics_at_locked_thr"] = {k: mdev[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "detected_ae_p95", "prediction_coverage", "precision@0.5", "recall@0.5"]}
    lock["selected_before_confirm"] = True
    lock["refreshed_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(lock, out / "segphase_method_lock.json")
    save_json({"SegPhase": True, "DKPN": False, "utc": lock["refreshed_utc"]}, out / "COMPARATORS_LOCKED")

    # confirm once
    meta_c = pd.read_csv(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_s_eval_manifest.csv")
    peaks_c = pd.read_parquet(out / "cache" / f"confirm_{scheme}" / f"segphase_{scheme}_peaks.parquet")
    aligned = keyed_align_predictions(meta_c, apply_thr(peaks_c, thr))
    m_seg = metrics_from_aligned(aligned)
    save_json({"SegPhase-100Hz": m_seg, "threshold": thr, "window_scheme": scheme}, out / "segphase_confirm_metrics.json")

    # reuse stage6/7 metrics
    s6 = load_json(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_method_metrics.json")
    s7 = pd.read_csv(artifacts_dir() / "results" / "stage7" / "comparator_metrics_confirm.csv")
    confirm_rows = [{"model": "SegPhase-100Hz", "threshold": thr, "window_scheme": scheme, **m_seg}]
    for _, r in s7.iterrows():
        confirm_rows.append(r.to_dict())
    pd.DataFrame(confirm_rows).to_json(out / "comparator_metrics_confirm.json", orient="records", indent=2)

    # bootstrap Ours - SegPhase and SegPhase - STEAD
    frozen = pd.read_parquet(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_predictions.parquet")
    names = meta_c.trace_name.astype(str).to_numpy()
    true = meta_c.s_arrival_sample.to_numpy(float)
    sr = meta_c.sampling_rate_hz.to_numpy(float)
    events = meta_c.event_id.astype(str).to_numpy()
    fixed = frozen.set_index("trace_name").reindex(names)["fixed_rescore_UNION"].to_numpy(float)
    stead = frozen.set_index("trace_name").reindex(names)["STEAD_top1"].to_numpy(float)
    seg = aligned.pred_s_sample.to_numpy(float)
    boot = {
        "Ours_minus_SegPhase": event_bootstrap(fixed, seg, true, sr, events),
        "SegPhase_minus_STEAD": event_bootstrap(seg, stead, true, sr, events),
        "Ours_minus_STEAD_frozen": load_json(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_bootstrap.json").get(
            "fixed_UNION_vs_STEAD_top1"
        ),
        "DKPN": {"status": "skipped_diagnostic_only_INSTANCE_leakage"},
    }
    # fix definitions explicitly
    boot["Ours_minus_SegPhase"]["definition"] = "delta = fixed_rescore_UNION - SegPhase"
    boot["SegPhase_minus_STEAD"]["definition"] = "delta = SegPhase - STEAD_top1"
    save_json(boot, out / "comparator_bootstrap.json")

    # complementarity on?dev (UNION oracle vs +SegPhase)
    # Need UNION candidates — reuse stage6 union parquet if present
    union_path = artifacts_dir() / "results" / "stage6" / "phaseB_union_candidates.parquet"
    # fallback: use confirm_union style on?dev if exists
    cand_oracle = {"status": "partial"}
    # Use top1 SegPhase peaks as 1-cand oracle vs STEAD top1 as proxy if full UNION cand missing
    # Prefer existing phaseB fixed union oracle metrics from stage6 for current UNION oracle
    # Compute SegPhase top5 oracle on?dev
    # Dedup tolerance 0.05s locked
    dedup_tol = 0.05
    # Build SegPhase top5 oracle
    pred_s5 = []
    for _, r in peaks_dev.iterrows():
        cands = []
        if isinstance(r.get("cand_json"), str) and r.cand_json.startswith("["):
            try:
                cands = json.loads(r.cand_json)
            except Exception:  # noqa: BLE001
                cands = []
        true_s = float(r.true_s_sample) if np.isfinite(r.true_s_sample) else np.nan
        sr_i = float(r.sampling_rate_hz)
        best = np.nan
        if cands and np.isfinite(true_s):
            # time-dedup roughly by sorting
            best = min(cands, key=lambda c: abs(float(c["sample"]) - true_s))["sample"]
        elif np.isfinite(r.pred_s_sample):
            best = float(r.pred_s_sample)
        pred_s5.append(best)
    seg5 = np.asarray(pred_s5, float)
    true_d = meta_dev.s_arrival_sample.to_numpy(float)
    sr_d = meta_dev.sampling_rate_hz.to_numpy(float)
    # align peaks_dev to meta_dev
    aligned_dev = keyed_align_predictions(meta_dev, peaks_dev[["trace_name", "event_id", "sampling_rate_hz", "pred_s_sample", "true_s_sample"]].assign(none_of_k=False))
    # replace pred with oracle seg5 in meta order
    peaks_ord = peaks_dev.set_index("trace_name").reindex(meta_dev.trace_name.astype(str))
    seg5 = []
    for tn, true_s, sr_i in zip(meta_dev.trace_name.astype(str), true_d, sr_d):
        row = peaks_ord.loc[tn]
        cands = []
        cj = row.cand_json if "cand_json" in peaks_ord.columns else "[]"
        if isinstance(cj, str) and cj.startswith("["):
            try:
                cands = json.loads(cj)
            except Exception:  # noqa: BLE001
                cands = []
        if cands and np.isfinite(true_s):
            # dedup by 0.05s then pick closest
            kept = []
            for c in sorted(cands, key=lambda x: -float(x["prob"])):
                samp = float(c["sample"])
                if any(abs(samp - k) / sr_i <= dedup_tol for k in kept):
                    continue
                kept.append(samp)
            seg5.append(min(kept, key=lambda s: abs(s - true_s)) if kept else np.nan)
        else:
            seg5.append(float(row.pred_s_sample) if np.isfinite(row.pred_s_sample) else np.nan)
    seg5 = np.asarray(seg5, float)
    m_seg5 = comprehensive_pick_metrics(seg5, true_d, sr_d)
    # Current UNION oracle from stage6 confirm is not?dev — load?dev fixed union if available
    # Use stage6 phaseB metrics file if present
    union_oracle_dev = None
    for cand in [
        artifacts_dir() / "results" / "stage6" / "phaseB_oracle_metrics.json",
        artifacts_dir() / "results" / "stage6" / "phaseC2" / "fixed_union_dev_metrics.json",
    ]:
        if cand.exists():
            union_oracle_dev = load_json(cand)
            break
    # Compute delta using locked?dev SegPhase top1 vs?need UNION oracle on same?dev
    # Fallback: use fixed_rescore_UNION on?dev if predictions exist
    fixed_dev_path = artifacts_dir() / "results" / "stage6" / "phaseB_fixed_rescore_UNION_preds.parquet"
    complementarity = {
        "dedup_tol_s": dedup_tol,
        "segphase_top5_oracle_dev": {k: m_seg5[k] for k in ["f1@0.5", "f1@0.1", "miss_rate", "prediction_coverage"]},
        "gate_delta_required": 0.01,
        "note": "Full UNION+SegPhase oracle requires Stage-6 UNION candidate cache on?dev; if unavailable, gate not claimed passed.",
    }
    # Try phaseB union parquet
    for up in [
        artifacts_dir() / "results" / "stage6" / "phaseB" / "union_candidates.parquet",
        artifacts_dir() / "results" / "stage6" / "union_candidates.parquet",
        artifacts_dir() / "results" / "stage6" / "phaseB_eval_union_candidates.parquet",
    ]:
        if up.exists():
            complementarity["union_candidates_path"] = str(up)
            break
    if "union_candidates_path" not in complementarity:
        complementarity["verdict_tag"] = "candidate_complementarity_insufficient"
        complementarity["reason"] = "UNION candidate cache on?dev not located; cannot verify Δoracle F1@0.5 ≥ +0.01"
        complementarity["add_segphase_to_union"] = False
    save_json(complementarity, out / "candidate_oracle.json")

    # cohort hashes
    cohort = {
        "dev_manifest_sha256": sha256_file(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv"),
        "confirm_manifest_sha256": sha256_file(artifacts_dir() / "results" / "stage6" / "final_confirm" / "confirm_s_eval_manifest.csv"),
        "dev_n": len(meta_dev),
        "confirm_n": len(meta_c),
        "stage6_method_lock_sha256": "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303",
        "post_confirm_external_comparator_evaluation": True,
    }
    save_json(cohort, out / "cohort_hashes.json")
    save_json({"SegPhase": mdev}, out / "segphase_dev_metrics.json")
    save_json({"status": "diagnostic_only", "enter_main_table": False}, out / "dkpn_dev_metrics.json")

    # verdict
    d = boot["Ours_minus_SegPhase"]["f1@0.5"]
    ci = d["ci95"]
    if ci[0] > 0:
        tag = "segphase_below_ours_same_protocol"
    elif ci[1] < 0:
        tag = "segphase_above_ours_same_protocol"
    else:
        tag = "statistically_indistinguishable"

    verdict = {
        "stage": "9",
        "final_verdict": tag,
        "sota_claim_allowed": False,
        "replace_stage6_main_method": False,
        "post_confirm_external_comparator_evaluation": True,
        "SegPhase": {
            "window_scheme": scheme,
            "threshold": thr,
            "confirm_f1@0.5": m_seg["f1@0.5"],
            "confirm_f1@0.1": m_seg["f1@0.1"],
            "confirm_miss": m_seg["miss_rate"],
            "confirm_coverage": m_seg["prediction_coverage"],
            "confirm_p95": m_seg["detected_ae_p95"],
        },
        "DKPN": {"role": "diagnostic_only_possible_leakage", "enter_main_table": False},
        "bootstrap_Ours_minus_SegPhase_f1@0.5": d,
        "candidate_complementarity": complementarity.get("verdict_tag"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(verdict, out / "stage9_final_verdict.json")

    # reports
    (reports / "same_protocol_benchmark.md").write_text(
        f"""# Stage 9 — Same-protocol benchmark

**post_confirm_external_comparator_evaluation:** true  
**sota_claim_allowed:** false

## Table A — Waveform-only

| Method | F1@0.1 | F1@0.5 | P@0.5 | R@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|--:|--:|
| SegPhase-100Hz | {m_seg['f1@0.1']:.4f} | {m_seg['f1@0.5']:.4f} | {m_seg['precision@0.5']:.4f} | {m_seg['recall@0.5']:.4f} | {m_seg['miss_rate']:.4f} | {m_seg['prediction_coverage']:.4f} | {m_seg['detected_ae_p95']:.3f} |
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 0.8675 | 0.7635 | 0.1198 | 0.8802 | 1.927 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 0.7530 | 0.7015 | 0.0683 | 0.9317 | 58.657 |
| PhaseNet-STEAD | 0.4852 | 0.8176 | 0.8176 | 0.8176 | 0 | 1 | 6.100 |
| DKPN | — | — | — | — | — | — | diagnostic_only_INSTANCE |

## Table B — Catalog-assisted

| Method | F1@0.1 | F1@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 0 | 1 | 3.510 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 0 | 1 | 2.4655 |
| UNION oracle | 0.6017 | 0.8676 | 0 | 1 | 1.780 |

## Bootstrap (Ours − SegPhase) F1@0.5

mean={d['mean_delta']:+.4f} CI={ci}

Locked: scheme={scheme} thr={thr}
"""
    )
    (reports / "candidate_complementarity.md").write_text(
        f"""# Stage 9 — Candidate complementarity

```json
{json.dumps(complementarity, indent=2)}
```
"""
    )
    (reports / "catalog_assisted_transfer.md").write_text(
        """# Stage 9 — Catalog-assisted transfer onto DKPN/SegPhase

**Not executed for DKPN** (INSTANCE leakage diagnostic-only).

**SegPhase + frozen historical rescore:** deferred unless complementarity gate passes.
Default: do not replace Stage-6 main method.
"""
    )
    (reports / "runtime_benchmark.md").write_text(
        """# Stage 9 — Runtime

See `artifacts/results/stage9/runtime.json` if populated by cache logs (traces/s inferred from shard timings).
"""
    )
    (reports / "stage9_final_report.md").write_text(
        f"""# Stage 9 Final Report

**Verdict:** `{tag}`  
**sota_claim_allowed:** false  
**replace_stage6:** false

SegPhase confirm F1@0.5={m_seg['f1@0.5']:.4f}, F1@0.1={m_seg['f1@0.1']:.4f}, miss={m_seg['miss_rate']:.4f}, cov={m_seg['prediction_coverage']:.4f}, P95={m_seg['detected_ae_p95']:.3f}

DKPN: diagnostic_only_possible_leakage — excluded from main table.

Ours−SegPhase ΔF1@0.5 mean={d['mean_delta']:+.4f} CI={ci}
"""
    )
    save_json({"note": "see cache logs"}, out / "runtime.json")
    (out / "STAGE9.DONE").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
