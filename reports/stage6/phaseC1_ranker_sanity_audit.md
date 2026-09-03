# Stage 6 Phase C.1 — Candidate Ranker Sanity Audit

**Confirm seal:** SEALED (start and end)  
**Verdict:** `implementation_or_evaluation_bug`  
**Recommended Stage 6 main method:** `fixed_rescore_UNION`  
**Multistation may start:** `false`  
**Confirm may unseal:** `false`

## 1. Implementation / evaluation bug

**Yes — overnight ranker metrics are invalid.**

- File: `scripts/evaluate_stage6_candidate_ranker.py`
- Bug: `pick_metrics(preds.pred, meta.true, meta.sr)` mixes **dataset group order** preds with **manifest row order** labels.
- Trace order agreement: **96.6103%**
- Misaligned positions: **2959**

| | F1@0.5 | F1@0.1 | P95 | miss/none |
|--|--:|--:|--:|--:|
| Overnight (wrong alignment) | 0.8084 | 0.5501 | 0.780 | 0.2257 |
| Corrected (same preds) | 0.8323 | 0.5671 | 0.480 | 0.2257 |
| fixed_rescore_UNION | 0.8668 | 0.5540 | 1.66 | 0.0 |

Predictions parquet itself is fine (`true_s_sample` paired correctly). Baselines were meta-aligned and remain valid.

**Fix (do not retrain this round):** score with `preds[['pred_s_sample','true_s_sample','sampling_rate_hz']]` or `reindex` to `meta.trace_name` before using meta labels; add order assert. Then re-run eval + bootstrap + finalize only.

## 2. Cohort

- events=5341, traces=87293, matches_expected=true
- confirm overlap: 0 / 0
- See `cohort_provenance.json` for hashes

## 3–5. Corrected R3 vs fixed (same cohort)

Corrected R3 still has **none_rate≈0.226**, coverage≈0.774, recall drop, and **P95 improvement is abstention-driven** (`p95_improvement_mainly_from_abstention=true`).

Even corrected, R3 does **not** meet strong-pass vs fixed (ΔF1@0.5 ≈ -0.035).

## 6. Selection

- `reported_best_ranker` = R3_seed2026_b0.2_g1.0
- `max_dev_f1_existing_ranker` = R3_seed2026_b0.2_g1.0 (same)
- Training still selected on ~12k-subset metrics (evaluation inconsistency, secondary)

## 7–8. Diagnostics / bootstrap

See `diagnostic_variants_metrics.json` and `bootstrap_vs_fixed.json` (computed with correctly aligned preds).  
Overnight `phaseC_final_verdict.json` left untouched as historical overnight output.

## 9–12. Decision

- verdict: `implementation_or_evaluation_bug`
- main method: `fixed_rescore_UNION`
- confirm sealed; multistation forbidden
- original overnight `marginal_pass` was based on **invalid ranker metrics** plus P95-from-abstention gate clause

## Artifacts

Under `artifacts/results/stage6/phaseC1/` including `evaluation_bug_repro.json` and `PHASEC1.DONE`.
