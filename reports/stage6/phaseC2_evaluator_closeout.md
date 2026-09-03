# Stage 6 Phase C.2 — Evaluator Closeout

**Verdict:** `ranker_failed_predeclared_gate`  
**Recommended main method:** `fixed_rescore_UNION`  
**Ranker role:** negative_ablation  
**Forced-choice / none-fallback:** posthoc_diagnostic_only  

## Fix

`evaluate_stage6_candidate_ranker.py` now scores via `keyed_align_predictions(trace_name)`.
Historical overnight `ranker_dev_metrics.json` / `phaseC_final_verdict.json` preserved.

## Corrected R3 (frozen preds, keyed)

- F1@0.5 = 0.832278
- F1@0.1 = 0.567068
- none/miss = 0.225734
- detected_ae_p95 = 0.480000 (denom=67588)

## fixed_rescore_UNION

- F1@0.5 = 0.866805
- detected_ae_p95 = 1.660000

ΔF1@0.5 (R3−fixed) = -0.034527 (fails +0.01 gate)
