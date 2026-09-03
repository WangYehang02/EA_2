# Stage 6 Phase C — Candidate Ranker Report

**Verdict:** `marginal_pass`  
**Best ranker:** `R3` {'seed': 2026, 'beta': 0.2, 'gamma': 1.0, 'n_params': 54722, 'ckpt': 'artifacts/models/stage6/ranker/R3_seed2026_b0.2_g1.0/best.pt'}  
**Strongest baseline:** `fixed_rescore_UNION` F1@0.5=0.8667  
**Ranker F1@0.5 / @0.1:** 0.8084 / 0.5501  
**ΔF1@0.5 / @0.1:** -0.0584 / -0.0038  
**detected_ae_p95:** 0.780 (Δ -0.890)  
**Oracle gap recovered:** -2.337 (oracle=0.8917)  
**Multistation may start:** `False`  
**Confirm sealed:** `true`

## Notes

- Candidate source frozen: UNION_STEAD5_IDA5 (Phase B schema).
- History: picker_train-only snapshot; shrinkage_k=50 frozen.
- Selected pick always from union candidates or none_of_k.

## Artifacts

- `artifacts/results/stage6/phaseC/phaseC_final_verdict.json`
- `artifacts/results/stage6/phaseC/phaseC_baselines.json`
- `artifacts/results/stage6/phaseC/phaseC_bootstrap.json`
