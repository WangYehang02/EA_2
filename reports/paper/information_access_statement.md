# Information Access Statement

Task scope: **catalog-assisted** S-phase candidate reranking. This is **not** a blind picker.

## TRAIN TIME (allowed)

- Waveforms for picker-train / ranker-train traces  
- Waveform-derived base-picker probabilities and UNION candidates  
- Train-only history / travel-time MLP used to form `expected_s` / residual geometry  
- Catalog labels for training supervision of ranking (train split only)  
- Candidate `fixed_score` features and scalar pairwise features  
- Calibration of τ on the **calibration** split only (frozen τ=0.50 thereafter)

## DEV TIME (full-dev / phaseB; allowed for validation)

- Same frozen candidate generator and frozen scalar checkpoint  
- Catalog-assisted current-event source/origin metadata needed for fixed/residual scoring  
- Evaluation labels for metrics only (no parameter update; no τ retune; no feature redesign)

## CONFIRM TIME (allowed)

- Frozen method lock + frozen checkpoint + frozen τ  
- Confirm waveforms → frozen candidates → frozen scoring/reranking  
- Catalog-assisted current-event source/origin metadata as in the locked protocol  
- Evaluation labels for final metrics / bootstrap only

## Forbidden at all evaluation stages (and confirm especially)

- Target true S / true P as model inputs  
- Neighbor true picks as model inputs  
- Confirm-derived parameters, gates, or τ  
- Using confirm margins to design new decision rules  
- Retraining / feature changes after method lock  

## Causality summary

| Information | Role |
| --- | --- |
| Base-picker probabilities | Candidate generation (frozen) |
| Train-only travel-time / history | Propagation prior inside fixed score & residual control |
| Catalog-assisted source/origin | Expected arrival construction (task is catalog-assisted) |
| Scalar pairwise model | Chooses c1 vs c2 only |
| True S/P labels | Metrics / training targets; never confirm-time features |

Residual equivalence audit: residual control correlates r≈0.998 with the same train-only geometry as `expected_s`; it is a reweighting control, not new causal source information.
