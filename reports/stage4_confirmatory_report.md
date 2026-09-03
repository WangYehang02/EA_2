# Stage 4 Confirmatory Report

## 1. Executive summary

- **Main method (frozen):** `fixed_catalog_rescore` — catalog-assisted S-phase candidate re-picking/refinement.
- **Holdout:** 108 traces / 9 events (unused chronological test leftovers).
- **Underpowered:** True — Stage-3 test-only already consumed most test events; target 20k/1k not reachable without reuse.
- **SOTA claim allowed:** false.
- **Stable improvement claim from Stage-4 alone:** False.

Key holdout S metrics:

| Method | F1@0.1 | F1@0.5 | e2e P95 |
|--------|-------:|-------:|--------:|
| PhaseNet | 0.421 | 0.678 | 1.030 |
| fixed catalog_rescore | 0.433 | 0.690 | 0.727 |
| learned gate (ablation) | 0.421 | 0.678 | 1.030 |

## 2. Method lock

Locked before inference (`artifacts/results/stage4/method_lock.json`):

- PhaseNet weight: STEAD + UTC remap
- K=5, shrinkage_k=50
- Lambdas from Stage-2 VAL search (not Stage-3/4 test): S lw=0.5, lh=2.0, lp=0.0
- **Disclosure:** Stage-3 test metrics were *viewed* when choosing fixed over gate as main method; lambdas themselves were selected on Stage-2 validation, not on Stage-3 test-only.

## 3. Holdout construction and leakage audit

- Selection used only event_id / origin_time / availability / counts (no labels/SNR/preds).
- Event overlap with Stage1–3 used events: **0**
- Overlap train/val: **0 / 0**
- Warning: Holdout is far below target 20k traces / 1k events because Stage-3 test-only already consumed 1482/1500 chronological test events. Confirmatory power is limited; report this explicitly.

## 4. Main results

See `artifacts/results/stage4/tables/main_results.csv` and `ablation_results.csv`.

Interpretation note: gains concentrated on reducing gross wrong-peak / long-tail e2e errors (F1@0.5 / P95) should **not** be phrased as sub-sample-point precision SOTA if F1@0.1 does not improve.

## 5. Bootstrap significance

Event-level paired bootstrap, n=5000.

See `artifacts/results/stage4/bootstrap_confirmatory.json` and `tables/bootstrap_results.csv`.

With only 9 events, CIs are wide; do not over-claim.

## 6. Ablation

Includes distance-only, raw path, unshrunk/shrunk residual, shuffled/biased history, no-history identity, learned gate, oracles.

## 7. Difficult subsets

Stage-2 predefined bins only: `tables/hard_subset_results.csv`. Groups with `small_n=true` are diagnostic only.

## 8. Weight transfer

`tables/transfer_weights_results.csv` (stead / ethz / scedc if run). Instance weight excluded from formal results.

## 9. Noise / fallback

Catalog rescore without history must identity-fallback toward PhaseNet candidate ranking. This method requires event context and path history; it is **not** a continuous event detector.

## 10. Compute cost

See `tables/compute_cost.csv`.

## 11. Failure cases / figures

See `artifacts/results/stage4/figures/`.

## 12. Limitations

1. Confirmatory holdout is tiny after Stage-3 consumption of test events.
2. Catalog-assisted (needs origin/location/path).
3. Learned gate did not beat fixed on Stage-3; kept as ablation only.
4. No public-benchmark same-protocol SOTA comparison.

## 13. Paper claim boundary

Allowed (if supported by Stage-2/3 + careful language): catalog-assisted residual-guided candidate re-ranking improves an external pretrained picker for S-phase repicking on INSTANCE.

**Forbidden:** blind phase-picking SOTA; continuous detector claims; hiding Stage-3 inspection of test metrics when choosing main method.

## 14. Final recommendation

- **Main method:** fixed catalog_rescore
- **Ablation:** learned gate
- **Next step:** paper writing — do **not** train GNN or retune on holdout
- Verdict JSON: `artifacts/results/stage4/stage4_final_verdict.json`

Recommended claim text:

> On the Stage-4 unused-test confirmatory holdout, fixed catalog_rescore remains the designated main method, but the holdout is severely underpowered (only leftover unused test events after Stage-3), so a 'stable improvement' claim is not allowed from Stage-4 alone. Rely on Stage-2/3 test-only evidence with explicit catalog-assisted scope; do not claim phase-picking SOTA.
