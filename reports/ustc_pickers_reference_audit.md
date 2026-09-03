# Stage 5.1 — USTC-Pickers Reference Audit

## 1. Executive summary

USTC-Pickers strengthens the **motivation** for regional domain shift and the
**complementarity** narrative (waveform adaptation vs catalog-assisted
candidate re-ranking). Train→validation analyses show that **path residuals are
repeatable**: correct path priors beat shuffled-path controls with event-level
bootstrap CIs excluding zero (correlation≈0.688).

A **coarse-to-fine hierarchical residual** passes the pre-registered validation
decision rules and is recommended only as
`promising future extension validated on the development set`.
Most of the validation pick gain equals **coarse (1°) residual backoff** on
fine-unseen paths; it must **not** enter Stage 3 main results.

Formal leakage-free full INSTANCE in-domain PhaseNet remains a
**reviewer-request** strengthening experiment (existing finetunes are debug-scale
and did not beat STEAD). Do **not** use Chinese USTC weights as INSTANCE
baselines; do **not** resume GNN work.

Frozen Stage 3/4 and `paper/main.*` hashes: **unchanged**.


> **SANITY AUDIT UPDATE:** `hierarchical_prior_recommended` set to **false**. Classification: `implementation_bug` (Stage5.1 `fine_shrunk` ≠ frozen `catalog_rescore`). See `reports/stage5_1_sanity_audit.md`.

## 2. USTC-Pickers overview

Zhu, Li & Fang (2023), *Earthquake Science* 36(2):95–112,
DOI 10.1016/j.eqs.2023.03.001.

- DiTing dataset; CN picker trained from scratch; fine-tuned tectonic (5) and
  provincial (33) pickers (+ Capital / CSES specials).
- Emphasizes regional waveform domain shift; reports diminishing returns from
  finer customization; S performance worsens with distance.
- Eval: ±0.6 s, 50 Hz — **not comparable** to this project's metrics.

## 3. Commonalities and differences

| | USTC-Pickers | This project |
|--|--|--|
| Goal | Adapt waveform picker weights | Freeze picker; re-rank S candidates |
| History | Regional labeled waveforms for fine-tuning | Train-only path travel-time residuals |
| Needs catalog at inference | No (blind picking) | Yes (catalog-assisted) |
| Hierarchy | CN → tectonic → provincial models | Optional multi-scale residual backoff (val-only here) |

## 4. Paper-argument borrow decisions

| Item | Covered now? | USTC support? | Decision | Section |
|--|--|--|--|--|
| A. Regional domain shift | Partial | Direct | **adopt** | Related Work / Intro |
| B. Complementarity | No | Direct (by contrast) | **adopt** | Related Work / Discussion |
| C. Diminishing returns vs complexity | No | Qualitative | **adopt_with_caveat** (not “GNN invalid”) | Discussion |
| D. Distance-related S difficulty | Stage2/4 frozen + val exploratory | Directional | **adopt_with_caveat** | Discussion |

Candidate text (not merged): `paper/candidate_edits/ustc_*.tex`.

### Forbidden exaggerations
No cross-paper F1 ranking; no causal claim that regional models “prove” path
residuals; no “10k samples universal recipe”.

## 5. Path residual repeatability (train→val)

Protocol: distance MLP frozen from Stage 2; history tables from **train only**;
evaluate on **validation only**; event-disjoint enforced.

| Prior | Val MAE (s) | median AE | P95 |
|--|--:|--:|--:|
| MLP only | 0.3687 | 0.2814 | 1.0058 |
| + distance-bin residual | 0.3671 | 0.2798 | 1.0028 |
| + shuffled path | 0.4111 | 0.3296 | 1.0610 |
| + correct path | 0.3366 | 0.2559 | 0.9186 |
| + shrunk path (k=50) | 0.3513 | 0.2678 | 0.9533 |
| + hierarchical | 0.2922 | 0.2268 | 0.7814 |

- Correct vs shuffled ΔMAE bootstrap CI: 0.0695–0.0800 (prob_improved=1.0)
- Seen-path residual correlation (val): **0.688**
- Between/within path MAD ratio: **1.750** (robust ratio, not formal ICC)
- Fine coverage on full val: seen frac=0.242

**Conclusion:** path residual behaves as a **repeatable directed source-region–station effect**, not a shuffled lookup artifact.

## 6. Hierarchical residual (validation-only feasibility)

Fixed eval **val** traces only (n=5220, events=682);
frozen K/λ/peaks; replace travel-time prior only.

| Method | Prior MAE | S F1@0.5 | e2e P95 | wrong-peak |
|--|--:|--:|--:|--:|
| PhaseNet | — | 0.594 | 70.55 | 0.261 |
| fine shrunk (current-like) | 0.3485 | 0.630 | 54.92 | 0.207 |
| coarse shrunk (1°) | 0.2956 | 0.686 | 1.39 | 0.136 |
| hierarchical | 0.2914 | 0.686 | 1.39 | 0.136 |
| hierarchical shuffled | 0.3943 | 0.681 | 1.72 | 0.143 |

Pass rules: `{"prior_mae_drop_ge_5pct": true, "f1_0.5_gain_ge_0.005": true, "e2e_p95_drop_ge_0.1s": true, "bootstrap_direction_stable": true, "gain_on_sparse_or_unseen": true, "fallback_unchanged": true, "complexity_acceptable": true}`

**Interpretation:** gains concentrate on **fine_unseen** paths where coarse history
is available (`hierarchical_val_by_group.csv`). Hierarchical ≈ coarse on this
val set; recommend as **future_work**, not a Stage 3 method change.

## 7. In-domain PhaseNet strong-baseline audit

Existing artifacts (`artifacts/phasenet_finetune/`, `finetuned_phasenet_metrics.json`):

1. Debug finetune (10k/2k traces, 5 epochs) never beat UTC-aligned STEAD on annotate val F1; `best.pt` kept pretrained.
2. Early failures: NPS vs PSN label order; BN running stats updated on crops.
3. Last-layer-only run: likewise `no_improve_keep_pretrained`.
4. **No** completed full-scale leakage-free INSTANCE train under `configs/phasenet_finetune_full.yaml`.
5. SeisBench `instance` weights remain **diagnostic-only** (leakage risk).
6. Reviewers may ask why only external STEAD is shown; a proper in-domain baseline would strengthen the paper but is **orthogonal** to the re-ranking claim.
7. Minimal sufficient experiment: full chronological train, event-disjoint val selection by annotate F1, BN/eval protocol locked, compare STEAD vs in-domain vs STEAD+fixed rescore on Stage 3 frozen list (**after** method lock; no retuning λ on test).
8. Cost: multi-GPU-day scale (full INSTANCE, ~30 epochs) — nontrivial.
9. Priority: **`reviewer_request`** (upgrade to before-submission if compute budget allows). Do not assert in-domain training is useless from debug failure; do not extrapolate USTC DiTing numbers to INSTANCE.

## 8. Non-comparable metrics

| Item | USTC-Pickers | This project |
|--|--|--|
| Data | DiTing | INSTANCE |
| Sampling rate | 50 Hz | 100 Hz |
| Window | 60 s | 120 s |
| Tolerance | ±0.6 s | ±0.1 / 0.5 s |
| Picker | China train/finetune | External STEAD + catalog rescore |
| Task | waveform picking | catalog-assisted repicking |

## 9. Risks and limits

- Val hierarchical gains may not transfer to Stage 3 test; not measured here by design.
- Coarse residual may over-smooth spatially; needs careful failure analysis.
- Catalog origin errors still bias residual priors.
- Stage 4 remains underpowered and unused for method choice.

## 10. Final decision table

| Candidate borrow | Decision |
|--|--|
| Regional domain-shift motivation | **adopt** |
| Add Related Work citation/paragraph | **adopt** |
| Waveform adaptation ↔ path re-ranking complementarity | **adopt** |
| Distance-related S difficulty discussion | **adopt_with_caveat** |
| Marginal gain vs deployment complexity | **adopt_with_caveat** |
| Hierarchical historical residual | **future_work** (not adopt_now) |
| Formal in-domain PhaseNet | **reviewer_request** |
| USTC models as INSTANCE baseline | **reject** |
| Continue GNN | **reject** |

See `artifacts/results/stage5_1_ustc/final_verdict.json`.
