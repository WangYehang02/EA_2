# Stage 5.1 Hierarchical Residual — Sanity Audit

**Final classification: `implementation_bug`**

`hierarchical_prior_recommended`: **false**

Stage 5.1 fine_shrunk is NOT the frozen catalog_rescore: it uses exact 0.1° path keys only (coverage 27.7%) without TemporalHistoryStore fallback (frozen coverage 85.9%). Identity fallback on fine_unseen leaves PhaseNet-scale tails (e2e P95~55s). Coarse/hierarchical appear strong only relative to this broken baseline; on the same val list, frozen catalog_rescore already has e2e P95~1.26s. Comparing 54.9s to Stage3 1.11s is also an evaluation_mismatch (different sets).

## 1. Same validation list, same cache, same λ/K/metrics

Evaluation set: Stage2 fixed-eval **validation** only (`n=5220` traces, `682` events).
PhaseNet candidates: frozen `phasenet_fixed_cache(_candidates).parquet`.
S λ frozen: `(lw,lh,lp)=(0.5,2.0,0.0)`, `k=50.0`, `K=5`.
Metric: `earthquake.metrics.match_picks` / `audit_pick_errors` (same as Stage 2/3 pipeline).

| Method | F1@0.1 | F1@0.5 | e2e MAE | e2e P95 | matched P95 | wrong-peak | miss |
|--|--:|--:|--:|--:|--:|--:|--:|
| PhaseNet only | 0.389 | 0.594 | 8.849 | 70.55 | 0.360 | 0.261 | 0.014 |
| Frozen catalog_rescore (Stage2 picks) | 0.440 | 0.689 | 0.943 | **1.26** | 0.370 | 0.133 | 0.000 |
| Frozen fine reimplemented | 0.440 | 0.689 | 0.943 | 1.26 | 0.370 | 0.133 | 0.000 |
| Stage5.1 fine_shrunk | 0.407 | 0.630 | 5.410 | **54.92** | 0.370 | 0.207 | 0.000 |
| Coarse-only (1°) | 0.443 | 0.694 | 0.752 | 1.08 | 0.370 | 0.126 | 0.000 |
| Hierarchical | 0.442 | 0.694 | 0.762 | 1.11 | 0.370 | 0.127 | 0.000 |
| Identity fallback | 0.379 | 0.581 | 8.853 | 70.56 | 0.360 | 0.269 | 0.000 |

**Coarse-only and hierarchical are reported separately** (not merged as a range).
On this val set they are essentially identical (`hier_equals_coarse=False`).

## 2. Frozen fine vs Stage5.1 fine_shrunk consistency

| Check | Value |
|--|--|
| Pick max |Δ| samples | 9.62e+03 |
| Pick max |Δ| seconds | 96.2 |
| Exact candidate agreement | 90.824% |
| Prior max |Δ| samples | 2.19e+03 |
| ΔF1@0.5 | -0.0593 |
| Δe2e P95 | 53.66 s |
| Equivalent? | **False** |

Coverage: frozen history available **85.9%** vs Stage5.1 fine_ok **27.7%**.

Frozen reimplementation vs Stage2 pick file agreement: 100.0% (max Δ samples 0).

## 3. e2e errors > 10 s (Stage5.1 fine_shrunk)

- Count: **346** (frozen catalog_rescore on same list: 60; PhaseNet: 555)
- CSV: `artifacts/results/stage5_1_ustc/sanity_large_errors_fine_shrunk.csv`
- No evidence of relative-time vs absolute-UTC mixing; sample indices stay on waveform grid.
- Dominant class: wrong-peak tails when fine history absent → identity/PhaseNet-like picks.

## 4. fine_unseen fallback

- Branch: `history_available=False` ⇒ `λ_h=0` (identity among K candidates).
- Does **not** inject 0 / NaN / origin / trace-start as an arrival time for ranking when fine_ok=False.
- Agreement fine_unseen vs identity: 1.0
- Agreement fine_unseen vs PhaseNet annotate pick: 0.9385430463576159 (may differ if annotate peak ∉ top-K)

## 5. e2e P95 definition

- Same functions as Stage 2/3: `match_picks` / `audit_pick_errors`.
- Miss encoding: missing pred → `missed_pick`; **excluded** from e2e AE/P95.
- e2e P95: 95th percentile of |pred−true|/sr over labeled∩predicted (includes wrong peaks).
- Matched P95: only |err|≤0.5 s.
- Units: samples ÷ sampling_rate_hz → seconds (no UTC timestamp arithmetic in the metric).

Why Stage2 val PhaseNet P95≈70 s while Stage3 P95≈2 s: **different evaluation sets**, not a unit bug.

## 6. Path bootstrap direction (clarified)

Definition used in Stage 5.1 repeatability:

`Δ = MAE(shuffled_path) − MAE(correct_path)`

| Quantity | Value |
|--|--|
| MAE shuffled | 0.411080 s |
| MAE correct | 0.336587 s |
| mean Δ | 0.074493 s |
| 95% CI | [0.069549, 0.080000] |
| CI entirely positive? | True |
| n_events | 1500 |
| n_traces | 13772 |
| P(Δ>0) | 1.0 |

Positive CI ⇒ correct path better than shuffled (repeatability still supported).

## 7. Stage3 1.11 s comparison

{'stage3_fixed_e2e_p95': 1.11, 'stage3_set': 'event-disjoint test-only 10k (NOT Stage2 fixed-eval val)', 'stage5_1_set': 'Stage2 fixed-eval validation traces only (n=5220)', 'comparable': False, 'reason': 'different trace/event lists; Stage2 val PhaseNet e2e P95 already ~70s'}

Frozen catalog_rescore on **this same val list** already has e2e P95 **1.26 s**, close to Stage5.1 coarse/hierarchical (~1.39 s). The 54.9 s figure is an artifact of the Stage5.1 fine_shrunk reimplementation gap, not evidence that the frozen main method has a 50 s tail on val.

## 8. Decision

| Question | Answer |
|--|--|
| Is 54.9 s a metric unit bug? | No |
| Is fine_shrunk == frozen fine? | **No (implementation_bug)** |
| Is hierarchical a proven upgrade over frozen fine? | **No** (≈ coarse ≈ frozen on val) |
| Keep hierarchical_prior_recommended? | **false** |
| Modify frozen Stage3/4? | No |
