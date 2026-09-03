# Paper candidate correction — timing metric naming (Stage 6)

**Does not modify frozen Stage 2–5 results or the main paper automatically.**

## Issue

The helper `match_picks` historically aliases absolute-error summaries as `e2e_*`.
Those statistics are computed on **labeled traces that also have a prediction**:

- wrong-peak errors **are** included;
- NaN / missed picks **are not** included in MAE/median/P95.

Therefore calling that P95 alone “complete end-to-end P95” is misleading.

## Preferred Stage-6 reporting

For each method report simultaneously:

1. F1@0.1 / 0.2 / 0.5
2. `detected_ae_median` / `detected_ae_mae` / `detected_ae_p95`
3. `miss_rate` (no-pick among labeled)
4. `wrong_peak_rate`
5. Optional penalized error only with an **explicit** miss penalty (not a replacement for the above)

## Frozen numbers

Stage 3/4 published “e2e P95” values remain numerically unchanged as artifacts;
interpret them as **detected non-missing AE P95** when comparing to Stage 6.
