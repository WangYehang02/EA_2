# Soft same-ring candidate rescoring — phaseB full-dev

## Sanity
{
  "lambda0_equals_baseline_sample": true,
  "lambda0_matches_frozen_npy": true,
  "no_neighbor_unchanged": true,
  "always_from_union_set": true,
  "confirm_untouched": true,
  "stage6_lock_untouched": true,
  "shuffle_event_delta_f1@0.5": 0.0005727836138064157,
  "shuffle_sp_delta_f1@0.5": 0.00043531554649289816,
  "shuffle_event_reduces_gain": true,
  "shuffle_sp_reduces_gain": true
}

## Baseline fixed_rescore_UNION
F1@0.5=0.8668 F1@0.1=0.5540 P95=1.660

## Hard ring gate best (ref)
min_support≥1, ±1.5s: ΔF1@0.5=+0.0078, P95=0.86, abstain≈5.8%

## Best soft
key=abs F1@0.5=0.8692 Δ=+0.0024 P95Δ=-0.090

## Modes
SP top1 Δ=+0.0021
absolute-S top1 Δ=+0.0024 (catalog-assisted origin_time path)
hybrid Δ=+0.0021
SP all_cand Δ=+0.0014
abs all_cand Δ=+0.0017

## Neighbor bins
neighbor_bin  n_traces  n_events  baseline_f1@0.5  new_f1@0.5  delta_f1@0.5  baseline_f1@0.1  new_f1@0.1  delta_f1@0.1  precision@0.5  recall@0.5  miss_rate  detected_ae_median  detected_ae_p95
           0      2924      1904         0.784884    0.784884      0.000000         0.446990    0.446990      0.000000       0.784884    0.784884        0.0                0.13           1.7855
           1      6547      2499         0.759432    0.764472      0.005040         0.439132    0.440813      0.001680       0.764472    0.764472        0.0                0.13           9.1280
         2-3     13506      3695         0.801199    0.804605      0.003406         0.477195    0.478972      0.001777       0.804605    0.804605        0.0                0.11           2.6700
         4-7     26769      4024         0.858306    0.861108      0.002802         0.563151    0.564160      0.001009       0.861108    0.861108        0.0                0.08           1.9800
         >=8     37547      2477         0.921565    0.922950      0.001385         0.603377    0.604869      0.001491       0.922950    0.922950        0.0                0.08           0.7700

## Recoverable
{
  "n_union_has_cand_within_0.5": 77839,
  "n_baseline_wrong_but_recoverable": 2173,
  "fixes": 337,
  "breaks_all": 131,
  "breaks_on_recoverable_traces": 131,
  "net_fixes_minus_breaks": 206,
  "recoverable_oracle_gap_recovered_pct": 0.15508513575701796
}

## Bootstrap
{
  "f1@0.5": {
    "mean_delta": 0.002360265354278626,
    "ci95": [
      0.0019003963893698267,
      0.0028251617484571444
    ]
  },
  "f1@0.1": {
    "mean_delta": 0.001357239796756286,
    "ci95": [
      0.0008121416130470238,
      0.0018934220053135663
    ]
  },
  "detected_ae_p95": {
    "mean_delta": -0.09065274999999996,
    "ci95": [
      -0.11999999999999988,
      -0.06987500000000053
    ]
  }
}

## Gap to UNION oracle 0.8917: 0.0225

Soft rescoring did **not** beat hard-gate best (+0.0078) on F1@0.5, and far below +0.010 success criterion.
catalog-assisted; confirm/Stage-6 locks untouched. No GNN started.
