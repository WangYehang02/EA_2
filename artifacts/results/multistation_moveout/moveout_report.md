# Moveout residual pilot + ranking-gap forensic — phaseB full-dev

## Sanity
{
  "lambda0_equals_baseline": true,
  "lambda0_matches_frozen_npy": true,
  "no_neighbor_unchanged": true,
  "always_from_union_set": true,
  "loo_n_neighbors_is_n_minus_1": true,
  "train_only_tt_no_phaseB_fit": true,
  "confirm_untouched": true,
  "stage6_lock_untouched": true,
  "shuffle_event_delta": 0.005372710297503969,
  "shuffle_resid_delta": 0.005372710297503969,
  "shuffle_note": "shuffle_event/resid barely reduced \u0394F1; gain largely single-station |resid_s| vs T_S_hat",
  "single_station_resid_control_delta_f1@0.5": 0.006930681727057131,
  "moveout_vs_singlestation_agree": 0.9828966812917416
}

## Ranking-gap
{
  "n_recoverable": 2173,
  "rank_good_eq_2": 1820,
  "rank_good_eq_3": 307,
  "rank_good_eq_4": 40,
  "rank_good_eq_5": 6,
  "rank_good_gt_5": 0,
  "P_rank_good_le_2": 0.8375517717441325,
  "P_rank_good_le_3": 0.9788311090658076,
  "P_rank_good_le_5": 1.0,
  "mean_rank_good": 2.1863782788771284,
  "median_rank_good": 2.0,
  "mean_n_candidates": 2.4960883571099863,
  "median_delta_score_good_minus_bad": -0.45396205496610503
}

## AUC
{
  "same_ring_s": 0.5870960491038927,
  "same_ring_sp": 0.5686782549706015,
  "moveout_s": 0.6829152945888216,
  "moveout_sp": 0.6244044621145061,
  "single_station_abs_resid_s": 0.7227820579592814
}

## Oracle ceiling
ΔF1@0.5=+0.0069

## Best legal moveout
{
  "tag": "absolute_s_weighted_median_sr0.5_l1.0_mn1",
  "f1@0.1": 0.5575361140068505,
  "f1@0.5": 0.8724296335330439,
  "precision@0.1": 0.5575361140068505,
  "recall@0.1": 0.5575361140068505,
  "precision@0.5": 0.8724296335330439,
  "recall@0.5": 0.8724296335330439,
  "miss_rate": 0.0,
  "detected_ae_median": 0.09,
  "detected_ae_p95": 1.52,
  "wrong_peak_rate": 0.1275703664669561,
  "frac_ae_gt_1s": 0.07065858659915457,
  "frac_ae_gt_5s": 0.029418166405095484,
  "frac_ae_gt_10s": 0.025259757368861192,
  "frac_ae_gt_30s": 0.016816926901355205,
  "delta_f1@0.5": 0.005624735087578658,
  "delta_f1@0.1": 0.00357416975015179,
  "delta_p95": -0.1399999999999999,
  "fixes": 626,
  "breaks_all": 135,
  "net_fixes_minus_breaks": 491,
  "recoverable_oracle_gap_recovered_pct": 0.2880809940174873,
  "n_baseline_wrong_but_recoverable": 2173,
  "changed_rate": 0.03501999014812184,
  "change_precision": 0.8226018396846255
}

## Single-station residual CONTROL
{
  "note": "score_fixed + \u03bb exp(-resid_s^2/(2\u03c3^2)); NO neighbors",
  "f1@0.1": 0.5592773761928219,
  "f1@0.5": 0.8737355801725224,
  "precision@0.1": 0.5592773761928219,
  "recall@0.1": 0.5592773761928219,
  "precision@0.5": 0.8737355801725224,
  "recall@0.5": 0.8737355801725224,
  "miss_rate": 0.0,
  "detected_ae_median": 0.09,
  "detected_ae_p95": 1.51,
  "wrong_peak_rate": 0.12626441982747758,
  "frac_ae_gt_1s": 0.07016599269128108,
  "frac_ae_gt_5s": 0.029372343715990974,
  "frac_ae_gt_10s": 0.025202479007480554,
  "frac_ae_gt_30s": 0.016805471229079078,
  "delta_f1@0.5": 0.006930681727057131,
  "fixes": 711,
  "breaks_all": 106,
  "net_fixes_minus_breaks": 605,
  "recoverable_oracle_gap_recovered_pct": 0.3271974229176254,
  "n_baseline_wrong_but_recoverable": 2173,
  "changed_rate": 0.03656650590539906,
  "change_precision": 0.8702570379436965
}

prediction agree moveout vs control: 0.983

## Ambiguity best
{
  "gate": "margin_lowest_30pct",
  "frac_allowed": 0.3000011455672276,
  "delta_f1@0.5": 0.004364611137204544,
  "net_fixes_minus_breaks": 381,
  "change_precision": 0.8290155440414507
}

## Bootstrap
{
  "f1@0.5": {
    "mean_delta": 0.005633067505107708,
    "ci95": [
      0.004972926739979122,
      0.006309767956090404
    ]
  },
  "f1@0.1": {
    "mean_delta": 0.003581943944068177,
    "ci95": [
      0.0027698884986905093,
      0.00438169012274537
    ]
  },
  "detected_ae_p95": {
    "mean_delta": -0.1458797500000003,
    "ci95": [
      -0.17999999999999994,
      -0.11999999999999988
    ]
  }
}

## Gap to UNION oracle: 0.0193

## Go/No-Go
{
  "GO": false,
  "criterion_A_delta_f1_ge_0.008_p95_ok": false,
  "criterion_B_recoverable_ge_30pct": false,
  "criterion_C_moveout_auc_ge_0.70": false,
  "verdict": "NO-GO for further multi-station geometry development",
  "next_step_if_nogo": "waveform-level pairwise ranker (P(rank_good<=2)=83.8%; Q2 large)",
  "key_evidence": "shuffle barely hurts; single-station |resid_s| control \u2248/\u2265 moveout \u2192 multi-station geometry not the primary driver"
}

catalog-assisted; confirm untouched. No GNN.
