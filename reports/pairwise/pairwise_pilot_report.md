# Pairwise pilot report (HARD STOP)

## Residual control audit
See `reports/pairwise/residual_control_equivalence_audit.md` — reweighting of train-only geometry; no label leakage; PAIRWISE.TRAIN_BLOCKED=false.

## Split / c1c2
{
  "quadrants_and_pool": {
    "n_traces_total": 206413,
    "n_events_total": 8058,
    "n_ge2": 156818,
    "Q1_c1_ok_c2_bad": 59449,
    "Q2_c1_bad_c2_ok": 5651,
    "both_correct": 82725,
    "both_wrong": 8993,
    "label_class_counts": {
      "choose_c1": 59324,
      "tie_ae_close": 58510,
      "single_candidate": 49595,
      "both_correct": 26432,
      "both_wrong": 7162,
      "choose_c2": 5390
    },
    "baseline_c1_acc_at_0.5": 0.9183190981188201,
    "top2_candidate_oracle_acc_at_0.5": 0.9456962497517114,
    "n_baseline_wrong": 16860,
    "n_ranking_recoverable_via_c2": 5651,
    "n_candidate_missing_or_both_wrong": 11209,
    "frac_wrong_that_are_ranking_recoverable": 0.3351720047449585,
    "frac_wrong_candidate_missing": 0.6648279952550415,
    "note_Q2_phaseB_was_1953": "phaseB Q2=1953 is a different split; this freeze is ranker_train only"
  },
  "split": {
    "calibration": 31657,
    "heldout_eval": 30252,
    "train": 144504
  }
}

## Overfit/smoke
PASS (see pilot.log)

## Calibration tau
tau=0.5
[
  {
    "tau": 0.5,
    "fixes": 799,
    "breaks": 71,
    "net": 728,
    "precision": 0.9183908045977012,
    "n_switch": 8185
  },
  {
    "tau": 0.6,
    "fixes": 762,
    "breaks": 48,
    "net": 714,
    "precision": 0.9407407407407408,
    "n_switch": 6191
  },
  {
    "tau": 0.7,
    "fixes": 716,
    "breaks": 31,
    "net": 685,
    "precision": 0.9585006693440429,
    "n_switch": 4391
  },
  {
    "tau": 0.8,
    "fixes": 661,
    "breaks": 15,
    "net": 646,
    "precision": 0.977810650887574,
    "n_switch": 3044
  },
  {
    "tau": 0.9,
    "fixes": 575,
    "breaks": 6,
    "net": 569,
    "precision": 0.9896729776247849,
    "n_switch": 1222
  }
]

## Held-out (scalar only; waveform NOT RUN)
{
  "fixed_rescore_UNION": {
    "f1@0.1": 0.64289964299881,
    "f1@0.5": 0.9195755652518842,
    "precision@0.5": 0.9195755652518842,
    "recall@0.5": 0.9195755652518842,
    "miss_rate": 0.0,
    "detected_ae_median": 0.07,
    "detected_ae_p95": 1.07,
    "delta_f1@0.5": 0.0,
    "fixes": 0,
    "breaks": 0,
    "net_fixes": 0
  },
  "resid_s_control": {
    "f1@0.1": 0.6556260743091366,
    "f1@0.5": 0.9350456168187227,
    "precision@0.5": 0.9350456168187228,
    "recall@0.5": 0.9350456168187228,
    "miss_rate": 0.0,
    "detected_ae_median": 0.07,
    "detected_ae_p95": 0.67,
    "delta_f1@0.5": 0.01547005156683845
  },
  "scalar_pairwise": {
    "f1@0.1": 0.6633611000925559,
    "f1@0.5": 0.9403014676715589,
    "precision@0.5": 0.9403014676715589,
    "recall@0.5": 0.9403014676715589,
    "miss_rate": 0.0,
    "detected_ae_median": 0.06,
    "detected_ae_p95": 0.61,
    "delta_f1@0.5": 0.020725902419674647,
    "delta_f1@0.1": 0.020461457093745916,
    "delta_p95": -0.4600000000000001,
    "fixes": 705,
    "breaks": 78,
    "net_fixes": 627,
    "switch_precision": 0.9003831417624522,
    "switch_coverage": 0.2482811053814624,
    "q2_n": 798,
    "q2_recovery": 0.8834586466165414,
    "tau": 0.5
  },
  "waveform_pairwise": "NOT_RUN_IO_BOUND",
  "waveform_plus_scalar_pairwise": "NOT_RUN_IO_BOUND"
}

## Bootstrap (scalar)
{
  "n_boot": 2000,
  "seed": 42,
  "delta_f1@0.5": {
    "mean": 0.02070000995904244,
    "ci95": [
      0.01816312577844821,
      0.023312352964793205
    ]
  }
}

## Gates
{
  "1_delta_f1_vs_fixed_ge_0.010": false,
  "2_delta_f1_vs_resid_ge_0.004": false,
  "3_bootstrap_ci_low_gt_0": true,
  "4_net_fixes_ge_1pct": true,
  "5_switch_precision_ge_0.80": true,
  "6_q2_recovery_ge_0.50": true,
  "7_f1_0.1_not_worse_than_0.001": true,
  "8_p95_not_worse_0.05": true,
  "9_miss_rate_zero": true,
  "10_outputs_from_c1_c2_only": true,
  "11_waveform_independent_info": false,
  "waveform_plus_scalar_completed": false
}

## Verdict
PAIRWISE.PILOT.FAILED — primary waveform_plus_scalar not completed (IO). No full-dev. No confirm. Hard stop.
