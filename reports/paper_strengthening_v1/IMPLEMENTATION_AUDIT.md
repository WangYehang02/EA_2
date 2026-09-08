# Implementation audit (frozen artifacts)

```json
{
  "base_tau_as_expected_sig0p5": {
    "source_script": "scripts/audit_residual_control_equivalence.py",
    "formula": "fixed_candidate_scores(expected=base_tau_as_sample, sigma=0.5*sr, hist=all, lw=0.5, lh=2)",
    "historical_selection_scope": "FULL UNION argmax (audit)",
    "historical_phaseB_f1@0.5": 0.8791,
    "note": "Main table in this package restricts the SAME formula to shared c1/c2."
  },
  "resid_s_control": {
    "formula": "fixed_score + 1.0*exp(-0.5*(resid_s/0.5)^2); switch if s2>s1",
    "scope_pairwise": "c1/c2 only",
    "fulldev_f1@0.5": 0.8732
  },
  "scalar": {
    "ckpt": "artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt",
    "ckpt_sha256": "7f0d1caae205fcfddc9981609a18bda33f31d9881953a2d93dc1ef55ee4fdaea",
    "tau": 0.5,
    "features": 10,
    "standardization": "NONE in original scalar training",
    "pos_weight": "n_neg/n_pos on decisive train (~11.01); swap augmentation flips y",
    "pos_weight_issue": "Sample weight depends on label after optional swap; DOES affect original training. This package does NOT silently replace scalar; linear control forbids order-dependent weights.",
    "fulldev_f1@0.5": 0.8805
  },
  "splits": {
    "protocol": "ranker_train_only_event_split_70_15_15 seed=42",
    "split_lock_sha256": "747ab0c190caf35b63715b92dcb2a0216639dd19f92fa629285122f9988bad96",
    "heldout_and_fulldev_are_historical": true,
    "confirm_already_used_not_new_test": true
  }
}
```
