# Waveform cache equivalence audit

**passed=True**  n_checked=500  n_exact=500
max_diff=0.0  mean_diff=0.0

Method: reload ENZ from InstanceHDF5Reader, apply frozen crop_pair_from_wave,
compare to memmap cache float32 crops/masks.

```json
{
  "n_checked_total": 500,
  "n_exact_total": 500,
  "max_observed_diff_global": 0.0,
  "mean_observed_diff_global": 0.0,
  "passed": true,
  "threshold": "max_abs_diff == 0 preferred; fail if >1e-6",
  "by_split": [
    {
      "split": "train",
      "n_checked": 200,
      "n_exact": 200,
      "max_observed_diff": 0.0,
      "mean_observed_diff": 0.0,
      "n_mismatch_gt_1e-6": 0,
      "mismatches_head": [],
      "passed": true
    },
    {
      "split": "calibration",
      "n_checked": 100,
      "n_exact": 100,
      "max_observed_diff": 0.0,
      "mean_observed_diff": 0.0,
      "n_mismatch_gt_1e-6": 0,
      "mismatches_head": [],
      "passed": true
    },
    {
      "split": "heldout_eval",
      "n_checked": 200,
      "n_exact": 200,
      "max_observed_diff": 0.0,
      "mean_observed_diff": 0.0,
      "n_mismatch_gt_1e-6": 0,
      "mismatches_head": [],
      "passed": true
    }
  ],
  "wall_s": 45.118510484695435,
  "cache_lock_sha256": "ebe9371c0485f99cf1d07b782d533fa9c4bad1f8799293d714f3b9882a68a93a"
}
```
