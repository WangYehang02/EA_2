# Waveform pairwise pilot report (IO-fix resume)

Original FAILED preserved: `PAIRWISE.PILOT.FAILED` (incomplete_io).
Resumed: `PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX`.

Primary: `waveform_plus_scalar`  tau=0.5
Params waveform_only=39937  waveform_plus_scalar=40577

## Held-out F1@0.5
{
  "fixed_rescore_UNION": {
    "f1@0.5": 0.9195755652518842,
    "delta_f1@0.5": null,
    "fixes": 0,
    "breaks": 0,
    "net_fixes": 0,
    "switch_precision": NaN,
    "q2_recovery": null,
    "detected_ae_p95": 1.07,
    "tau": null
  },
  "resid_s_control": {
    "f1@0.5": 0.9350456168187227,
    "delta_f1@0.5": 0.01547005156683845,
    "fixes": 482,
    "breaks": 14,
    "net_fixes": 468,
    "switch_precision": 0.9717741935483871,
    "q2_recovery": null,
    "detected_ae_p95": 0.67,
    "tau": null
  },
  "scalar_pairwise": {
    "f1@0.5": 0.9403014676715589,
    "delta_f1@0.5": 0.020725902419674647,
    "fixes": 705,
    "breaks": 78,
    "net_fixes": 627,
    "switch_precision": 0.9003831417624522,
    "q2_recovery": 0.8834586466165414,
    "detected_ae_p95": 0.61,
    "tau": 0.5
  },
  "waveform_only": {
    "f1@0.5": 0.927773370355679,
    "delta_f1@0.5": 0.008197805103794753,
    "fixes": 327,
    "breaks": 79,
    "net_fixes": 248,
    "switch_precision": 0.8054187192118226,
    "q2_recovery": 0.40977443609022557,
    "detected_ae_p95": 0.8,
    "tau": 0.9
  },
  "waveform_plus_scalar": {
    "f1@0.5": 0.9407642469919344,
    "delta_f1@0.5": 0.021188681740050175,
    "fixes": 711,
    "breaks": 70,
    "net_fixes": 641,
    "switch_precision": 0.910371318822023,
    "q2_recovery": 0.8909774436090225,
    "detected_ae_p95": 0.6,
    "tau": 0.5
  }
}

## delta_waveform (wps - scalar) = 0.000463
bootstrap CI: {'mean': 0.0004569138201188082, 'ci95': [-0.0001671656736925181, 0.0011385179326135685]}

## Case summary vs scalar
{
  "n_waveform_fixes_over_scalar": 42,
  "n_waveform_breaks_over_scalar": 28,
  "net_waveform_vs_scalar": 14,
  "both_correct": 28418,
  "both_wrong": 1764,
  "scalar_only_correct": 28,
  "waveform_only_correct": 42
}

## Gate A / Gate B
{
  "A": {
    "best_pairwise_delta_f1_ge_0.010": true,
    "p95_not_materially_worse_best": true,
    "passed": true
  },
  "B": {
    "delta_f1_ge_0.003": false,
    "bootstrap_ci_excludes_0": false,
    "p95_not_worse_0.05": true,
    "or_q2_and_precision": false,
    "not_material_lt_0.001": true,
    "worse_than_scalar": false,
    "verdict": "waveform_increment_not_material",
    "passed": false
  }
}

## Cache
- Sequential HDF5 shards (4 workers), full-trace ZNE norm, window [c−1.5s, c+2.5s]
- n=92547 pairs, 1.185 GB, wall≈625 s, ~142–156 tr/s aggregate
- Equivalence: 500/500 exact (max_abs_diff=0)

## Scalar ablation (held-out ΔF1@0.5 vs fixed)
See `artifacts/results/pairwise_pilot/waveform_scalar_ablation.csv`.
resid_s removal hurts (~+0.0167); source removal almost neutral; waveform does **not** recover resid_s loss.

## Recommendation
- Prefer **scalar_pairwise** (Gate A). Do **not** enlarge CNN / continue waveform branch (Gate B: increment not material).
- **Do not** auto-enter full-dev or confirm. Full-dev only if you explicitly approve a **scalar** held-out→dev promotion later.

Verdict: `PAIRWISE.PILOT.FAILED_AFTER_RESUME` (`waveform_increment_not_material`)
Original `PAIRWISE.PILOT.FAILED` (incomplete_io) preserved.

Hard stop. No full-dev. No confirm. recommend_full_dev=false.
