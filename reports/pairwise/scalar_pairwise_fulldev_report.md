# Scalar pairwise phaseB full-dev report

n_eval=87293 events=5341 tau=0.5

## Metrics
- fixed F1@0.5=0.866805 F1@0.1=0.553962 P95=1.6600
- resid  F1@0.5=0.873197 Δfixed=+0.006392
- scalar F1@0.5=0.880517 Δfixed=+0.013712 Δresid=+0.007320
- fixes=1526 breaks=329 net=1197 switch_prec=0.8226 Q2_rec=0.7814

## Bootstrap
- scalar vs fixed: {'mean': 0.01369929162495082, 'ci95': [0.012656906329984199, 0.014803389414578488]}
- scalar vs resid: {'mean': 0.007315861069286596, 'ci95': [0.006505596737837615, 0.008158885173542001]}

## Gates
{
  "1_vs_fixed": {
    "delta_f1_ge_0.008": true,
    "bootstrap_ci_gt_0": true,
    "p95_not_worse_0.05": true,
    "passed": true
  },
  "2_vs_resid": {
    "delta_f1_ge_0.003": true,
    "bootstrap_ci_gt_0": true,
    "not_material_lt_0.001": false,
    "verdict": "scalar_beats_resid",
    "passed": true
  },
  "3_switch_quality": {
    "switch_precision_ge_0.80": true,
    "net_fixes_gt_0": true,
    "passed": true
  }
}

Verdict: PASSED
Hard stop. No confirm.
