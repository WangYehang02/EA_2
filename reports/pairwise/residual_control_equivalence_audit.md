# Residual control equivalence audit

## Question
Why did `single-station |resid_s| soft` gain ≈ +0.0069 F1@0.5 over `fixed_rescore_UNION`, if fixed rescore already uses a history/theoretical S prior?

## Short answer
Both expected_s and base_tau_s derive from the same picker_train-only travel-time MLP / history store. resid_s control adds an absolute-time Gaussian on (tau_S - base_tau_s) with fixed σ=0.5s and λ=1 on top of fixed_score (which already includes lh=2 log-Gaussian vs expected_s with history_sigma). Gain is explained by kernel/centering/weight differences on the same train-only geometric prior — not by multi-station info and not by held-out labels. Treat as posthoc scalar control only; do not modify blind-confirmed Stage-6 method.

- Label leakage: **False**
- New external information: **False**
- Reweighting/kernel change of train-only geometry: **True**
- PAIRWISE.TRAIN_BLOCKED: **False**

## Definitions

### Old `expected_s_sample`
attach_expected_s: pred_tau_s = base_tau_s + shrunk_residual_s (path history or global); mapped into waveform sample index using origin_time/trace_start_time/sr

### New `T_hat_S`
base_tau_s from Stage-6 travel_time_baseline_mlp.pkl (picker_train-only fit)

### Residuals
- Old normalized: `(candidate_sample - expected_s_sample) / history_sigma_samples`
- New absolute: `tau_S(candidate) - base_tau_s  [seconds]`

### Scores
- Old: `0.5*log(p) + 2.0*log(exp(-0.5*z^2)+eps) when history_available`
- Control: `fixed_score + 1.0 * exp(-0.5*(resid_s/0.5)^2)`

## Train-only provenance
- history_fit_split: `picker_train_only_frozen_snapshot`
- confirm_excluded_from_fit: `True`
- baseline_sha256: `4295711d11c75d2ec71c9180e9eb9d02592c3bbe87fb9ca6432df443758a7d44`
- features_sha256: `da74662d09d4a83f641828d6677746fff5f613c0d072964067d8e234b1ccb7d9`
- phaseB labels used for T_hat fit: `False`

## Geometry comparison (phaseB candidate table)
- corr(resid_vs_expected_tau, resid_vs_base_tau) = **0.9983**
- median |resid_expected − resid_base| = **0.3249 s**
- median |expected_tau − base_tau| = **0.3249 s**
- history_available fraction = **0.811**

## Selection agreement / metrics
- table fixed argmax vs frozen npy agree = **1.000000**

| variant | F1@0.5 | ΔF1@0.5 | agree vs resid_control |
|--|--:|--:|--:|
| fixed_rescore_table_argmax | 0.8668 | +0.0000 | 0.963 |
| resid_control_base_tau | 0.8737 | +0.0069 | 1.000 |
| resid_add_vs_expected_tau | 0.8725 | +0.0057 | 0.972 |
| lh4 | 0.8679 | +0.0011 | 0.951 |
| lh8 | 0.8686 | +0.0018 | 0.922 |
| lh16 | 0.8691 | +0.0023 | 0.903 |
| fixedsig0p5_force_hist | 0.8753 | +0.0085 | 0.905 |
| base_tau_as_expected_sig0p5 | 0.8791 | +0.0123 | 0.881 |
| wave_plus_resid_base_only | 0.8680 | +0.0011 | 0.895 |

## Can λ/σ/kernel alone reproduce +0.0069?
Best non-identical repro variant: **base_tau_as_expected_sig0p5** with ΔF1@0.5=**+0.0123**.
Increasing `lh` alone does not fully match; switching the center to `base_tau_s` and/or using a fixed σ≈0.5 s absolute kernel recovers most of the effect. Additive `fixed + λ·exp(-resid_base²…)` is the reported control.

## Tie-break / ordering
All variants use argmax over UNION candidates with the same candidate_index ordering for ties (numpy nanargmax / pandas idxmax on score). No STEAD/IDA rank or label-based ordering.

## Policy implication
- Treat resid_s control as a **posthoc scalar control**, not a new method claim.
- Do **not** rewrite Stage-6 locked `fixed_rescore_UNION`.
- Safe to use as a pairwise-pilot baseline comparator if training data stays train-only.

## Artifacts
- `/home/yehang/yehang/Earthquake/artifacts/results/pairwise_pilot/residual_control_audit.json`
