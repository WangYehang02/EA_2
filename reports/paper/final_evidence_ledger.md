# Final Evidence Ledger

Status: **LOCKED**  
FINAL_EVIDENCE_LOCK.sha256: `85bddb86ec4603fcb670cc0e0b813097b8f91119b7536e2cc6959ad9cd46eb20`  
Full-dev method lock: `1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b`  
Confirm execution lock: `0839e1a8f70123c196867f1d2da417e9f913d8359e917c557959b9848f620a15`

Declaration: All primary method development ended before final evidence packaging.

| Claim ID | Claim | Split | Value | Artifact | SHA256 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | Scalar pairwise improves F1@0.5 over fixed | confirm | +0.0162 (scalar=0.8535, fixed=0.8373) | confirm_scalar_metrics.json / confirm_fixed_metrics.json | 946ecae9… / 63c62fae… | LOCKED |
| C02 | Bootstrap CI scalar vs fixed excludes zero | confirm | mean +0.0162; 95% CI [+0.0146, +0.0178] | confirm_bootstrap_scalar_vs_fixed.json | 2afdb3e7… | LOCKED |
| C03 | Scalar exceeds residual-only control | confirm | +0.0076 (resid=0.8459) | confirm_scalar_metrics.json / confirm_resid_metrics.json | 946ecae9… / 332f2636… | LOCKED |
| C04 | Bootstrap CI scalar vs resid excludes zero | confirm | mean +0.0076; 95% CI [+0.0065, +0.0088] | confirm_bootstrap_scalar_vs_resid.json | 5d31035f… | LOCKED |
| C05 | Confirm coverage | confirm | n=43090 traces; 2669 events; ≥2 cand=27560 | confirm_final.json | 991b359d… | LOCKED |
| C06 | Switch outcomes | confirm | fixes=907; breaks=210; net=+697; switch_prec≈0.812; Q2≈0.773 | confirm_scalar_metrics.json | 946ecae9… | LOCKED |
| C07 | Effect replicates on held-out | pilot-heldout | scalar-fixed +0.0207; scalar-resid +0.0053 | ablation_metrics.json | 5ade854c… | LOCKED |
| C08 | Effect replicates on full-dev | phaseB-full-dev | scalar-fixed +0.0137; scalar-resid +0.0073 | fulldev_final.json | dedc35d9… | LOCKED |
| C09 | Effect ratios (confirm/full-dev) | cross-split | scalar-fixed ≈1.18×; scalar-resid ≈1.05× | effect_replication_summary.csv | 6aaba1ac… | LOCKED |
| C10 | Ranking-recoverable good candidates concentrated at rank 2 | forensic | 1820/2173=0.838; top3=0.979; top5=1.0 | ranking_gap_rank_distribution.json | 3a1ac67d… | LOCKED |
| C11 | Residual control ≈ same train-only geometry | pilot audit | corr(resid expected, base_tau)≈0.998 | residual_control_audit.json | 0a69a68d… | LOCKED |
| C12 | Waveform conditional increment not material | pilot | Δ=+0.00046; CI crosses 0 | waveform_bootstrap.json | 06b41922… | LOCKED |
| C13 | Moveout legal best ≤ single-station resid story | phaseB | ΔF1@0.5≈+0.0056; agree≈0.983; NO-GO | moveout_best.json | (see moveout dir) | LOCKED |
| C14 | Hard same-ring abstain protocol conflict | phaseB | ΔF1≈+0.0078 but abstain≈5.8% | soft_ring_best.json (hard_gate_best_ref) | (soft dir) | LOCKED |
| C15 | Soft same-ring small | phaseB | ΔF1≈+0.0024 | soft_ring_best.json | (soft dir) | LOCKED |
| C16 | Method / confirm locks intact | meta | FULLDEV.PASSED; CONFIRM.CONFIRMED; τ=0.50 | FULLDEV_METHOD_LOCK / CONFIRM_EXECUTION_LOCK | body hashes above | LOCKED |
| C17 | Fixed P95 / F1@0.1 on confirm | confirm | F1@0.1=0.5035; P95≈2.47 s | confirm_fixed_metrics.json | 63c62fae… | LOCKED |

Full SHA256 values are recorded in `artifacts/results/final_evidence/FINAL_EVIDENCE_LOCK.json` under `artifact_sha256`.

Rounding rule: read full-precision floats from artifacts, then round for report display (F1 to 4 decimals; rates to 3; P95 to ~2).
