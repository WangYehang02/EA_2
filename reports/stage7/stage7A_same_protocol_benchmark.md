# Stage 7A — Same-Protocol External Baseline Benchmark

**Disclosure:** This is *post-confirm external comparator evaluation*. Stage-6 confirm is already `CONFIRM.CONSUMED`. Comparators were **not** co-preregistered with the Stage-6 main method.

- Main method (unchanged): `fixed_rescore_UNION`
- Stage-6 method_lock SHA256: `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303`
- `sota_claim_allowed`: **false**

## Locked thresholds (dev-only)

- **PhaseNet-SCEDC**: threshold=0.05 (dev F1@0.5=0.7584)
- **PhaseNet-ETHZ**: threshold=0.05 (dev F1@0.5=0.8355)

## Confirm metrics (S-labelled)

| Method | F1@0.1 | F1@0.5 | P95 | miss | coverage |
|--|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 1.927 | 0.1198 | 0.8802 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 58.657 | 0.0683 | 0.9317 |
| STEAD_top1 | 0.4852 | 0.8176 | 6.100 | 0.0000 | 1.0000 |
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 3.510 | 0.0000 | 1.0000 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 2.465 | 0.0000 | 1.0000 |
| oracle_UNION | 0.6017 | 0.8676 | 1.780 | 0.0000 | 1.0000 |

## Bootstrap (fixed_UNION − comparator)

- **fixed_UNION_vs_PhaseNet-ETHZ**: ΔF1@0.5=+0.0251 95% CI=[+0.0215, +0.0288]
- **fixed_UNION_vs_PhaseNet-SCEDC**: ΔF1@0.5=+0.1110 95% CI=[+0.1060, +0.1161]
- **fixed_UNION_vs_STEAD_top1_frozen**: ΔF1@0.5=+0.0198 95% CI=[+0.0173, +0.0223]

## Interpretation rule

If fixed UNION's ΔF1@0.5 CI is entirely above 0 vs all valid external comparators, we may state it *outperformed the evaluated external pretrained baselines under our Stage 6 protocol*. We do **not** claim SOTA / best on INSTANCE / surpasses all phase pickers.

Generated UTC: 2026-08-17T07:13:05.582650+00:00
