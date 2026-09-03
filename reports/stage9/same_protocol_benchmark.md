# Stage 9 — Same-protocol benchmark

**post_confirm_external_comparator_evaluation:** true  
**sota_claim_allowed:** false

## Table A — Waveform-only

| Method | F1@0.1 | F1@0.5 | P@0.5 | R@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|--:|--:|
| SegPhase-100Hz | 0.4310 | 0.6543 | 0.7032 | 0.6118 | 0.1299 | 0.8701 | 70.270 |
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 0.8675 | 0.7635 | 0.1198 | 0.8802 | 1.927 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 0.7530 | 0.7015 | 0.0683 | 0.9317 | 58.657 |
| PhaseNet-STEAD | 0.4852 | 0.8176 | 0.8176 | 0.8176 | 0 | 1 | 6.100 |
| DKPN | — | — | — | — | — | — | diagnostic_only_INSTANCE |

## Table B — Catalog-assisted

| Method | F1@0.1 | F1@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 0 | 1 | 3.510 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 0 | 1 | 2.4655 |
| UNION oracle | 0.6017 | 0.8676 | 0 | 1 | 1.780 |

## Bootstrap (Ours − SegPhase) F1@0.5

mean=+0.1830 CI=[0.1771963180860547, 0.18874328030937734]

Locked: scheme=B thr=0.7
