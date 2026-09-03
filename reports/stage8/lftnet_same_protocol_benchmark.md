# LFTNet same-protocol (Stage 6) benchmark

**Status:** not run (gate failed).

Frozen Stage-6/7A baselines remain the reference (reused, not re-inferred):

| Method | F1@0.1 | F1@0.5 | miss | coverage | P95 |
|--|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 0.1198 | 0.8802 | 1.927 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 0.0683 | 0.9317 | 58.657 |
| PhaseNet-STEAD | 0.4852 | 0.8176 | 0 | 1 | 6.100 |
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 0 | 1 | 3.510 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 0 | 1 | 2.4655 |
| UNION oracle | 0.6017 | 0.8676 | 0 | 1 | 1.780 |
| LFTNet | — | — | — | — | not run |

`sota_claim_allowed = false`
