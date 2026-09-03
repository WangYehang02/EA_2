# Stage 6 Final Confirmatory Report

**Status:** `strong_confirmed`  
**Main method:** `fixed_rescore_UNION`  
**Confirm:** CONSUMED  
**SOTA claim allowed:** false  
**Multistation:** false  

## Population

- confirm events/traces (all): 2700 / 74753
- S-labelled eval: 43090 traces / 2669 events

## Core metrics

| Method | F1@0.1 | F1@0.5 | P95 | miss | coverage |
|--|--:|--:|--:|--:|--:|
| STEAD top-1 | 0.4852 | 0.8176 | 6.100 | 0.0000 | 1.000 |
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 3.510 | 0.0000 | 1.000 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 2.465 | 0.0000 | 1.000 |
| UNION oracle (ceiling) | 0.6017 | 0.8676 | 1.780 | — | — |

## Bootstrap (fixed UNION − STEAD)

ΔF1@0.5 mean=+0.0198 CI=[0.01734504340402719, 0.022288600930271577]

## Claims

Catalog-assisted S-phase candidate re-picking/refinement only. Not blind picker. Not SOTA.
