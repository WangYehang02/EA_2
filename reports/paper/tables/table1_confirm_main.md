## Table 1. Confirm 主结果

| Method | F1@0.1 | F1@0.5 | Precision@0.5 | Recall@0.5 | P95 (s) | ΔF1@0.5 vs fixed | ΔF1@0.5 vs residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 0.8373 | 0.8373 | 2.47 | — | -0.0085 |
| resid_s control | 0.5108 | 0.8459 | 0.8459 | 0.8459 | 2.12 | +0.0085 | — |
| **scalar_pairwise** | 0.5277 | **0.8535** | 0.8535 | 0.8535 | 1.85 | **+0.0162** | **+0.0076** |

**Event bootstrap (5000):**

| Contrast | Mean ΔF1@0.5 | 95% CI |
| --- | ---: | --- |
| scalar vs fixed | +0.0162 | [+0.0146, +0.0178] |
| scalar vs resid | +0.0076 | [+0.0065, +0.0088] |

---
