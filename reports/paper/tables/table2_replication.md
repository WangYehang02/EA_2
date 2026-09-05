## Table 2. 跨 split replication（F1@0.5）

| Split | Fixed | Residual | Scalar | Scalar − Fixed | Scalar − Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 0.9196 | 0.9350 | 0.9403 | +0.0207 | +0.0053 |
| full-dev | 0.8668 | 0.8732 | 0.8805 | +0.0137 | +0.0073 |
| confirm | 0.8373 | 0.8459 | 0.8535 | +0.0162 | +0.0076 |

Effect ratios（confirm / full-dev）：scalar−fixed ≈ 1.18×；scalar−resid ≈ 1.05×。

解释：effect replicated across splits；**不要**写成“完全无 distribution shift”。

---
