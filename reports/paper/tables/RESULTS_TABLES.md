# 论文结果主表（Frozen Evidence）

来源：`artifacts/results/` frozen artifacts；由 `scripts/build_final_paper_tables.py` 同源数字生成。

- Confirm 覆盖：n = 43090 traces，events = 2669，≥2 candidates = 27560
- 主方法：`scalar_pairwise`（catalog-assisted S-phase candidate reranking）
- 状态：`FINAL_EVIDENCE.LOCKED` / confirm `CONFIRMED`
- 方法角色总表（为什么有这么多方法）：见 [`METHOD_TABLE.md`](METHOD_TABLE.md)

---

## Table 0. 方法角色速览

| Method | 角色 | 是否主方法 |
| --- | --- | --- |
| fixed_rescore_UNION | Baseline | 否 |
| resid_s control | Strong control | 否 |
| **scalar_pairwise** | **Proposed** | **是** |
| hard/soft ring, moveout, waveform* | Ablation / 负结果 | 否 |
| candidate oracle | Diagnostic only | 否 |

详情与取舍理由：[`METHOD_TABLE.md`](METHOD_TABLE.md)

---

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

## Table 2. 跨 split replication（F1@0.5）

| Split | Fixed | Residual | Scalar | Scalar − Fixed | Scalar − Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 0.9196 | 0.9350 | 0.9403 | +0.0207 | +0.0053 |
| full-dev | 0.8668 | 0.8732 | 0.8805 | +0.0137 | +0.0073 |
| confirm | 0.8373 | 0.8459 | 0.8535 | +0.0162 | +0.0076 |

Effect ratios（confirm / full-dev）：scalar−fixed ≈ 1.18×；scalar−resid ≈ 1.05×。

解释：effect replicated across splits；**不要**写成“完全无 distribution shift”。

---

## Table 3. Pairwise mechanism

| Split | Eligible pairs | Switches | Fixes | Breaks | Net | Switch precision | Q2 recovery | Q1 break rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confirm | 27560 | 11537 | 907 | 210 | 697 | 0.812 | 0.773 | 0.011 |
| full-dev | 55366 | 22216 | 1526 | 329 | 1197 | 0.823 | 0.781 | 0.008 |

---

## Table 4. Ranking forensic（recoverable cases）

n_recoverable = 2173

| Good candidate rank | Count | Fraction | Cumulative fraction |
| ---: | ---: | ---: | ---: |
| 2 | 1820 | 0.838 | 0.838 |
| 3 | 307 | 0.141 | 0.979 |
| 4 | 40 | 0.018 | 0.997 |
| 5 | 6 | 0.003 | 1.000 |

- P(rank_good ≤ 2) = 0.838
- P(rank_good ≤ 3) = 0.979
- P(rank_good ≤ 5) = 1.000

表述限定：Among **ranking-recoverable** errors, the correct candidate was overwhelmingly concentrated near the top of the candidate list.

---

## Supplement Table. Negative / non-primary branches

| Method | ΔF1@0.5 | Notes | Verdict |
| --- | ---: | --- | --- |
| hard same-ring gate | +0.0078 | P95→0.86 s，abstain=0.058 | NO-GO |
| soft same-ring | +0.0024 | ΔP95=-0.09 s | small / not primary |
| moveout (legal best) | +0.0056 | ΔP95=-0.14 s；≈单台 resid | NO-GO (geometry) |
| waveform-only | +0.0082 | held-out；孤立有信息，低于 scalar | not chosen |
| waveform+scalar vs scalar | +0.0005 | 95% CI [-0.0002, +0.0011]（跨 0） | not material |

---

## 数字 provenance

| 表 | 主要 artifact |
| --- | --- |
| Table 1 | `pairwise_confirm/confirm_{fixed,resid,scalar}_metrics.json` + bootstrap JSON |
| Table 2 | pilot `ablation_metrics.json` + `fulldev_final.json` + confirm metrics |
| Table 3 | confirm / fulldev scalar metrics |
| Table 4 | `multistation_moveout/ranking_gap_rank_distribution.json` |
| Supplement | soft_ring_best / moveout_best / waveform_bootstrap / ablation_metrics |

