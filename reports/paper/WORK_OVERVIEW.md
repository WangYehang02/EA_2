# 工作介绍与结果索引

## 我们在做什么

地震 S 波拾取里，一类常见可恢复错误是：

> **正确候选已经在 base picker 的候选列表里，但被另一个 plausible 候选排在前面。**

本工作在 **catalog-assisted** 设定下，把问题表述为：

**S-phase candidate reranking / re-picking**

而不是 blind picking 或端到端检测。

最终提出并确认的主方法是：

### `scalar_pairwise`

对冻结 UNION 候选中 `fixed_score` 最高的两个候选（c1、c2）做标量 pairwise 比较，τ=0.50，只允许二选一。

---

## 主结论（Confirm）

| | |
| --- | --- |
| 样本 | 43090 traces / 2669 events |
| fixed F1@0.5 | 0.8373 |
| residual control | 0.8459（+0.0085） |
| **scalar_pairwise** | **0.8535（相对 fixed +0.0162；相对 resid +0.0076）** |
| bootstrap vs fixed | [+0.0146, +0.0178] |
| bootstrap vs resid | [+0.0065, +0.0088] |
| switches | fixes 907 / breaks 210 / net +697 |
| 状态 | **CONFIRMED** + **FINAL_EVIDENCE.LOCKED** |

Ranking forensic（recoverable n=2173）：rank2 = **83.8%**，top3 = **97.9%**。

---

## 文档地图

| 文档 | 用途 |
| --- | --- |
| [README.md](../README.md) | 仓库总览（英文读者也可从这里进入） |
| [tables/METHOD_TABLE.md](tables/METHOD_TABLE.md) | 方法角色：为何有 baseline/control/消融 |
| [tables/RESULTS_TABLES.md](tables/RESULTS_TABLES.md) | 论文结果主表 |
| [final_results_summary.md](final_results_summary.md) | Results 叙事骨架 |
| [final_claim_audit.md](final_claim_audit.md) | 可写 / 需限定 / 禁止的 claim |
| [final_evidence_ledger.md](final_evidence_ledger.md) | 数字 → artifact 溯源 |
| [TERMINOLOGY_LOCK.md](TERMINOLOGY_LOCK.md) | 术语 |
| [information_access_statement.md](information_access_statement.md) | 训练/验证/confirm 信息边界 |
| [discussion_points.md](discussion_points.md) | Discussion 要点 |
| [方法介绍与讲解要点.md](方法介绍与讲解要点.md) | 讲解用中文要点 |

---

## 上传到 GitHub 的结果位置

```text
artifacts/results/final_evidence/     # lock + 表图
artifacts/results/pairwise_confirm/   # confirm 主结果
artifacts/results/pairwise_fulldev/   # full-dev + method lock
artifacts/results/pairwise_pilot/     # held-out + scalar ckpt
artifacts/results/multistation_*/     # 几何负结果摘要
reports/paper/                        # 论文素材
reports/pairwise/                     # 阶段报告
```

未上传：大体量 parquet / npy / waveform crop cache / 原始 INSTANCE 波形。
