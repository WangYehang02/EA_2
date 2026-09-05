# EA_2：Catalog-assisted S-phase Candidate Reranking

本仓库对应 GitHub 项目 [WangYehang02/EA_2](https://github.com/WangYehang02/EA_2)。

## 工作一句话

在 **已知 catalog 震源/发震信息** 的设定下，对 base picker 给出的 **冻结 UNION S 候选** 做 **候选重排序（reranking）**，而不是训练一个新的 blind / end-to-end phase picker。

**最终主方法：`scalar_pairwise`（scalar pairwise reranker）**  
状态：confirm **CONFIRMED**；证据包 **FINAL_EVIDENCE.LOCKED**

> The proposed method does not generate new phase candidates; it reranks the two highest-scoring S-phase candidates from a frozen candidate set.

---

## 任务边界（必须先读）

| 是 | 不是 |
| --- | --- |
| catalog-assisted S-phase **candidate reranking** | blind phase picking |
| 在已有候选中 **select** c1/c2 | end-to-end picker / 新生成到时 |
| 单台候选重排 | continuous detector |
| 相对 fixed + residual control 的确认增益 | multi-station SOTA picker claim |

---

## 最终主方法

1. Base pickers → **UNION** S candidates（冻结）  
2. `fixed_score`（lw=0.5, lh=2, lp=0）→ **c1 / c2**  
3. **scalar pairwise** 比较 c1 vs c2，阈值 **τ=0.50**  
4. 输出只能是 c1 或 c2；候选不足 2 个时输出 c1  
5. 禁止：abstain、选 rank≥3、生成新 pick、改 arrival time

方法角色说明（为什么有 baseline / control / 消融）：  
[`reports/paper/tables/METHOD_TABLE.md`](reports/paper/tables/METHOD_TABLE.md)

---

## Confirm 主结果（frozen）

覆盖：n = **43090** traces，**2669** events，≥2 candidates = **27560**

| Method | F1@0.5 | Δ vs fixed | Δ vs resid |
| --- | ---: | ---: | ---: |
| fixed_rescore_UNION | 0.8373 | — | −0.0085 |
| resid_s control | 0.8459 | +0.0085 | — |
| **scalar_pairwise** | **0.8535** | **+0.0162** | **+0.0076** |

Event bootstrap 5000：

- scalar vs fixed：mean +0.0162，95% CI **[+0.0146, +0.0178]**  
- scalar vs resid：mean +0.0076，95% CI **[+0.0065, +0.0088]**

跨 split replication（F1@0.5）：

| Split | Fixed | Resid | Scalar | S−F | S−R |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 0.9196 | 0.9350 | 0.9403 | +0.0207 | +0.0053 |
| full-dev | 0.8668 | 0.8732 | 0.8805 | +0.0137 | +0.0073 |
| confirm | 0.8373 | 0.8459 | 0.8535 | +0.0162 | +0.0076 |

完整表图与 claim 边界：

- 结果总表：[`reports/paper/tables/RESULTS_TABLES.md`](reports/paper/tables/RESULTS_TABLES.md)  
- 方法表：[`reports/paper/tables/METHOD_TABLE.md`](reports/paper/tables/METHOD_TABLE.md)  
- Evidence ledger：[`reports/paper/final_evidence_ledger.md`](reports/paper/final_evidence_ledger.md)  
- Claim audit：[`reports/paper/final_claim_audit.md`](reports/paper/final_claim_audit.md)  
- 术语锁：[`reports/paper/TERMINOLOGY_LOCK.md`](reports/paper/TERMINOLOGY_LOCK.md)  
- 信息访问 / leakage：[`reports/paper/information_access_statement.md`](reports/paper/information_access_statement.md)

---

## 为什么还有很多“别的方法”

它们不是并列主方法，而是实验链条上的角色：

```text
fixed baseline
  → residual control（简单物理先验能否解释增益？）
    → scalar_pairwise（学习 pairwise 是否还有额外增益？）← 主方法
      → geometry / waveform ablations（更复杂信息有无条件增益？）
```

负结果摘要：hard-ring abstain 不兼容 always-output；moveout ≈ 单台 residual；waveform 条件增量不显著。

---

## 仓库里上传了什么结果

为便于复现论文数字，本仓库上传了 **冻结证据包**（JSON/CSV/MD/图/锁文件），**不**上传大体量 waveform cache / parquet / npy：

| 路径 | 内容 |
| --- | --- |
| `artifacts/results/final_evidence/` | FINAL_EVIDENCE_LOCK、audit、表图 |
| `artifacts/results/pairwise_confirm/` | confirm 主结果、bootstrap、CONFIRMED 锁 |
| `artifacts/results/pairwise_fulldev/` | full-dev PASSED + method lock |
| `artifacts/results/pairwise_pilot/` | held-out ablation + 主 checkpoint（小） |
| `artifacts/results/multistation_*` | soft-ring / moveout 负结果关键汇总 |
| `reports/paper/` | 论文素材（表、图、claim、术语） |
| `reports/pairwise/`、`reports/multistation/` | 阶段报告 |

锁定哈希（以仓库内文件为准）：

- Full-dev method lock：`1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b`  
- Confirm execution lock：`0839e1a8f70123c196867f1d2da417e9f913d8359e917c557959b9848f620a15`

本地核验：

```bash
conda activate PS
cd /path/to/Earthquake
python scripts/audit_final_evidence.py
python scripts/build_final_paper_tables.py
python scripts/build_final_paper_figures.py
```

---

## 环境

```bash
conda activate PS
cd ~/yehang/Earthquake
pip install -e .
export INSTANCE_ROOT=/mnt/yehang/PSdetec/INSTANCE
```

完整数据与大规模 cache 仍需本地 `INSTANCE_ROOT`；GitHub 上是 **代码 + 冻结结果证据**，不是完整波形库。

---

## 早期阶段说明

仓库仍保留 PhaseNet / path-prior / multistation 早期诊断与工程脚本。  
**论文主线以 scalar pairwise confirm 为准**；早期 Stage 1.5 诊断结论不要覆盖最终主方法声明。

更细的中文讲解要点：[`reports/paper/方法介绍与讲解要点.md`](reports/paper/方法介绍与讲解要点.md)
