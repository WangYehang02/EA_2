# 方法表：为什么有这么多方法，最终主方法是什么

## 一句话结论

**最终主方法 = `scalar_pairwise`（scalar pairwise reranker）**  
任务 = **catalog-assisted S-phase candidate reranking**  
它只做一件事：在冻结的 UNION 候选里，对 `fixed_score` 最高的两个候选（c1/c2）做二选一，**不生成新到时**。

其余方法不是“并列主方法”，而是：

1. **Baseline**（必须打败的起点）  
2. **Control**（检验增益是否可被更简单物理先验解释）  
3. **Ablation / 负结果分支**（说明更复杂信息未必带来条件增益）  
4. **Diagnostic oracle**（只用于证明有 headroom，不可部署）

---

## Table 0. 方法角色总表（建议放 Methods / Results 开头）

| Method | 角色 | 用了什么信息 | 做什么 | 是否主方法 | 最终定位 |
| --- | --- | --- | --- | --- | --- |
| UNION candidates | 候选生成（冻结） | 波形 → base pickers | 产生 S 候选集合 | 否（输入） | 方法输入，不是贡献点 |
| fixed_rescore_UNION | **Baseline** | 候选概率 + catalog-assisted expected arrival | 按固定公式打分取 argmax | 否 | 对照起点 |
| resid_s control | **Strong control** | 同上 + train-only residual kernel | 对固定分做残差重加权后取 argmax | 否 | 强物理对照；证明 pairwise 增益不可被简单 residual 完全解释 |
| **scalar_pairwise** | **Proposed / 主方法** | c1/c2 的标量特征（含 residual 等） | 学习比较 c1 vs c2，τ=0.50 二选一 | **是** | Confirm **CONFIRMED**；三阶段 replicated |
| hard same-ring gate | Ablation（几何硬门控） | 同环邻站几何一致性 | 不一致则 abstain | 否 | NO-GO：abstain 破坏 always-output |
| soft same-ring | Ablation（几何软加权） | 同环几何 soft score | 软重打分 | 否 | 显著但很小；非主线 |
| moveout | Ablation（多台 moveout） | 多台几何 / moveout | 几何重打分 | 否 | NO-GO：增益≈单台 residual |
| waveform_only | Ablation（波形 pairwise） | 局部波形 crop | 仅波形 pairwise | 否 | 孤立有信息，弱于 scalar |
| waveform+scalar | Ablation（波形增量） | 波形 + scalar | 条件增量检验 | 否 | 增量不显著（CI 跨 0） |
| candidate oracle | Diagnostic only | 真值（仅诊断） | 若已知正确候选能否选中 | 否 | 证明有 recoverable headroom；**不可写为方法** |

---

## 为什么看起来“方法很多”

实验不是在堆 SOTA picker，而是回答一条因果链：

```text
候选里有没有正确 S？
    ↓ 有（ranking-recoverable）
错误是不是“排错序”而不是“没生成”？
    ↓ 是（rank2 占 83.8%）
简单固定打分够不够？
    ↓ Baseline = fixed_rescore_UNION
简单残差先验能否解释全部增益？
    ↓ Control = resid_s
学习 pairwise 比较能否再进一步？
    ↓ Proposed = scalar_pairwise  ← 主方法
更复杂信息（多台几何 / 波形）还有没有条件增益？
    ↓ Ablations → 大多 NO-GO / 很小
```

因此：

| 你看到的“方法” | 实际含义 |
| --- | --- |
| fixed / resid / scalar 三行主表 | **Baseline → Control → Proposed** |
| hard/soft/moveout/waveform | **消融与负结果**，证明“不是越复杂越好” |
| oracle | **动机诊断**，不是可比方法 |

---

## 最终主方法定义（锁死）

| 项目 | 冻结定义 |
| --- | --- |
| 名称 | `scalar_pairwise` / scalar pairwise reranker |
| 候选源 | UNION candidates（冻结，不改 generator） |
| c1 | `argmax fixed_score`（lw=0.5, lh=2, lp=0；并列 → `candidate_index`） |
| c2 | 第二高 `fixed_score` |
| 决策 | 仅允许输出 c1 或 c2；`<2` 候选时直接输出 c1 |
| τ | **0.50**（冻结） |
| 禁止 | abstain；选 rank≥3；生成新 pick；修改 arrival time |
| 不是什么 | blind picker / end-to-end picker / continuous detector / multi-station picker |

推荐句：

> The proposed method does not generate new phase candidates; it reranks the two highest-scoring S-phase candidates from a frozen candidate set.

---

## 主表里三行分别回答什么问题

| 行 | 问题 | Confirm F1@0.5 | 答案 |
| --- | ---: | --- |
| fixed_rescore_UNION | 固定打分选候选有多好？ | 0.8373 | Baseline |
| resid_s control | 加上 train-only residual 够不够？ | 0.8459（+0.0085） | 有帮助，但不是终点 |
| **scalar_pairwise** | 学习 pairwise 是否还有额外增益？ | **0.8535**（相对 fixed **+0.0162**；相对 resid **+0.0076**） | **是 → 定为主方法** |

Bootstrap（confirm，event，5000）：

- vs fixed：[+0.0146, +0.0178]（不含 0）  
- vs resid：[+0.0065, +0.0088]（不含 0）

---

## 消融分支为什么保留（但不进主方法）

| 分支 | 我们本来想检验的假设 | 结果摘要 | 为何不是主方法 |
| --- | --- | --- | --- |
| hard same-ring | 多台几何硬一致性可纠错 | ΔF1≈+0.0078，但 abstain≈5.8% | 协议不兼容 always-output |
| soft same-ring | 软几何加权可稳健提升 | ΔF1≈+0.0024 | effect 太小 |
| moveout | 多台 moveout 提供额外信息 | 合法最佳 ΔF1≈+0.0056；与单台 resid 预测一致≈98.3% | 额外多台信息未主导 |
| waveform_only | 局部波形足以 rerank | 相对 fixed 有增益，但低于 scalar | 不优于主方法 |
| waveform+scalar | 波形在 scalar 之上还有条件增益 | Δ≈+0.0005，CI 跨 0 | increment not material |

这些应放 **Supplement / Discussion**，用来支撑：

> 在本研究设定下，额外波形与多台几何信息的条件增益有限。

而不是写成“我们提出了五个新方法”。

---

## 论文里建议怎么写（防误解）

**可以写：**

- 我们比较了 baseline、residual control 与 proposed pairwise reranker  
- 并对几何 / 波形分支做了 controlled ablations  

**不要写：**

- 我们提出了多种 phase picker  
- 最终方法是 multi-station / waveform 融合系统  
- oracle / hard-gate 是可部署主方法  

**主故事只留一条线：**

`fixed → resid control → scalar_pairwise（confirmed）`  
旁支：`geometry / waveform ablations → limited conditional benefit`
