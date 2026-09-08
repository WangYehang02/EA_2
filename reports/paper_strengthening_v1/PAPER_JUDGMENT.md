# 一页论文判断（paper_strengthening_v1）

日期：2026-09-05  
Git：见 `artifacts/results/paper_strengthening_v1/locks/PROVENANCE.json`  
主强对照锁：`locks/MAIN_STRONG_CONTROL.LOCK.json`

---

## 1. Scalar 是否优于“强简单对照”？

**主强对照（仅用 calibration 自然总体选出，选完后冻结）：**

`C_base_tau_c1c2`，`σ=0.25 s`，`λ_history=4.0`  
（在 B resid / C 20 格 / D 线性 中，calibration F1@0.5 最高）

**唯一主比较：冻结 scalar − 主强对照（ΔF1@0.5）**

| Split | 角色 | Δ | 95% CI (event boot 5000) | 解读 |
| --- | --- | ---: | --- | --- |
| calibration | 选对照用，非独立 | +0.0026 | [+0.0014, +0.0042] | 正但不独立 |
| heldout_eval | 历史研发，非独立 | +0.0019 | [+0.0007, +0.0033] | 正但很小 |
| full-dev phaseB | 历史研发，非独立 | +0.0005 | [+0.00003, +0.0011] | CI 刚离开 0；**远小于**实用讨论尺度 0.003 |

结论（在已有历史分割上）：

- 相对 **fixed** 与 **登记 resid control**，scalar 仍更好（与旧文一致）。  
- 相对 **校准选出的强传播先验 C**，scalar 的增量 **很小**；full-dev 上几乎可忽略。  
- **不能**再把“显著高于简单 residual”叙述成论文主贡献而不提强简单先验。

---

## 2. 独立时段上整体流程是否有收益？

**INDEPENDENT_VALIDATION_NOT_COMPLETED**

原因：INGV FDSN 不可达；本地无合格 2021–2022 人工 S 包。  
**禁止**用 confirm / heldout / full-dev 替代。

---

## 3. 当前仍支持的结论

1. 任务是 **catalog-assisted candidate reranking**（选 c1/c2），不是 blind picker。  
2. Ranking-recoverable 错误中 good candidate 高度集中在近顶部（既有 forensic）。  
3. 传播/历史先验对照（resid 与 base_tau-as-expected）能解释大部分相对 fixed 的增益。  
4. 在共享 c1/c2 协议下，**强简单先验 C** 已接近 scalar；学习 pairwise 的额外增益不确定且幅度小。  
5. 线性 pairwise（无截距、无顺序相关权重）**弱于** B/C，说明“随便一个 pairwise”并非自动更强。

---

## 4. 应收缩的结论

| 旧表述倾向 | 应改为 |
| --- | --- |
| scalar 相对 residual 的确认增益是核心卖点 | 需并列报告：相对 **强简单传播先验** 增益很小 |
| confirmed improvement 暗示方法复杂度必要 | 仅支持相对 fixed/登记 resid；相对强先验 **未在独立时段验证** |
| 实用价值已充分证明 | full-dev Δ≈0.0005 < 0.003 讨论尺度 → **实际意义存疑** |
| 跨区域/普遍泛化 | **未做**；即使将来有 2021–22，也只是同区域独立时段 |

---

## 5. 若简单方法同样有效，论文如何调整贡献

建议贡献重心调整为：

1. **问题形式化**：catalog-assisted S 候选中的 near-top ranking 错误。  
2. **协议与对照**：在冻结候选与 c1/c2 下，系统比较 fixed / resid / 传播先验网格 / 线性 / 学习 pairwise。  
3. **发现**：强简单传播先验已捕获大部分可恢复增益；学习 reranker 的增量有限。  
4. （可选）机制与负结果：waveform / multi-station 条件增益有限（沿用既有证据）。

**不再主张**：复杂 pairwise 相对所有合理简单对照具有实质必要增益——除非独立时段主比较 CI 稳定且幅度可辩护。

---

## 6. 关键数字快照（历史分割，非独立）

F1@0.5：

| Method | cal | heldout | full-dev |
| --- | ---: | ---: | ---: |
| A fixed | 0.9161 | 0.9196 | 0.8668 |
| B resid | 0.9324 | 0.9350 | 0.8732 |
| C best (σ0.25,λ4) | 0.9365 | 0.9384 | 0.8800 |
| C registered (σ0.5,λ2) | 0.9357 | 0.9375 | 0.8782 |
| D linear | 0.9275 | 0.9306 | 0.8701 |
| E scalar (frozen) | 0.9391 | 0.9403 | 0.8805 |

补充：full UNION 上 `base_tau_as_expected_sig0p5` = **0.8791**（与历史审计一致）；c1/c2 限制同公式 = 0.8782。

---

## 7. 停止条件

本轮已完成：强对照、选择锁、开发分割评估、预测哈希、标签隔离 smoke、INGV 探测、数据需求、独立验证 runner 骨架。  
**不再**扩展网络或继续搜参至“成功”。
