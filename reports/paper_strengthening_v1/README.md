# paper_strengthening_v1 — README

补强简单对照 + 独立验证框架（**不覆盖**旧 confirm / method locks）。

## 快速结论

- **主强对照** = `C_base_tau_c1c2`（σ=0.25, λ_history=4.0），由 calibration 自然总体选出。  
- 冻结 **scalar** 相对该对照：full-dev ΔF1@0.5 ≈ **+0.0005**（CI 刚离开 0）。  
- **独立验证尚未完成**（INGV 不可达）。  

详见：[`PAPER_JUDGMENT.md`](PAPER_JUDGMENT.md)、[`DATA_REQUIREMENTS.md`](DATA_REQUIREMENTS.md)

## 命令

```bash
conda activate PS
cd /path/to/Earthquake

python scripts/paper_strengthening_v1/run_strong_controls_dev.py
python scripts/paper_strengthening_v1/probe_ingv.py
python scripts/paper_strengthening_v1/estimate_power_proxy.py
python scripts/paper_strengthening_v1/run_independent_period_eval.py

pytest -q tests/test_paper_strengthening_v1.py
```

## 目录

| Path | 内容 |
| --- | --- |
| `configs/paper_strengthening_v1/protocol.yaml` | 预登记网格与选择准则 |
| `artifacts/models/paper_strengthening_v1/` | 线性对照模型 |
| `artifacts/results/paper_strengthening_v1/` | 锁、预测、dev 表、INGV probe |
| `reports/paper_strengthening_v1/` | 报告与判断 |
| `src/earthquake/paper_strengthening/` | 共享评分与线性对照 |

## 方法 A–E

| ID | 定义 |
| --- | --- |
| A | fixed → 始终 c1 |
| B | 登记 resid_s control（λ=1,σ=0.5），仅 c1/c2 |
| C | base_tau-as-expected 网格；主表仅 c1/c2 |
| D | 线性 pairwise：sigmoid(wᵀ(x2−x1))，无截距，无样本权重 |
| E | 冻结 scalar_pairwise（ckpt + τ=0.50） |

## 主比较

**唯一主比较**：`E_scalar − calibration 选定的主强对照`  
其余为次要/探索性。
