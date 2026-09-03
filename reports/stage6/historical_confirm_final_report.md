# 2026-08-17 历史 confirm 完整性审计与冻结裁决

**CONFIRM.BLOCKED 保持。** 本次未重新运行 confirm inference，未重新生成 predictions。

完整性 marker：`artifacts/results/stage6/final_confirm/HISTORICAL_CONFIRM.INTEGRITY.json`  
裁决：`artifacts/results/stage6/final_confirm/HISTORICAL_CONFIRM.VERDICT.json`

未修改：`CONFIRM.CONSUMED`、`FINAL_METHOD.LOCK`、`CONFIRM_ANALYSIS.LOCK`、`CONFIRM_READINESS`、`PRECONFIRM_AUDIT.ADDENDUM`。

---

## 1. 完整性

**判决：`HISTORICAL_CONFIRM.INTEGRITY.PASSED`**

完整性 marker 写完之后，才读取冻结的 `confirm_method_metrics.json` 与 `confirm_bootstrap.json`。

### 1.1 时间顺序（必须：method/protocol lock 早于运行）

| 事件 | UTC | SHA256 |
|---|---|---|
| Stage6 method lock | 2026-08-17T00:48:40.061985Z（mtime 00:48:41） | `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303` |
| confirm protocol | mtime 2026-08-17T00:48:42Z | `b68261e63894eb94b9e21bb19737cdd6274dd8e44c099df3f3339bd8b98ca863` |
| AUTHORIZED | 2026-08-17T00:49:03Z | — |
| RUNNING | 2026-08-17T00:49:34Z | PID 4029591 |
| predictions / metrics 落盘 | 2026-08-17T01:34:59Z | 见下 |
| bootstrap + CONSUMED | 2026-08-17T01:39:12.235375Z | 见下 |

primary = `fixed_rescore_UNION`；baseline = `STEAD_top1`；`locked_before_confirm=true`；`confirm_seen_before_lock=false`。

IDA checkpoint：`best.pt` epoch 14，SHA `b03fc6ff6c8429d877e2d69f77a31f0fd426ba285d901bdc5682bdb26b158fc4`（cache cfg 与 method lock 一致）。  
STEAD：单一 `PhaseNet.from_pretrained('stead')`，SHA `761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec`。  
K=5/5，λw=0.5 λh=2.0 λp=0.0，peak min_probability=0.1。未扫描多个 checkpoint。未在 confirm 上选择 threshold / K / model。

### 1.2 运行完整性

| 检查 | 结果 |
|---|---|
| expected / processed traces | 43090 / 43090 |
| expected / processed events | 2669 / 2669 |
| stations / networks（S 标注评测总体） | 460 / 11 |
| STEAD/IDA cache、union、history 覆盖 | 各 43090 unique traces，差集 0 |
| 行序与 manifest 一致 | 是 |
| NaN（全部预测列 + npy） | 0 |
| Inf | 0 |
| cache shard fails | 0（16/16 DONE） |
| 日志 Traceback / exception | 无 |
| 静默 skip / 损坏输入跳过 | 0 |
| 完整退出 CONSUMED | 是 |
| predictions/metrics/bootstrap SHA 与 CONSUMED 一致 | 是 |
| 同一次运行顺序完成 | 是 |

全量 confirm 划分 2700 events / 74753 traces；评测总体是全部 S 标注子集 2669 / 43090。31 个无 S 标注事件是预注册总体过滤，不是对评测 traces 的静默 skip。

旧 runner 代码含 NaN skip 路径。**此次实际 skipped=0、NaN=0**，按规则不因此判失败。

Oracle 在同一次 run 中计算，协议角色为 diagnostic_only；primary 身份在 AUTHORIZED 之前已由 method lock 确定，oracle 未参与 primary 预测或选型。

输出 SHA（与 CONSUMED 一致）：

- predictions：`0ce3e3b9e04401e16a4d9bb912796096b2adbfd292aeab7d53e55b34e22bbf20`
- metrics：`48a7b3a0d4cc28fe9d08597d2074a76d39958de84081e22f34ea31dac51163a7`
- bootstrap：`92a9d715d7e31aac30c2e17818d1738afafe53656ff9cdbb7f7baa61d2450476`

---

## 2. 冻结指标与 2026-08-17 gate（完整性 PASSED 之后）

预注册：

```
strong_confirmed =
  dF1_05 >= 0.01
  and CI_lo > 0
  and dF1_01 >= -0.003
  and dP95 <= 0
  and no selective recall collapse
```

bootstrap：n=5000，seed=20260817，unit=event。  
执行顺序：读取已有 primary/baseline 汇总 → 读取已有 paired event bootstrap → **立即写本冻结 verdict** → 之后才允许预注册 subgroup。UNION 协议预注册 subgroup 列表为空；未导入 DKPN grouping；未新增分组；未改门槛。

### 2.1 F1

| 方法 | F1@0.1 | F1@0.5 |
|---|---|---|
| `fixed_rescore_UNION` | 0.503458 | 0.837317 |
| `STEAD_top1` | 0.485171 | 0.817568 |

### 2.2 Δ 与 CI（UNION − STEAD_top1）

- ΔF1@0.5 = **0.019749**
- 95% CI = **[0.017345, 0.022289]**
- ΔF1@0.1 = **0.018287**
- P95：UNION 2.4655 s，STEAD 6.1 s，ΔP95 = **−3.6345 s**
- ΔRecall@0.5 = **+0.019749**（双方 miss_rate=0；无选择性召回崩溃）

Oracle UNION F1@0.5 = 0.867626，仅诊断，不进入 gate。

### 2.3 strong_confirmed 逐条

| 条件 | 值 | 通过 |
|---|---|---|
| dF1_05 ≥ 0.01 | 0.019749 | 是 |
| CI_lo > 0 | 0.017345 | 是 |
| dF1_01 ≥ −0.003 | 0.018287 | 是 |
| dP95 ≤ 0 | −3.6345 | 是 |
| no selective recall collapse（冻结 runner：ΔRecall@0.5 < −0.02 视为崩溃） | ΔRecall@0.5 = +0.019749 | 是 |

**最终 verdict：`strong_confirmed`**

与 2026-08-17 历史脚本写入的 `confirm_final_verdict.json` 状态字符串一致；本文件为按协议原文对冻结 metrics/bootstrap 的独立重算，未重跑模型。

---

## 3. Confirm 后方法是否改变

否。Stage6 `method_lock.json` SHA 仍为消耗时的 `a02dc28e…`。Stage10 `FINAL_METHOD.LOCK` 仍锁定 primary=`UNION_STEAD5_IDA5` fixed-rescore、baseline=`STEAD_top1`、同一 IDA/STEAD SHA、同一 K/λ。`CONFIRM.CONSUMED` 仍要求不得用 confirm 做模型选择。

---

## 4. 最终汇报清单

- 完整性是否通过：**是**（`HISTORICAL_CONFIRM.INTEGRITY.PASSED`）
- 是否发生 skip/NaN：**否**（skipped=0，NaN=0，Inf=0）
- confirm 样本覆盖是否完整：**是**（S 标注评测总体 43090 traces / 2669 events 全覆盖）
- UNION F1@0.1 / F1@0.5：**0.503458 / 0.837317**
- STEAD F1@0.1 / F1@0.5：**0.485171 / 0.817568**
- ΔF1@0.5 与 95% CI：**0.019749，[0.017345, 0.022289]**
- P95 变化：**−3.6345 s**（改善）
- selective recall collapse：**无**
- strong_confirmed 每个条件：**全部通过**
- 最终 verdict：**`strong_confirmed`**
- confirm 后方法是否改变：**否**
- 是否运行了任何新推理：**false**
