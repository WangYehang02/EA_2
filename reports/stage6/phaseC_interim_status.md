# Stage 6 Phase C — 通宵挂起状态

**时间：** 2026-08-16 ~22:06  
**Confirm：** 仍封印  
**method_lock / multi-station：** 未启动、通宵脚本也不会自动启

## 已完成

1. Schema：`UNION_STEAD5_IDA5`（`candidate_schema.json`）
2. History：picker_train-only，shrinkage_k=50；train 覆盖 ~99.1%，dev ~91.0%
3. 非学习基线全量 87,293：最强 **`fixed_rescore_UNION` F1@0.5=0.8668**（oracle 0.8917）
4. Dev 特征：`dev_union` / `dev_features_R1|R2`
5. ranker_train 候选缓存：**STEAD+ID-A top10 均 206413/206413 完成**
6. Ranker 代码 + overnight 编排脚本就绪；单元测试已通过

## 通宵进行中

| 任务 | 状态 |
|--|--|
| `build_stage6_ranker_dataset.py` | 运行中（PID 见 OVERNIGHT_README） |
| `run_phaseC_overnight.sh` | 等待 dataset → R1→R2→(R3)→seeds→eval→bootstrap→verdict |

状态文件：`artifacts/results/stage6/phaseC/overnight_status.json`  
完成标记：`artifacts/results/stage6/phaseC/OVERNIGHT.DONE`

## 早上先看

```bash
cat artifacts/results/stage6/phaseC/overnight_status.json
cat artifacts/results/stage6/phaseC/phaseC_final_verdict.json
```

对照门：相对 `fixed_rescore_UNION`；即便 `strong_pass` 也需你确认后再启封 confirm / 开 multi-station。
