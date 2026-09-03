# DKPN v2 运行时规模与 GPU 策略

## 已停止的错误策略

旧 watcher 在 **1 张** `memory.used&lt;500MiB` 的卡上就会启动 30 epoch。该策略已废除。

## 新定义

| 量 | 含义 |
|--|--|
| virtual epoch | 每条训练 trace **恰好一个** online crop（不再 3× 展开） |
| crop cycle | 3 个 virtual epoch：P-centered → S-centered → background |
| optimizer_steps | Adam step 数 |
| windows_seen | 实际见过的窗数 |

P–S&gt;30 s 的 sample_weight=2，保证长间隔迹上 P/S crop 都够。  
P-only：P-centered 与 partial-label background 按 hash 轮换。  
Noise：固定 `noise_ratio`，且 **只有真实 noise HDF5**，不用 event 未知区冒充。

## GPU 空闲（同时满足）

- `memory.used &lt; 500 MiB`
- utilization ≈ 0（≤2%）
- 无其他用户 compute 进程  
**不得抢占、不得杀别人的任务。**

`MIN_GPUS=4`，`PREFERRED_GPUS=8`。  
少于 4 张：**只等待**，禁止 full / corrected pilot。  
1 张卡：**只允许** 50-step 吞吐 bench（单卡），禁止 10 天训练。

`0% util` 但显存仍占 10GB+、且有 python compute 进程 = **非空闲**。

## 吞吐 benchmark

固定 **50 step、恰好 1 张真正空闲 GPU**（不跑 4/8 卡）：

- windows/s、每 GPU windows/s（此处即单卡）
- GPU util、显存
- 按 1 卡吞吐外推 virtual epoch / crop cycle ETA（4/8 卡不测）

完成后 **退出，不接 full 训练**。  
结果：`artifacts/results/stage10/dkpn_v2_throughput.json`。

本机当前若 0 张合格空闲卡，JSON 会标记 `skipped_reason`，ETA **不得**当成已测 4090 吞吐。

旧 CPU 4-step 记录（~2.8 windows/s）**不能**用来报 4090 ETA。

在线采样规模（约）：758k windows / virtual epoch（P+S 377087 + P-only + noise，**不再 ×3**）。  
相对旧 3× 展开，每个 virtual epoch 工作量约降至 1/2–1/3（视 P-only 双窗而定）。

未测 GPU 时不填写虚假 hours。测完后再回填下表：

| 配置 | windows/s | /GPU | virtual epoch ETA | crop cycle ETA |
|--|--:|--:|--|--|
| 1 GPU | （待测） | | | |
| 4 GPU | （待测） | | | |
| 8 GPU | （待测） | | | |

## 两段训练

1. **Corrected pilot**（≥4 张真正空闲 4090；推荐 8）  
   seed42 随机初始化；3 virtual epochs = 1 crop cycle；eval 用固定 event-balanced Stage6 **dev** 子集；confirm 不读。  
   Gate：相对首 500 step 的 train loss 下降；P/S 不塌；dev S F1@0.5（官方 `extract_picks`, 0.5s）优于 init 且 epoch3≥epoch1；grad/LR 正常；无 NaN；P/S/background 与 partial-label 都出现。  
   失败：`PILOT.FAILED` 停止。通过：`PILOT.PASSED` 再最多 30 virtual epochs，patience=8，每 3 epoch 至少一次较大 dev，checkpoint 只看 dev，然后 full-dev stop gate。

2. 旧 buggy `train_seed42` **保留不动**。

## 启动命令

```bash
bash scripts/launch_dkpn_v2_train.sh status
bash scripts/wait_and_launch_dkpn_v2.sh throughput   # 有空闲卡先 bench，然后退出
bash scripts/wait_and_launch_dkpn_v2.sh pilot         # 仅 ≥4 空闲卡时 3-epoch pilot
```
