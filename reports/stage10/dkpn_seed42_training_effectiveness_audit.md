# DKPN seed42 training-effectiveness audit

**UTC:** 2026-08-25T00:20:23.642498+00:00  
**Verdict:** `implementation_bug` (`softmax_then_log_softmax`)  
**Formal full-dev stop gate:** **not run** (checkpoint ineligible)

## Provenance

| Item | Value |
|--|--|
| DKPN commit | `cbced5a58282ff6ad2703f9c9f41f8728e334bd0` |
| config hash | `e39a6dca22aaacc7be8c4d760693625128abc4fb77d09b8ef2a1193470192b77` |
| train split hash | `f3618db5e3c7c939025351ec7e848ab3cff54c5b1d25aa1297f62f0d35ef88af` |
| last.pt sha256 | `d1fe480fef57be9e1b3365cacc02f5a3713959d1ad16a1186bab36d43591823f` |
| best.pt | **missing** (only last.pt) |
| TRAIN.DONE | 2026-08-23T23:12:35.611100+00:00 |
| env | /home/yehang/miniconda3/envs/PS/bin/python / torch 2.5.1+cu121 |

## Why mean_loss was flat (0.219 → 0.218)

**Primary:** `DKPN.forward()` returns **softmax probabilities** by default. seed42 `train_loop` did `loss = masked_soft_ce(model(x), ...)` which applies **`log_softmax` a second time**. Official DKPN (`dkpn/train.py`) uses `-y * log(p+eps)` on softmax outputs, **without** another softmax.

This is exactly the forbidden case: *softmax output cannot be treated as logits*.

Secondary contributors:

- CE is **averaged over 3001 samples**; P/S Gaussians occupy ≪1% of the window, so the scalar loss is dominated by the noise channel and can sit near ~0.22 after the first epoch.
- No LR schedule (constant Adam 1e-3).
- No `best.pt` / no full-dev checkpoint selection during training.

## Did weights actually update?

- seed42 init hash ≠ last.pt: **True**
- parameter L2 vs seed42 init: **193343.7234**
- module L2: `{'inc': 512.9710384711179, 'down_branch': 140393.6012120647, 'up_branch': 132932.94761260806, 'out': 20.84100545720774, 'other': 0.0}`
- all params trainable: **True** (N=268555; frozen=[])
- optimizer holds all trainable params: **True**

Weights **did move**; the run is not a frozen-encoder no-op. The bug is **loss/activation mismatch**, not a complete freeze.

## Batch diagnostics (8 of 128 train crops)

- input min/max/mean/std: -0.999 / 11.930 / 0.923 / 1.101
- P/S/N time-positive frac (>0.1): {'P': 0.01421661488711834, 'S': 0.014328557066619396, 'N': 0.9940254092216492}
- mask frac: {'P': 1.0, 'S': 1.0, 'N': 1.0}
- P-only S-channel masked (not treated as S-noise): **True**
- softmax out min/max/mean/std: 0.0000 / 1.0000 / 0.3333 / 0.4708
- CE (buggy log_softmax∘softmax): 0.5636
- CE (official log(p)): 0.1174
- CE (correct logits + mask): 0.1615
- per-channel CE from logits P/S/N: {'P': 0.028187472373247147, 'S': 0.4483895003795624, 'N': 0.008001711219549179}
- grad norm correct / buggy: 0.0607 / 0.0008
- NaN/Inf grads: False
- LR: 0.001 (no schedule)

## 128-train localization (S argmax ±0.2 s)

| Init | Acc |
|--|--:|
| last.pt softmax | 0.0000 |
| last.pt logits argmax | 0.0000 |
| random seed42 softmax | 0.0000 |

Trained beats random: **False**. last.pt 与随机初始化的 S-argmax 准确率都是 **0**：seed42 的 S 通道塌缩（N 占主导）。这与 buggy 梯度被压扁一致（grad-norm 0.0008 vs 正确路径 0.061，约 70×）。

Overlays: `artifacts/results/stage10/dkpn/figures`（20 张）。

**Hash 说明：** 仓库里的 `train_script_sha256` / `config_hash` 是审计时写入 `logits=True` **之后**的树。seed42 实际训练用的是 `model(x)`（softmax）再 `log_softmax`。

## Alignment

- HDF5 (3, 12000) ENZ: True; 100 Hz: True
- train∩dev events: 0; train∩confirm events: 0
- Confirm waveforms/metrics: **not read**

## Post-fix small overfit (required; not a 30-epoch retrain)

logits=True, 32 traces, 80 max epochs:

- loss 0.4951 → 0.0487
- S-acc 0.219 → 0.969
- overfit_ok: **True**; smoke_ok: **True**

## Minimal fix (do not silently retrain 30 epoch)

1. Call `model(x, logits=True)` (or `dkpn_logits`) before `masked_soft_ce`.
2. Alternatively match official: softmax then `-y*log(p+eps)` — **do not mix**.
3. Re-run small-overfit (done) + 1-GPU smoke; **only then** consider a new 30-epoch seed.

## Stop gate

Not executed on this checkpoint. `stop_gate_verdict.json`: **implementation_bug**. Extra seeds: **no**. UNION: **no**. SegPhase full train: **no**.
