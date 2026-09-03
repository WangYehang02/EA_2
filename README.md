# Earthquake: PhaseNet + Historical Path Prior Fusion

## Current stage conclusion (1.5)

> 历史路径先验在已知当前事件震源位置和发震时刻的 **catalog-assisted** 场景中，能够提供明显优于当前外部 PhaseNet 的粗粒度到时估计。但当前 PhaseNet 推理链路或域适配仍存在问题，且历史先验尚未达到精确拾取要求，因此暂不能证明最终融合模型有效。

注意：catalog-assisted 历史结果 **不是** blind phase picking，因为它使用了当前事件震源区域、深度、origin time 以及台站对应关系。

### Diagnosed root cause of ~20–27 s MAE

Previous STEAD MAE was **not** primarily domain shift. Custom sliding-window evaluation treated prediction sample 0 as waveform sample 0, while official `annotate()` shifts `starttime` by ~+2.5 s and returns ~11500 samples on a 12000-sample input. After **UTC remapping**, custom wrapper ≡ official annotate (0 sample difference); STEAD median AE falls to ~0.06 s on the 512-trace diagnostic set. Residual domain gap remains (INSTANCE diagnostic weight is better but labeled `diagnostic_only_data_leakage`).

### Debug finetune status

- Soft labels must follow SeisBench STEAD `labels=PSN` (not default class `NPS`).
- Keep BatchNorm in `eval()` while training weights; updating BN running stats on 3001-sample crops breaks full-trace `annotate()`.
- With those fixes, train/val loss decreases, but 5-epoch debug finetune did **not** beat UTC-aligned STEAD on annotate F1; `checkpoints/best.pt` remains the pretrained STEAD init.
- Full finetune config prepared only: `configs/phasenet_finetune_full.yaml` (not auto-started).
- Working PhaseNet for fusion remains **UTC-aligned STEAD annotate**.

## Environment

```bash
conda activate PS
cd ~/yehang/Earthquake
pip install -e .
export INSTANCE_ROOT=/mnt/yehang/PSdetec/INSTANCE
```

## Stage 1.5 diagnostic pipeline

```bash
# 1) Align custom wrapper vs official annotate (must agree within 1 sample)
python scripts/diagnose_phasenet_alignment.py --n-traces 512 --weight stead

# 2) Compare pretrained weights (instance = diagnostic only)
python scripts/compare_pretrained_weights.py --weights stead ethz scedc instance

# 3) Fixed eval set (>=10k events + >=2k noise)
python scripts/build_fixed_eval_set.py --n-events-traces 10000 --n-noise 2000

# 4) Debug finetune (only if alignment + instance diagnostic pass)
python scripts/finetune_phasenet.py --config configs/phasenet_finetune_debug.yaml
python scripts/evaluate_finetuned_phasenet.py

# 5) Re-run fusion / shuffle / distance-bin baselines
python scripts/reeval_fusion_stage15.py --max-traces 2000

pytest -q
```

Full finetune config is prepared but not auto-started: `configs/phasenet_finetune_full.yaml`.

## Pick matching protocol (Stage 2)

Single-peak protocol (one prediction per trace):

- **TP@w**: label present, prediction present, `|pred−true| ≤ w`
- **FP@w**: prediction present and not TP (wrong peak beyond tolerance, or prediction without label)
- **FN@w**: label present and not TP (missed pick, or wrong peak beyond tolerance)
- A prediction outside tolerance is **never** TP and is not double-counted as two FPs.

Timing metrics (names matter):

- **e2e MAE / e2e P95**: absolute error on all labeled traces that also have a prediction (includes catastrophic wrong peaks; this is why median≪MAE/P95)
- **matched-timing MAE / P95**: absolute error only on TP@tolerance (usually 0.5s)

Multi-peak ceiling: success if **any** of K PhaseNet candidates falls within tolerance of the label.

## Stage 2 pipeline

```bash
python scripts/audit_pick_metrics.py --config configs/fusion_fixed.yaml --device cuda
python scripts/fit_travel_time_baseline.py --config configs/fusion_fixed.yaml
python scripts/build_residual_history.py --config configs/fusion_fixed.yaml --protocol frozen
python scripts/evaluate_candidate_rescoring.py --config configs/fusion_fixed.yaml
python scripts/bootstrap_significance.py --config configs/fusion_fixed.yaml --n-bootstrap 2000
python scripts/analyze_hard_cases.py --config configs/fusion_fixed.yaml
pytest -q
```

Catalog-assisted residual re-scoring is **not** blind picking. Blind-S uses PhaseNet P candidates + historical Δ(S−P) only (no current origin time).

## Stage 3: learned scalar gate (S-only catalog-assisted repicking)

Model role: **catalog-assisted S-phase repicking/refinement** — re-ranks existing PhaseNet S candidates with an interpretable `gate_s` (1=PhaseNet, 0=history). Not a standalone continuous detector. P stays PhaseNet. Noise bypasses the gate.

```bash
python scripts/audit_stage3_splits.py
python scripts/build_gate_dataset.py --config configs/gate_debug.yaml
python scripts/compute_candidate_oracle.py --config configs/gate_debug.yaml
python scripts/train_learned_gate.py --config configs/gate_debug.yaml --seed 42
python scripts/evaluate_learned_gate.py --config configs/gate_debug.yaml --seed 42
python scripts/bootstrap_gate.py --config configs/gate_debug.yaml --n-bootstrap 2000
python scripts/analyze_gate_behavior.py --config configs/gate_debug.yaml
python scripts/plot_gate_cases.py --config configs/gate_debug.yaml
pytest -q
```

Formal catalog configs: `configs/gate_catalog.yaml` (3 seeds 42/123/2026). Blind-S: `configs/gate_blind_s.yaml` (trained/evaluated separately).
Use `artifacts/diagnostics/fixed_eval_test_only_*` for Stage-3 test (original fixed list mixed val+test; not overwritten).
