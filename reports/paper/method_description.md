# Method description (locked primary)

Catalog-assisted **S-phase candidate re-picking / refinement**, not a blind picker and not a SOTA claim.

**Deployable primary:** `fixed_rescore_UNION`  
**Name in locks:** `UNION_STEAD5_IDA5` + fixed-rescore (`primary_lock_name=fixed_rescore_UNION`)  
**Baseline:** PhaseNet-STEAD top-1 S peak.

---

## 1. How top-5 STEAD and top-5 IDA candidates are produced

Two independent PhaseNet characteristic functions are computed with SeisBench `annotate` (in_samples=3001 @ 100 Hz, default overlap/stacking, window normalization only). Component order: HDF5 ENZ → SeisBench ZNE. Label order: PSN.

| Source | Weights | SHA256 | Role |
|---|---|---|---|
| STEAD | `PhaseNet.from_pretrained('stead')` (`stead.pt.v2`) | `761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec` | waveform baseline and first candidate family |
| IDA | `artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt` epoch 14 | `b03fc6ff6c8429d877e2d69f77a31f0fd426ba285d901bdc5682bdb26b158fc4` | second candidate family only |

K=5 means **five peaks from one model**, not five models.

Peak extraction (`scipy.signal.find_peaks`) on the S channel:

- extract K=10 raw peaks, then keep top **5** per source in the union
- `distance=50` samples (0.50 s @ 100 Hz)
- `prominence=0.05`
- `min_probability=0.1`
- fallback: global argmax if no peak passes the threshold

These hyperparameters were frozen in the Stage 6 method lock **before** confirm. They were not scanned on confirm.

---

## 2. Union pool

`union_candidates_phaseC` (`src/earthquake/stage6/ranker/union_schema.py`):

1. Take STEAD top-5 and IDA top-5.
2. Merge peaks closer than **0.05 s**.
3. If both sources support a peak, **keep the STEAD sample/time** as the representative; store IDA time and disagreement. Probabilities are **never averaged**.
4. Sort by `(-both_support, -max_available_prob, stead_rank)` and truncate to `max_union=10`.

Rescore uses `p = max(stead_probability, ida_probability, 1e-6)`.

---

## 3. Fixed-rescore: inputs, formula, parameters, tie-break

Implementation: `src/earthquake/fusion/candidate_rescorer.py`.  
Locked λ and history paths: `FINAL_METHOD.LOCK.json` / Stage 6 `method_lock.json`.

**Inputs per trace**

- union candidates (sample index, source probabilities)
- frozen expected S from catalog origin time + travel-time MLP + shrunk path residual
- history availability flag (`min_history=5`)

History is a **picker-train-only** snapshot. Confirm queries are past-only and **never update** the store.

Expected S:

```
pred_tau_s = base_tau_s + shrunk_residual_s
sigma_s = clip(1.4826 * residual_s_MAD, 0.05, 1.0)
```

Shrinkage k=50. If MAD > 1.0 s, history is disabled for that path. Frozen global residual medians (picker-train):

- residual_p_median = −0.03076 s
- residual_s_median = +0.00459 s
- residual_sp_median = −0.00706 s

**Score** (S phase; λp=0 so prominence is unused at deploy time):

```
hist  = exp( -0.5 * ((c - expected_s_sample) / sigma_samples)^2 )
score = λw * log(p + ε) + λh * log(hist + ε) + λp * (prominence / max_prominence)
```

with ε=1e-8, **λw=0.5, λh=2.0, λp=0.0**. If history is unavailable, λh is forced to 0 (waveform term only).

**Tie-break:** strictly greater score wins; equal scores keep the first candidate in rescore iteration order (union pool already sorted as above).

No learned ranker, no forced-choice abstention, no confirm-time λ search.

---

## 4. Why the two sources are complementary

On full-dev, K=5, 0.5 s (candidate correctness, not the final picker):

- both correct 84.0%
- STEAD-only 2.1%
- IDA-only **3.1%**
- neither 10.7%

STEAD is the stronger single top-1 picker (F1@0.5 0.840 vs IDA 0.839) and has a much better error tail (P95 5.92 s vs 24.95 s). IDA still contributes exclusive correct peaks. Simple max-prob union **does not** harvest that complementarity (F1 0.842, P95 6.57 s). Fixed rescore does (F1 0.867, P95 1.66 s).

Oracle on the same union pool is 0.892 — a ceiling, not a deployable system.

---

## 5. Inference flow and compute

Per trace:

1. Load E,N,Z from INSTANCE HDF5 (fail on missing components; no imputation).
2. STEAD PhaseNet annotate → S peaks.
3. IDA PhaseNet annotate → S peaks.
4. Union + freeze representative times.
5. Query frozen history → expected S / σ.
6. Score candidates → one S pick.

No extra filter beyond SeisBench annotate. Output is exactly one pick per labelled eval trace in the frozen runs (miss_rate=0).

**Observed cost (historical confirm, 43,090 traces, 8× RTX 4090):**

- STEAD candidate cache ≈ 17.4 min (8-GPU shards)
- IDA candidate cache ≈ 17.3 min
- union merge ≈ 0.5 min
- fixed-rescore UNION ≈ 4.6 min CPU
- fixed-rescore STEAD (diagnostic) ≈ 4.3 min
- end-to-end wall 00:49:34Z–01:39:12Z (AUTHORIZED→CONSUMED), including history query

Micro-benchmark (Stage 3 table, n=108, 1 GPU): PhaseNet-STEAD ≈ 0.17 s/trace. Fixed rescore has **zero trainable parameters**. DKPN is not in this path.

---

## 6. Event-level split and leakage controls

Protocol: `stage6_full_chronological_event_nested` (`full_split_audit.json`).

| Split | Events | Traces | Time window (origin) |
|---|---:|---:|---|
| picker_train | 37806 | 631795 | 2005-04-16 → 2016-11-07 |
| ranker_train | 8101 | 312273 | 2016-11-07 → 2017-08-26 |
| dev | 5401 | 140428 (87293 S-labelled) | 2017-08-26 → 2019-02-13 |
| internal confirm | 2700 | 74753 (43090 S-labelled) | 2019-02-13 → 2020-01-30 |

- event-disjoint and trace-disjoint
- chronological non-overlapping windows
- IDA PhaseNet trained on picker_train only
- distance MLP + residual history fit on picker_train only
- ranker (failed, unused) trained on ranker_train only
- hyperparameters / method choice on **dev only**
- confirm: one-shot, `CONFIRM.CONSUMED`; history never updated; human S not used to place windows or choose peaks
- confirm not used for model selection, K, λ, or thresholds after consumption
