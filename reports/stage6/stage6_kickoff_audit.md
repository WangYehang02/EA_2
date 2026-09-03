# Stage 6 kickoff audit and plan

Generated before any Stage 6 training. Frozen Stage 2–5 / paper hashes recorded in
`artifacts/results/stage6/frozen_hashes_before.json`.

## 1. Real data scale (not “events ≈ traces”)

| Split (original INSTANCE chrono) | Events | Traces | Traces/event median |
|--|--:|--:|--:|
| train | **7,000** | **42,659** | 4.0 (mean 6.09, max 90) |
| val | 1,500 | 23,900 | 14.0 |
| test | 1,500 | 19,834 | 11.0 |
| **total** | 10,000 | 86,393 | — |

- Stations: **233**
- Time range: 2005-04-16 → 2009-07-15
- Sampling rate: **100 Hz** all traces
- P labels: 100%; S labels: **~54.9%** of traces
- Channel prefix: HH 70.6k, EH 12.1k, HL 2.6k, HN 1.1k
- Noise: 20,000 (train 14k / val 3k / test 3k)

**Implication:** Stage 6 nested splits are carved from the **7,000 train events** only
(original val/test already observed in Stage 3/4 and must not drive method selection).

## 2. Compute / disk

- GPUs: **8× RTX 4090 24GB** (GPU0 partially occupied; 1–7 mostly free)
- Disk: `/home` ~132 GB free; `/data` (`/mnt`) ~67 TB free
- INSTANCE data: `/mnt/yehang/PSdetec/INSTANCE`
- Estimated top-K candidate cache (K≤10, tabular feats only): **≪ 10 GB** (well under 150 GB)
- Estimated Phase A wall time (1×4090, all ~26k–38k train traces, 50 epochs, annotate eval): **~8–20 h / seed** depending on annotate frequency; 3 seeds sequential ≈ **1–3 days** on one GPU

## 3. Nested Stage 6 splits (chronological, event-disjoint)

From train pool only, **adjusted** so `internal_confirm ≈ 1,500` events
(original 70/15/10/5 would yield only 350 confirm events):

| Subset | Events | ≈ fraction | Role |
|--|--:|--:|--|
| `stage6_picker_train` | 3,850 | 55% | PhaseNet training |
| `stage6_ranker_train` | 1,050 | 15% | ranker / OOF |
| `stage6_dev` | 600 | ~8.6% | **all hyperparameter selection** |
| `stage6_internal_confirm` | 1,500 | ~21.4% | **one-shot** after method lock |

Rules: chronological by event origin; event- and trace-disjoint; station overlap audited;
confirm sealed until `method_lock_stage6.json`; no labels/SNR/preds used when building lists.

## 4. Anti-leakage plan

- No SeisBench `instance` weights as formal baseline
- No human P/S / uncertainty / label-SNR as ranker features
- History: time-ordered, past-only relative to target event
- In-domain PhaseNet → ranker: **3-fold event-grouped OOF** candidates (no in-sample candidates)
- `internal_confirm` never used for tuning
- Stage 3/4 frozen results read-only

## 5. Stop gates (summary)

- **Phase A:** +0.01 F1@0.5 or @0.1 or K=5 oracle, or −0.15 s P95 vs STEAD on **dev**, P/noise OK → else keep STEAD
- **Phase B:** union only if oracle +≥0.01 vs best single model
- **Phase C ranker:** +0.01 F1@0.5 or −0.15 s P95 vs best fixed on **dev**; else stop
- **Phase D/E:** only if prior phase passes
- **Confirm:** strong / modest / no / invalid after one locked eval

## 6. Files to create

Under `configs/stage6/`, `scripts/*stage6*`, `src/earthquake/stage6/`,
`artifacts/results|models|cache/stage6/`, `reports/stage6/`.

## 7. Execution status

Kickoff → splits → STEAD dev baseline → smoke → Phase A (background if long).
