# Stage 6 Full Transition Audit

## Pilot final status

| Field | Value |
|--|--|
| Label | **`stage6_phaseA_pilot_10k_index`** (permanent) |
| Status | **`pilot_cancelled_after_smoke`** |
| Reason | Still in preload (~61%); prior smoke already covered crop-train + annotate + BN-eval |
| Termination | SIGTERM (not kill -9) |
| First full `stage6_dev` annotate from this pilot | **None** (training loop never started) |
| Additional seeds | **not launched** |

See `artifacts/results/stage6/pilot_final_status.json`.

---

## Why 10k existed / fix

- Root cause: `scripts/build_index.py --max-events` **defaulted to 10000**.
- Fix: default is now **`None` (keep all)**; writing fewer than raw unique events without an explicit limit **raises**.
- Default output dir is **`artifacts/index_full/`** (legacy `artifacts/index/` untouched).
- Unit test: `tests/test_stage6_full_transition.py`.

---

## Full event / noise scale

| | Events / rows | Traces |
|--|--:|--:|
| Events metadata | 54,008 | 1,159,249 |
| Noise metadata | 132,288 | 132,288 |
| Legacy capped index (unchanged) | 10,000 | 86,393 |
| Legacy noise index | 20,000 | from old `max_noise=min(2*max_events,20000)` |

Manifest: `artifacts/index_full/manifest.json` (includes CSV SHA256, truncated=false).

---

## Full Stage6 nested splits (`splits_full/`)

| Subset | Events | Traces | S / P+S labelled | Origin time |
|--|--:|--:|--:|--|
| picker_train | 37,806 | 631,795 | 377,087 | 2005-04-16 → 2016-11-07 |
| ranker_train | 8,101 | 312,273 | 206,413 | 2016-11-07 → 2017-08-26 |
| dev | 5,401 | 140,428 | 87,293 | 2017-08-26 → 2019-02-13 |
| **internal_confirm** | **2,700** | **74,753** | 43,090 | **2019-02-13 → 2020-01-30** |

- Chronological, event/trace disjoint: **yes**
- Confirm all never-used in Stage1–5: **yes** (contaminated_excluded=0)
- `CONFIRM_SEALED` written; method_lock required for confirm waveform/label/prediction
- Pilot `splits/` **not replaced**

History protocol (primary confirm): frozen snapshot from **picker_train only** (distance MLP, TemporalHistoryStore, PhaseNet). Ranker fits on ranker_train; dev for selection only; confirm never fits. Rolling history = secondary only.

---

## Missing S labels

- ~38.4% traces lack S; **must not treat as noise**.
- Dominant pattern: P-only / no human S; missing rate rises sharply with distance.
- Phase A v1: supervise **P∩S only** + real noise; **P-only excluded** until masked loss exists.
- picker_train usable **P+S traces: 377,087** (events 37,806); planned noise ~20% ≈ 75k of 132,288.

---

## Loader / DDP

| GPUs | Approx global traces/s |
|--|--:|
| 1 | ~90 |
| 4 | ~116 |
| 8 | ~135 |

HDF5 single-file I/O plateau — sharded `/data` cache only if needed later. Lazy per-worker handles; no `/home` waveform dump for full training.

Smokes: **1 GPU OK**, **8 GPU DDP OK** (`find_unused_parameters=True` with BN frozen).

---

## Metric naming correction

`match_picks` P95 = **detected / non-missing** AE (wrong peaks included; NaN misses excluded).  
**Do not call it complete end-to-end P95.** Prefer `detected_ae_p95` + `miss_rate` + `wrong_peak_rate`. Old artifacts unchanged; Stage6 uses new names + warning string.

---

## Phase A launch gate

All required gates passed (split/noise/S-missing/lazy1000/1gpu/8gpu/PSN/STEAD weights/frozen hashes/pytest).

Full ID-A seed42 8×4090 DDP started only after this audit (see report footer / progress JSON).


---

## Phase A full launch

- **Started:** yes (ID-A, seed42, 8×RTX4090 DDP)
- **torchrun PID:** 1590248
- **Log:** `artifacts/results/stage6/logs/phasenet_ida_full_seed42.log`
- **Out:** `artifacts/models/stage6/phasenet_ida_full_seed42/`
- **Train meta:** 377,087 P+S traces + 75,417 noise; P-only=false
- **Pretrained annotate (1024 dev subset):** S F1@0.5≈0.624, detected_ae_p95≈6.05 s (naming: not complete e2e)
- **ETA:** ~47 min/epoch × ≤50 ep ≈ **~39 h** (+ annotate); early stop patience 8
- **pytest:** 64 passed
- **Stage2–5 hashes:** unchanged
- Ranker / multi-station: **blocked** until Phase A stop-gate
