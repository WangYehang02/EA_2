# Stage 6 Full-Dataset Provenance Audit

**Status:** read-only audit complete. Current Phase A seed42 is **`stage6_phaseA_pilot_10k_index`**, not full INSTANCE training.  
**Confirm:** remains sealed. **Additional seeds / ranker / paper edits:** blocked.

Artifacts:

- `artifacts/results/stage6/full_dataset_counts.json`
- `artifacts/results/stage6/prior_event_usage_registry.csv`
- `artifacts/results/stage6/proposed_full_split.json`
- `artifacts/results/stage6/pilot_status.json`
- `artifacts/results/stage6/never_used_event_ids.txt` (44,008 IDs)

Frozen Stage 2–5 / paper hashes: **unchanged** (`frozen_hashes_during_provenance_audit.json`).

---

## 1. Why only 10,000 events?

**Root cause (definitive):**

```text
scripts/build_index.py
  --max-events  DEFAULT = 10000
```

Behavior:

1. Stream `metadata_Instance_events_v3.csv`
2. Collect unique `source_id` → min `source_origin_time`
3. Sort ascending by time
4. **`keep_ids = first 10,000 events`**
5. Write `artifacts/index/events.parquet` + `artifacts/splits/{train,val,test}_events.txt`

Related caps (same 10k mindset, not independent full-data pipelines):

| Location | Limit |
|--|--|
| `scripts/build_history.py --max-events` | 10000 |
| `scripts/analyze_history_coverage.py` | 10000 |
| `configs/fusion_fixed.yaml max_events` | 10000 |
| `scripts/build_fixed_eval_set.py --n-events-traces` | 10000 **traces** |
| `configs/phasenet_finetune_debug.yaml n_train_traces` | 10000 |
| noise in `build_index.py` | `min(2*max_events, 20000)=20000` |

There is **no** evidence of a random 10k sample from the full 54k; it is a **chronological head** filter.

**Concrete file providing the working 10k universe:**

```text
artifacts/index/events.parquet
```

built from:

```text
data/INSTANCE/events/metadata_Instance_events_v3.csv
(= /mnt/yehang/PSdetec/INSTANCE/events/metadata_Instance_events_v3.csv)
```

---

## 2. Raw metadata (streamed)

| Quantity | Value |
|--|--:|
| CSV data rows | **1,159,249** |
| Lines incl. header | 1,159,250 |
| Unique `source_id` | **54,008** |
| Unique `trace_name` | **1,159,249** |
| Stations (net.sta) | **631** |
| P-labelled traces | **1,159,249** (100%) |
| S-labelled traces | **713,883** (~61.6%) |
| Origin time range | **2005-04-16 → 2020-01-30** |
| Sampling rate | primarily 100 Hz (`1/trace_dt_s`) |
| Noise rows | see `full_dataset_counts.json` |

---

## 3. Layer funnel (events / traces)

| Layer | Events | Traces | Why reduced |
|--|--:|--:|--|
| Raw metadata | 54,008 | 1,159,249 | Full INSTANCE |
| `artifacts/index` | **10,000** | **86,393** | `--max-events=10000` earliest-by-time |
| Stage1 splits | 7k/1.5k/1.5k | same 86,393 | Split only inside capped index |
| Stage3 test-only list | 1,482 | 10,000 | Trace subsample, `max_per_event=8` |
| Stage6 nested (current) | 3,850 / 1,050 / 600 / 1,500 | 18,859 / 4,812 / 5,368 / 13,620 | Nested **only from Stage1 train 7k** |

**Never used in Stage1–5 index universe:** **44,008 events** (~1.07M traces).

---

## 4. Prior event usage registry

`prior_event_usage_registry.csv` covers all **10,000** indexed events with:

- Stage1 split
- locations across Stage1–6 artifacts
- roles (train/val/test/history/gate/stage6_*)
- `metrics_observed`
- `human_labels_used_as_train_targets`
- `in_stage6_internal_confirm`

---

## 5. Current `internal_confirm` (1,500) contamination

**All 1,500 confirm events ⊆ Stage1 `train_events.txt`.**

| Old component | Confirm involvement |
|--|--|
| PhaseNet debug/last-layer finetune indexes | **608 / 1500** events overlap |
| Travel-time / distance baseline (`travel_time_baseline.pkl`) | Fit on Stage1 **train** pool → **1500/1500** in fit pool |
| History store / features | Confirm rows present; events are late **train** history sources |
| Learned gate / gate baselines `split=train` | **1500/1500** events, **12,501** traces |
| Stage3 test-only eval | **0** overlap |
| Stage1 val/test | **0** overlap |

### Restrictions (mandatory)

1. Confirm **may** evaluate a **new** PhaseNet that never trained on these labels.
2. Confirm **must not** be treated as clean for **old** `catalog_rescore` that uses Stage2 distance-MLP / history / gate artifacts.
3. Any full Stage6 catalog path must **refit** distance MLP, history, and ranker **excluding** confirm labels.
4. Prefer a **never-used** confirm pool (≥2000 available) for independent confirmation.

---

## 6. Proposed full Stage6 plan (NOT applied; current splits kept)

Pools:

| Pool | Role |
|--|--|
| **A** | Full development from never-used events (excl. proposed confirm) |
| **B** | Previously-unused confirm (latest-by-time, ≥2000) |
| **C** | Old 10k pilot pool (current nested splits) |

Recommended never-used chronological split (from `proposed_full_split.json`):

| Subset | Events | Traces | S-labelled traces (approx) | Time window |
|--|--:|--:|--:|--|
| picker_train | 33,257 | 762,640 | 474,769 | 2009-07 → 2017-03 |
| ranker_train | 5,251 | 162,082 | 103,921 | 2017-03 → 2018-04 |
| dev | 3,500 | 92,863 | 55,954 | 2018-04 → 2019-05 |
| **confirm (seal)** | **2,000** | **55,271** | **31,805** | 2019-05 → 2020-01 |

**Rebuild Stage6 split: YES** before claiming full in-domain training.  
Do **not** replace pilot splits until pilot finishes pipeline sanity.

Training rule for full run: **event-balanced sampler over all available traces**; forbidding “few traces/event” while calling it full training.

---

## 7. Full ~1.16M-trace resource estimate

| Item | Estimate / rule |
|--|--|
| `/home` free | ~131 GB — **do not** copy all waveforms here |
| `/data` free | ~67 TB — preferred cache root |
| Full waveform float32 cache (3×120s@100Hz) | **~167 GB** if naively copied — **avoid on /home** |
| Top-K + local feature cache | ≪ 150 GB if only candidates (~few GB–tens GB) |
| I/O | HDF5 lazy from `INSTANCE_ROOT`; resumeable cache on `/data/.../stage6_full` |
| GPUs | 8× RTX 4090 24GB |
| DDP | 8-GPU DDP; BN **eval / frozen running stats**; crop train + full `annotate` val for ckpt |
| Wall-clock | Event-balanced coverage: order **1–several days** for 30–50 ep on 8×4090; true all-traces-every-epoch much longer — use epoch subsample of traces/event with high unique coverage + periodic full-dev annotate |

---

## 8. Metric scope (Stage6 STEAD `stage6_dev`)

| Quantity | Value |
|--|--:|
| Total traces | **5,368** |
| Events | **600** |
| **S-labelled traces** | **2,593** |
| S F1@0.5 / @0.1 | 0.562 / 0.383 |
| e2e P95 | 1.334 s |
| miss_rate | 0.0 (on labelled) |

**Explicit:**

- S F1 is over traces with human S labels (`n_eval = has_true`).
- **e2e P95 = `all_labeled_with_prediction`** — **NaN misses excluded**; miss_rate is separate; P95 ≠ full end-to-end including misses.
- This STEAD baseline is **annotate top-1**, not “identity among K”. If a method’s fallback is not bit-identical to the claimed base picker per trace, **rename it**.

---

## 9. Pilot status

```json
{
  "current_run_is_full_instance": false,
  "current_run_role": "pipeline_pilot",
  "current_run_label": "stage6_phaseA_pilot_10k_index",
  "launch_additional_seeds": false,
  "confirm_remains_sealed": true
}
```

After seed42 pilot: only pipeline checks on **current** dev (loss / annotate / BN). **No** full-domain claim.

---

## 10. Verdicts

| Question | Answer |
|--|--|
| Why 10k? | `build_index.py --max-events=10000` chronological head |
| Full raw size? | 54,008 events / 1,159,249 traces |
| New never-used confirm available? | **44,008** never-used; proposed confirm **2,000** |
| Current confirm contaminated by old train artifacts? | **YES** (train pool; finetune∩608; gate train; distance MLP; history) |
| Rebuild Stage6 split? | **YES** for full training; keep pilot until sanity done |
| Full train resources? | Lazy HDF5 + `/data` cache; 8×4090 DDP; no full wave copy to `/home` |
