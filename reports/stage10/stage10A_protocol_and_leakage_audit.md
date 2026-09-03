# Stage 10A — Protocol & Leakage Audit

**Created (UTC):** 2026-08-18T04:58:03.590792+00:00  
**Project root:** `/home/yehang/yehang/Earthquake`  
**sota_claim_allowed:** **false**  
**exact_lftnet_protocol_reproducible:** **false**

## Task split

| Model | Inputs | Claim type |
|--|--|--|
| **Ours-Blind** | waveform only | fair vs PhaseNet/DKPN/SegPhase/LFTNet-class blind pickers |
| **Ours-Path** | waveform + distance/history/shrinkage | *catalog-assisted S-phase candidate re-picking/refinement* (not blind) |
| Frozen main | `fixed_rescore_UNION` | Stage 6 confirm CONSUMED — do not retune on confirm |

## 1–2. Splits (from `full_split_audit.json`)

| Split | Events | Traces |
|--|--:|--:|
| picker_train | 37806 | 631795 |
| ranker_train | 8101 | 312273 |
| dev | 5401 | 140428 |
| internal_confirm | 2700 | 74753 |

- `event_disjoint=True`, `trace_disjoint=True`
- Pairwise event overlaps: `{'picker_train∩ranker_train': 0, 'picker_train∩dev': 0, 'picker_train∩confirm': 0, 'ranker_train∩dev': 0, 'ranker_train∩confirm': 0, 'dev∩confirm': 0, 'event_disjoint_all_pairs': True}`

Supervision plan (full training manifest):

- P+S both traces in picker_train: **377087**
- picker_train all traces: **631795**
- Phase A historically **excluded P-only** (not treated as noise). Stage 10 allows **masked P-only** if tested.
- Full noise available: **132288** (is 132288: **True**)

## 3–6. Baseline provenance

### DKPN
- realpath: `/home/yehang/EARTHQUAKE/baseline/DKPN`
- HEAD: `cbced5a58282ff6ad2703f9c9f41f8728e334bd0`
- LICENSE: MIT; version 0.4.12; SeisBench-oriented; `in_channels=5`, component **ZNE**
- Official paper weights: **INSTANCE-trained** → **`diagnostic_only`**, **not** for main Table / formal SOTA
- Clean DKPN: **random init** (or non-INSTANCE external) on Stage-6 `picker_train` only

### SegPhase
- realpath: `/home/yehang/EARTHQUAKE/baseline/SegPhase`
- HEAD: `27e7e5d9ce02fbc2aea5ec646a569a381a5e7b6e`
- Japan/JMA pretrained (100 Hz UD,NS,EW); Stage 9 confirm F1@0.5≈0.654 — architecture not exhausted
- Stage 10C: in-domain INSTANCE retrain (later)

### LFTNet
- Local drop incomplete (`/home/yehang/yehang/Earthquake/baseline/Tianjiyu1-LFTNet-79994f6`): no checkpoint, no 10k list, no metric scripts
- **Cannot** claim “exceeded LFTNet / INSTANCE SOTA”
- Allowed phrasing: *best among evaluated reproducible methods under our event-disjoint INSTANCE protocol*

## 7. INSTANCE IO

- realpath: `/data/mnt_data/yehang/PSdetec/INSTANCE`
- 100 Hz; HDF5 waveform order **ENZ** via `InstanceHDF5Reader`
- Map to DKPN/SegPhase as required (ZNE / UD-NS-EW) with UTC remap tests

## 8. Noise

- Full noise traces: **132288** (target 132288: **yes**)
- S-labelled event traces (all INSTANCE): **713883**; P-only: **445366**

## 9–10. LFTNet protocol

`exact_lftnet_protocol_reproducible = false` — missing official list/code/weights/threshold definitions. No second-hand table guessing.

## Verified frozen metrics (re-read from JSON)

### Confirm (S-labelled n=43090)
| Method | F1@0.5 | F1@0.1 | P95 |
|--|--:|--:|--:|
| STEAD top-1 | 0.8176 | 0.4852 | 6.100 |
| fixed_rescore_UNION | 0.8373 | 0.5035 | 2.465 |
| UNION oracle | 0.8676 | 0.6017 | 1.780 |

### Dev
- fixed_rescore_UNION F1@0.5 ≈ **0.8667**
- UNION oracle F1@0.5 ≈ **0.8917** (phaseB UNION_K10=0.8917)
- Old ranker failed +0.01 gate (≈ −0.034 vs fixed after correction)

## GPU at audit

```
0, 3017, 99
1, 3034, 99
2, 15392, 97
3, 15422, 100
4, 15392, 100
5, 15406, 100
6, 14214, 99
7, 14324, 98
```

Free GPUs (<500 MiB): **none** — do not preempt others.

## Confirm policy

- `CONFIRM.CONSUMED` present; Stage 10 **must not** tune on confirm
- Any confirm numbers are **post-hoc** only

## Frozen hash check

method_lock match: **True** (`a02dc28e…`)

## HDF5 realpaths (lazy, not copied)
- events: `/data/mnt_data/yehang/PSdetec/INSTANCE/events/Instance_events_counts.hdf5`
- noise: `/data/mnt_data/yehang/PSdetec/INSTANCE/noise/Instance_noise.hdf5`
- cache: `/data/mnt_data/yehang/PSdetec/Earthquake_stage10`
