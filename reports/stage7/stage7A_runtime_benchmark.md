# Stage 7A — Runtime Benchmark

**Weight:** PhaseNet-ethz (SeisBench annotate / predict_row e2e)  
**Manifest:** sha256-sorted Stage-6 dev first 10k (`runtime_manifest_meta.json`)  
**Warmup excluded from traces/s** (per-GPU timed window). Aggregate = sum of per-GPU sustained rates.  
**Free GPUs at start of corrected rerun:** [0, 1, 2, 3, 4, 5, 6]

## Throughput (median of 3 reps)

| Setting | GPUs | traces/s (median) | range |
|--|--:|--:|--|
| 1gpu | 1 | 6.458 | [6.409, 6.463] |
| 4gpu | 4 | 24.404 | [24.312, 24.770] |
| 8gpu | requested 8 / actual 7 | n/a | insufficient_free_gpus |
| 7gpu probe (8 unavailable) | 7 | 40.301 | single rep |

## Scaling

- 1→4: **3.779×** (near-linear)
- 4→8: **unavailable** (max free was 7; GPU7 occupied by other user)
- 4→7 proxy: **1.651×** (ideal ≈1.75×)

## Projections (@ 24.40 traces/s, 4-GPU median)

- 100k traces: **1.14 h**
- confirm 74753 traces: **0.85 h**
- full INSTANCE 1,159,249: **13.20 h**

## HDF5 I/O

- Per-trace time fraction in HDF5 read ≈ **55%** (single shared file).
- 1→4 scaling remains strong (~3.78×), so not saturated at 4 GPUs.
- 4→7 proxy **1.65×** vs ideal 1.75× → mild contention; `hdf5_io_bottleneck_likely=False` for the 4→7 test (threshold was scale<1.5).
- Peak GPU memory (SMI): ~506–545 MiB; torch allocated ~5.3 MiB (annotate path dominates host/HDF5).

Generated: 2026-08-17T07:13:03.966751+00:00
