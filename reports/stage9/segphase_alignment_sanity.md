# Stage 9 — SegPhase Alignment Sanity

**alignment_gate:** `PASS`  
**checkpoint:** `/home/yehang/EARTHQUAKE/baseline/SegPhase/model/model_100Hz.pth`  
**SHA256:** `ec5d48cdc575b9ffc638a9039669f08de6b2349acd5d9a36748e4c89606d0727`  
**checkpoint_not_random:** `True`

## Mapping

- Input shape: `(B,3,3000)` @ 100 Hz
- INSTANCE ENZ → **UD,NS,EW = Z,N,E**
- Normalize: per-channel z-score
- Peaks: official `find_peaks(distance=100, height=thr)`
- Window A starts: `[0, 3000, 6000, 9000]`
- Window B starts: `[0, 1500, 3000, 4500, 6000, 7500, 9000]`

## Smoke cases

See `artifacts/results/stage9/segphase_alignment_cases.csv`.

## Batch / determinism / CPU-GPU

- batch_consistency: `True`
- determinism: `True`
- cpu_gpu_ok: `True`
