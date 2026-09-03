# Stage 7B — In-Domain Training Feasibility (estimate only)

**No training started.** Estimates use Stage-7A / Stage-6 measured throughput proxies, not theoretical FLOPS.

- Annotate/e2e reference rate: **24.40 traces/s** (from Stage7A runtime `4gpu`)
- Assumed train set size: **800000** traces
- Conservative train throughput proxy: **3.66 traces/s** (≈15% of annotate e2e; single-HDF5 I/O bound)
- Seconds / epoch: **60.71 h**

| Job | Seeds | Epochs (assumed) | Wall estimate | 1-week feasible? |
|--|--:|--:|--|--|
| EQTransformer in-domain seed42 | 1 | 50 | 3035.4 h | no |
| EQTransformer 3 seeds | 3 | 50 | 9106.1 h | marginal/no |
| LFTNet seed42 | 1 | n/a | blocked | `LFTNet_status=not_reproducible_from_official_release` |
| LFTNet 3 seeds | 3 | n/a | blocked | same |

## Disk

- Checkpoints + logs: order **50–200 GB** depending on retention
- Candidate caches (UNION-style): order **tens of GB** if rebuilt for new pickers

## Candidate cache time

At annotate rate 24.4 t/s on 4 GPUs: full INSTANCE 1,159,249 traces ≈ **13.2 h**.

## Verdict

Do **not** start Stage 7B training from this document alone. Revisit after official EQT weights / LFTNet release gates pass.
