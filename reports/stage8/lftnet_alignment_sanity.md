# Stage 8 — LFTNet Alignment / Smoke Sanity

**Status:** `BLOCKED` — full inference **not started**.

## Why

Provenance class `E. incomplete_or_unusable`: drop is not a runnable release.

- No official checkpoint under the drop.
- `se-tcn-Eqt_utils.py` has **no import statements** and references Keras/custom layers (`SeqSelfAttention`, `FeedForward`, `f1`, …) that are not defined in-file.
- Refusing to run with **random initialization** (would be scientifically invalid).

## Checklist (all not_run / fail)

| Check | Result |
|--|--|
| checkpoint loads (not random) | FAIL |
| input shape / 100 Hz / ENZ order | not_run |
| output class order / length / sample offset | not_run |
| 60s↔120s / UTC remap / boundary | not_run |
| batch consistency / CPU-GPU / determinism | not_run |

## Gate

`alignment_gate = FAIL_STOP_NO_FULL_INFER`

Per Stage-8 rules: **do not** run Stage-6-protocol confirm comparator or candidate complementarity GPU jobs until alignment passes.
