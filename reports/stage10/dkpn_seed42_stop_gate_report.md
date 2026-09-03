# DKPN seed42 stop gate

**Verdict:** `implementation_bug`  
**Formal A/B/C gates:** not evaluated (checkpoint ineligible)

Seed42 used `log_softmax(softmax(z))`. Full Stage-6 dev inference, threshold sweep, UNION oracle, and event bootstrap were **not** run as official numbers.

| Gate | Result |
|--|--|
| A ΔF1@0.5 ≥ +0.01 vs waveform-only | *not evaluated* |
| B F1 not down and P95 ≥ 0.15 s | *not evaluated* |
| C UNION oracle +0.01 with CI lo > 0 | *not evaluated* |

- extra seeds 123/2026: **not started**
- DKPN not added to UNION
- SegPhase full training: **not started** (implementation_bug stop)
- Stage 6 confirm: **not read** for metrics/threshold

See `dkpn_seed42_training_effectiveness_audit.md`.
