# DKPN 30 s crop & partial-label audit

**UTC:** 2026-08-25T02:45:33.435376+00:00  
Confirm waveforms: **not read**. Split lists only (picker_train ∩ confirm = 0).

## Geometry

- Raw crop: **3401** samples (~34.01 s) including 400 sample FP-stab
- Label / network window: **3001** samples (~30.01 s) @ 100 Hz
- A 30 s crop **cannot** always hold both P and S; we do **not** require it.

## picker_train P–S (n=377087 traces, 254708 P-only)

Seconds: min=-0.060 median=5.030 P75=7.860 P90=11.640 P95=15.260 P99=25.990 max=56.190

| threshold | traces | events | frac traces |
|--|--:|--:|--:|
| >20 s | 8339 | 1747 | 0.0221 |
| >25 s | 4358 | 739 | 0.0116 |
| >30 s | 1481 | 302 | 0.0039 |

## v1 (buggy seed42) crop

Single midpoint `both` crop. Visibility fractions (one crop / PS trace):

- P+S visible: 0.9670
- P only: 0.0056
- S only: 0.0263
- neither: 0.0010

v1 **still applied full-window N/CE** when a phase was outside → that phase was trained as noise. **Invalid.**

## v2 policy

Per PS trace: **P-centered + S-centered + background**. Long P–S **not dropped** (1481 traces >30 s).

- Any-crop covers P: 1.0000; covers S: 1.0000
- Outside phase: **not** a negative N class; use `-log[p(S)+p(N)]` or `-log[p(P)+p(N)]`
- Truncated Gaussians renormalized in-window; padding `pad_mask=0`
- Noise traces: full-window N only

See `crop_visibility_stats.json`.
