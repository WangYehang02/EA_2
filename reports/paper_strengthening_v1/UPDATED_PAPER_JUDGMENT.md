# Updated paper judgment (after independent-period validation)

## Execution
- Independent-period formal eval: **COMPLETED**
- Primary ΔF1@0.5 (scalar−C) = 0.005473, 95% CI [0.003592, 0.007291]
- Secondary ΔF1@0.5 (C−fixed) = 0.011787, 95% CI [0.009247, 0.014457]
- Tail ΔMAE (scalar−C) = -1.252127; Δfrac>5s = -0.010173

## Paper-facing conclusions
- **Simple prior (C / base_τ ranking) remains a primary effective method** relative to fixed on this new period (ΔF1=0.0118). Treat simple catalog-assisted prior + candidate ranking as the main story backbone.
- Scalar shows a **repeatable F1@0.5 gain** vs C on the new period (Δ≈0.0055, CI above zero and above the 0.003 discussion scale).
- Scalar also retains **tail refinement value** (MAE ↓≈1.25 s; frac>5s ↓≈0.010); keep these as secondary, pre-registered tail evidence — not a post-hoc primary swap.

## Still not claimed
- Cross-region generalization
- Blind / end-to-end picking superiority
- Equivalence solely from CI crossing zero

Full report: `reports/paper_strengthening_v1/INDEPENDENT_PERIOD_VALIDATION_REPORT.md`
