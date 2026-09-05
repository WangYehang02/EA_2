# Figure 1 — Method flow (select, not generate)

## Nodes

1. **Waveform** — three-component seismic trace for one station-event pair.
2. **Base pickers** — frozen PhaseNet / EQTransformer (or equivalent) probability streams.
3. **UNION candidates** — merged S-phase candidate set from base pickers (frozen generator).
4. **fixed_score** — deterministic rescoring with `lw=0.5`, `lh=2`, `lp=0` using catalog-assisted expected arrival.
5. **c1 / c2** — highest and second-highest `fixed_score` candidates (tie → `candidate_index`).
6. **scalar_pairwise** — learned comparison of scalar features of (c1, c2) with frozen τ=0.50.
7. **Final candidate** — select c1 or c2 only; never generate a new arrival time.

## Decision rule

- If `<2` candidates: output c1 (no pairwise decision).
- Else: if `P(prefer c2) > τ` switch to c2; else keep c1.
- Forbidden: abstain, rank≥3 selection, new pick generation, arrival-time edit.

## Emphasis

The method **selects** among frozen candidates; it does **not** generate candidates.
