# Final Claim Audit

Audit basis: frozen artifacts under `artifacts/results/{pairwise_*,multistation*,stage6}` + `FINAL_EVIDENCE.LOCKED`.

## SUPPORTED

1. **Scalar pairwise reranking improved S-phase candidate selection on confirm**  
   F1@0.5: 0.8535 vs fixed 0.8373 (Δ=+0.0162); bootstrap CI excludes zero.

2. **Improvement replicated across held-out, full-dev, and confirm**  
   Δ(scalar−fixed): +0.0207 / +0.0137 / +0.0162.

3. **Scalar pairwise exceeded a residual-only control on confirm**  
   Δ=+0.0076; bootstrap CI [+0.0065, +0.0088].

4. **Among ranking-recoverable errors, the correct candidate was overwhelmingly near the top**  
   Rank2: 83.8%; top3: 97.9%; top5: 100% (n_recoverable=2173).

5. **Pairwise switches were mostly beneficial on confirm**  
   Fixes 907 vs breaks 210 (net +697); switch precision ≈0.812; Q2 recovery ≈0.773.

6. **Residual control is a train-only propagation/history reweighting control**  
   Correlation with base_tau residual ≈0.998; not new source info / not neighbor labels.

7. **Waveform conditional increment over scalar is not material on the pilot held-out**  
   Δ≈+0.00046; bootstrap CI crosses zero.

8. **Multi-station moveout geometry did not dominate single-station residual control**  
   Legal best Δ≈+0.0056; high agreement with single-station resid; `MULTISTATION_MOVEOUT.NO_GO`.

## SUPPORTED WITH QUALIFIER

| Claim phrasing to avoid | Required qualifier |
| --- | --- |
| “phase picking improvement” | **catalog-assisted S-phase candidate reranking / re-picking** |
| “picker improvement” | Method does **not** generate new picks; selects c1/c2 from frozen UNION |
| “errors are mainly ranking errors” | Only among **ranking-recoverable** cases; candidate-missing errors remain large |
| “no distribution shift” | Effects **replicated** across splits; not “fully shift-free” |
| “waveform has no information” | Waveform is informative **in isolation**; negligible **conditional** gain given scalars |
| “multi-station failed” | Apparent geometry gain largely explained by **single-station theoretical residual** |
| “confirmed improvement” | Allowed **only** for frozen confirm metrics / locks above |
| “generalizes” | Must say **replicated on three evaluation stages** in this dataset/domain |

Canonical sentence:

> The proposed method does not generate new phase candidates; it reranks the two highest-scoring S-phase candidates from a frozen candidate set.

## NOT SUPPORTED (forbidden)

- State-of-the-art phase picker  
- Blind picker / blind continuous detector  
- End-to-end picker  
- Real-time earthquake monitoring system  
- Multi-station model improves the final main method  
- Universal generalization / cross-region proof  
- Candidate oracle as a deployable method  
- Waveform “useless” / “no signal”  
- All S-phase errors are ranking errors  
- Confirm-derived gates / τ retuning justified by confirm margins  

## Contribution draft (not locked as paper text)

1. Identify candidate-ranking error as an important recoverable failure mode in catalog-assisted S-phase repicking.  
2. Introduce a lightweight pairwise reranking framework that selects between the two highest-ranked frozen S candidates without generating new picks.  
3. Demonstrate replicated improvement over fixed rescoring and a strong train-only residual control, including a fully held-out confirm set.  
4. (Optional / Discussion) Controlled ablations show limited conditional benefit from additional waveform and multi-station geometric information in this setting.
