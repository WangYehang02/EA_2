# TERMINOLOGY LOCK

## Preferred terms

| Concept | Locked term |
| --- | --- |
| Main method | **scalar pairwise reranker** / `scalar_pairwise` |
| Task | **catalog-assisted S-phase candidate reranking** (or re-picking) |
| Candidate set | **UNION candidates** |
| Baseline | **fixed_rescore_UNION** |
| Control | **residual-only control** / `resid_s` control |
| Decision units | **c1**, **c2** (top-1 / top-2 by `fixed_score`) |
| Threshold | **τ = 0.50** (frozen) |

## Canonical sentence

> The proposed method does not generate new phase candidates; it reranks the two highest-scoring S-phase candidates from a frozen candidate set.

## Do not mix unless explicitly scoped

- picker  
- detector  
- association  
- locator  

If used, name the module (e.g., “base PhaseNet picker provides candidates”).

## Forbidden paper shorthand

- “our picker” for scalar_pairwise  
- “SOTA picker”  
- “blind picker” for this method  
- “multi-station model” as the final method  

## Allowed short forms

- scalar pairwise  
- pairwise reranker  
- confirm / full-dev / held-out (evaluation stages)
