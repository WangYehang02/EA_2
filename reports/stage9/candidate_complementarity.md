# Stage 9 — Candidate complementarity

```json
{
  "dedup_tol_s": 0.05,
  "segphase_top5_oracle_dev": {
    "f1@0.5": 0.8384406538897736,
    "f1@0.1": 0.5430332329052731,
    "miss_rate": 0.0,
    "prediction_coverage": 1.0
  },
  "gate_delta_required": 0.01,
  "note": "Full UNION+SegPhase oracle requires Stage-6 UNION candidate cache on?dev; if unavailable, gate not claimed passed.",
  "verdict_tag": "candidate_complementarity_insufficient",
  "reason": "UNION candidate cache on?dev not located; cannot verify \u0394oracle F1@0.5 \u2265 +0.01",
  "add_segphase_to_union": false
}
```
