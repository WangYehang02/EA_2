# Candidate edit — Stage 7A external comparators (NOT applied to paper body)

Status: **candidate only**. Do not merge into frozen paper text without human review.

Suggested addition (Results / Baselines):

> Under the identical Stage-6 confirmatory protocol (same manifests, tolerances, and `match_picks` definitions), we evaluated additional SeisBench pretrained PhaseNet weights (ETHZ, SCEDC) with thresholds selected **only** on the Stage-6 development set. These comparisons are **post-confirm** external baselines; they were not co-preregistered with the confirmatory analysis of `fixed_rescore_UNION`. We do not claim state-of-the-art performance.

Also record: INSTANCE-pretrained weights are excluded as `diagnostic_only_data_leakage`. EQTransformer official weights were unavailable in this environment (download failure). LFTNet was not reproducible from an official release.
