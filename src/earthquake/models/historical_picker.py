from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from earthquake.history.prior import build_priors_for_row
from earthquake.history.temporal_store import PathStats
from earthquake.models.fixed_fusion import linear_fusion, log_space_fusion
from earthquake.picking import extract_pick_features, pick_from_prob


@dataclass
class PickResult:
    p_sample: float
    s_sample: float
    p_prob: float
    s_prob: float
    alpha_p: float
    alpha_s: float
    history_available: bool
    history_count: int
    method: str


class HistoricalPicker:
    def __init__(
        self,
        alpha_p: float = 0.75,
        alpha_s: float = 0.75,
        fusion: str = "linear",
        prior_mode: str = "catalog_assisted",
        min_sigma_s: float = 0.05,
        max_sigma_s: float = 1.0,
        pick_threshold: float = 0.3,
    ):
        self.alpha_p = alpha_p
        self.alpha_s = alpha_s
        self.fusion = fusion
        self.prior_mode = prior_mode
        self.min_sigma_s = min_sigma_s
        self.max_sigma_s = max_sigma_s
        self.pick_threshold = pick_threshold

    def fuse(
        self,
        phasenet_out: dict[str, np.ndarray],
        row,
        stats: PathStats,
    ) -> dict[str, Any]:
        n = len(phasenet_out["p"])
        p_feat = extract_pick_features(phasenet_out["p"])
        priors = build_priors_for_row(
            row,
            stats,
            n_samples=n,
            mode=self.prior_mode,
            phasenet_p_sample=p_feat["peak_sample"],
            min_sigma_s=self.min_sigma_s,
            max_sigma_s=self.max_sigma_s,
        )
        force = bool(priors["force_phasenet"])
        fuse_fn = linear_fusion if self.fusion == "linear" else log_space_fusion
        final_p = fuse_fn(phasenet_out["p"], priors["prior_p"], self.alpha_p, force_phasenet=force)
        # In blind_s, P has no historical absolute constraint -> always PhaseNet for P
        if self.prior_mode == "blind_s":
            final_p = np.asarray(phasenet_out["p"], dtype=np.float32)
        final_s = fuse_fn(phasenet_out["s"], priors["prior_s"], self.alpha_s, force_phasenet=force)

        p_pick = pick_from_prob(final_p, threshold=self.pick_threshold)
        s_pick = pick_from_prob(final_s, threshold=self.pick_threshold)
        return {
            "final_p": final_p,
            "final_s": final_s,
            "prior_p": priors["prior_p"],
            "prior_s": priors["prior_s"],
            "p_pick": p_pick,
            "s_pick": s_pick,
            "history_available": bool(stats.history_available),
            "history_count": int(stats.history_count),
            "force_phasenet": force,
            "phasenet_features_p": p_feat,
            "phasenet_features_s": extract_pick_features(phasenet_out["s"]),
        }
