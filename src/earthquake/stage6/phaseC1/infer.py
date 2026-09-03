"""Ranker inference helpers including post-hoc diagnostic variants (no training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.models import build_ranker


@torch.no_grad()
def predict_ranker_variants(
    ckpt_path: Path | str,
    feat_df: pd.DataFrame,
    *,
    device: str = "cuda:0",
    fixed_fallback: np.ndarray | None = None,
    meta_trace_names: list[str] | None = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Return per-trace predictions for reported / forced_choice / none_fallback_fixed.

    Forced choice: argmax over non-padding candidate logits only (ignore none logit).
    None fallback fixed: keep ranker pick when a real candidate; else use fixed_fallback.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = list(blob["feature_names"])
    model = build_ranker(blob["variant"], len(names))
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    ds = TraceCandidateDataset(feat_df, names, max_k=10, hard_boost=False, seed=0)
    ds.indices = list(range(len(ds.groups)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_traces)
    max_k = 10

    fixed_map = {}
    if fixed_fallback is not None and meta_trace_names is not None:
        fixed_map = {str(t): float(v) for t, v in zip(meta_trace_names, fixed_fallback)}

    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)
        # reported: full argmax including none
        pred_idx = logits.argmax(dim=-1).cpu().numpy()
        # forced: only among valid candidates
        cand_logits = logits[:, :max_k].clone()
        neg = torch.finfo(cand_logits.dtype).min / 4
        cand_logits = cand_logits.masked_fill(~mask, neg)
        # if a row has no valid candidates, forced idx stays 0 but mask False → nan
        forced_idx = cand_logits.argmax(dim=-1).cpu().numpy()
        samples = batch["samples"].numpy()
        mask_np = mask.cpu().numpy()
        for i, tn in enumerate(batch["trace_name"]):
            pi = int(pred_idx[i])
            fi = int(forced_idx[i])
            reported = np.nan if pi >= max_k or (not bool(mask_np[i, pi])) else float(samples[i, pi])
            forced = np.nan if (not bool(mask_np[i].any())) or (not bool(mask_np[i, fi])) else float(samples[i, fi])
            is_none = bool(pi >= max_k)
            if is_none:
                fallback = fixed_map.get(str(tn), np.nan)
            else:
                fallback = reported
            rows.append(
                {
                    "trace_name": str(tn),
                    "event_id": str(batch["event_id"][i]),
                    "true_s_sample": float(batch["true_s"][i]),
                    "sampling_rate_hz": float(batch["sr"][i]),
                    "pred_reported": reported,
                    "pred_forced_choice": forced,
                    "pred_none_fallback_fixed": fallback,
                    "selected_index": pi,
                    "forced_index": fi,
                    "none_of_k": is_none,
                    "n_valid_cand": int(mask_np[i].sum()),
                    "none_logit": float(logits[i, max_k].detach().cpu()),
                    "max_cand_logit": float(cand_logits[i].max().detach().cpu()) if mask_np[i].any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def align_preds_to_meta(pred_df: pd.DataFrame, meta: pd.DataFrame, col: str) -> np.ndarray:
    m = dict(zip(pred_df["trace_name"].astype(str), pred_df[col].to_numpy(float)))
    return np.asarray([m.get(str(t), np.nan) for t in meta["trace_name"].astype(str)], dtype=float)
