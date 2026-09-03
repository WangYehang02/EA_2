"""Strict keyed alignment for Stage-6 evaluation (no row-order dependence)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AlignmentError(RuntimeError):
    """Raised when prediction/manifest keys cannot be safely joined."""


def assert_unique_keys(df: pd.DataFrame, key: str = "trace_name", *, label: str = "frame") -> None:
    if key not in df.columns:
        raise AlignmentError(f"{label} missing key column {key}")
    keys = df[key].astype(str)
    if keys.isna().any():
        raise AlignmentError(f"{label} has null {key}")
    n_dup = int(keys.duplicated().sum())
    if n_dup:
        examples = keys[keys.duplicated()].head(5).tolist()
        raise AlignmentError(f"{label} has {n_dup} duplicate {key}, e.g. {examples}")


def keyed_align_predictions(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    key: str = "trace_name",
    require_event_id: bool = True,
    require_sampling_rate: bool = True,
    pred_col: str = "pred_s_sample",
    true_col_manifest: str = "s_arrival_sample",
    true_col_pred: str | None = "true_s_sample",
    event_col: str = "event_id",
    sr_col: str = "sampling_rate_hz",
) -> pd.DataFrame:
    """One-to-one join of predictions onto manifest by trace_name (manifest order)."""
    man = manifest.copy().reset_index(drop=True)
    pred = predictions.copy().reset_index(drop=True)
    assert_unique_keys(man, key, label="manifest")
    assert_unique_keys(pred, key, label="predictions")

    man[key] = man[key].astype(str)
    pred[key] = pred[key].astype(str)
    man_keys = set(man[key])
    pred_keys = set(pred[key])
    missing = sorted(man_keys - pred_keys)
    extra = sorted(pred_keys - man_keys)
    if missing:
        raise AlignmentError(f"missing predictions for {len(missing)} traces, e.g. {missing[:5]}")
    if extra:
        raise AlignmentError(f"extra predictions not in manifest: {len(extra)}, e.g. {extra[:5]}")
    if len(man) != len(pred):
        raise AlignmentError(f"row count mismatch before join: manifest={len(man)} preds={len(pred)}")

    pred_idx = pred.set_index(key, drop=False)
    # Build aligned rows in manifest order
    rows = []
    for _, mrow in man.iterrows():
        tn = str(mrow[key])
        prow = pred_idx.loc[tn]
        if isinstance(prow, pd.DataFrame):
            raise AlignmentError(f"duplicate key slipped through: {tn}")
        if require_event_id and event_col in man.columns and event_col in pred.columns:
            if str(mrow[event_col]) != str(prow[event_col]):
                raise AlignmentError(f"event_id mismatch for {tn}: {mrow[event_col]} vs {prow[event_col]}")
        if require_sampling_rate and sr_col in man.columns and sr_col in pred.columns:
            ms = float(mrow[sr_col])
            ps = float(prow[sr_col])
            if not np.isclose(ms, ps, rtol=0, atol=1e-9):
                raise AlignmentError(f"sampling_rate mismatch for {tn}: {ms} vs {ps}")
        true_m = float(mrow[true_col_manifest]) if true_col_manifest in man.columns and pd.notna(mrow[true_col_manifest]) else np.nan
        if true_col_pred and true_col_pred in pred.columns and pd.notna(prow.get(true_col_pred, np.nan)):
            true_p = float(prow[true_col_pred])
            if np.isfinite(true_m) and not np.isclose(true_m, true_p, rtol=0, atol=1e-6):
                raise AlignmentError(f"true pick mismatch for {tn}: {true_m} vs {true_p}")
        out = {
            "trace_name": tn,
            "event_id": str(mrow[event_col]) if event_col in man.columns else str(prow.get(event_col, "")),
            "sampling_rate_hz": float(mrow[sr_col]) if sr_col in man.columns else float(prow[sr_col]),
            "true_s_sample": true_m,
            "pred_s_sample": float(prow[pred_col]) if pd.notna(prow[pred_col]) else np.nan,
        }
        if "none_of_k" in pred.columns:
            out["none_of_k"] = bool(prow["none_of_k"])
        if "selected_index" in pred.columns:
            out["selected_index"] = int(prow["selected_index"]) if pd.notna(prow["selected_index"]) else -1
        rows.append(out)
    merged = pd.DataFrame(rows)
    if len(merged) != len(man):
        raise AlignmentError(f"join changed row count: before={len(man)} after={len(merged)}")
    if merged[key].tolist() != man[key].tolist():
        raise AlignmentError("aligned frame does not preserve manifest order")
    if require_sampling_rate and ((merged[sr_col] <= 0).any() or merged[sr_col].isna().any()):
        raise AlignmentError("invalid sampling_rate_hz after alignment")
    return merged


def metrics_from_aligned(
    aligned: pd.DataFrame,
    *,
    pred_col: str = "pred_s_sample",
    true_col: str = "true_s_sample",
    sr_col: str = "sampling_rate_hz",
    none_col: str | None = "none_of_k",
) -> dict[str, Any]:
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    pred = aligned[pred_col].to_numpy(float)
    true = aligned[true_col].to_numpy(float)
    sr = aligned[sr_col].to_numpy(float)
    if none_col and none_col in aligned.columns:
        none = aligned[none_col].to_numpy(bool)
    else:
        none = ~np.isfinite(pred)
    return comprehensive_pick_metrics(pred, true, sr, none_mask=none)


def shuffle_invariant_check(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = keyed_align_predictions(manifest, predictions)
    m0 = metrics_from_aligned(base)
    man_shuf = manifest.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    pred_shuf = predictions.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    m1 = metrics_from_aligned(keyed_align_predictions(man_shuf, predictions))
    m2 = metrics_from_aligned(keyed_align_predictions(manifest, pred_shuf))
    m3 = metrics_from_aligned(keyed_align_predictions(man_shuf, pred_shuf))
    keys = ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate", "precision@0.5", "recall@0.5"]
    ok = all(
        np.isclose(m0[k], m1[k], equal_nan=True)
        and np.isclose(m0[k], m2[k], equal_nan=True)
        and np.isclose(m0[k], m3[k], equal_nan=True)
        for k in keys
    )
    if not ok:
        raise AlignmentError("shuffle-invariant metrics check failed")
    return {"ok": True, "baseline": {k: m0[k] for k in keys}}
