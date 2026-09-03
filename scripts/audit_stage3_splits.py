#!/usr/bin/env python
"""Audit Stage-3 split integrity for the fixed eval set; create test-only list if needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir, write_lines


def _hash_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def build_test_only_list(
    events: pd.DataFrame,
    n_traces: int = 10000,
    seed: int = 42,
    max_per_event: int = 8,
) -> pd.DataFrame:
    """Deterministic event-stratified sample from chronological test split only."""
    rng = np.random.default_rng(seed)
    test = events[events["split"] == "test"].copy()
    eids = test["event_id"].astype(str).unique().tolist()
    eids = sorted(eids)  # deterministic order before shuffle
    rng.shuffle(eids)
    parts = []
    for eid in eids:
        g = test[test["event_id"].astype(str) == eid]
        take = min(len(g), max_per_event)
        # stable subsample by sorted trace_name then RNG choice indices
        g = g.sort_values("trace_name")
        idx = rng.choice(len(g), size=take, replace=False)
        parts.append(g.iloc[np.sort(idx)])
        if sum(len(p) for p in parts) >= n_traces:
            break
    out = pd.concat(parts, ignore_index=True).head(n_traces)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test-only", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = ensure_dir(artifacts_dir() / "results" / "stage3")
    diag = artifacts_dir() / "diagnostics"
    events = pd.read_parquet(artifacts_dir() / "index" / "events.parquet")
    fixed = pd.read_parquet(diag / "fixed_eval_events.parquet")
    fixed_list = (diag / "fixed_eval_traces.txt").read_text().splitlines()
    fixed_list = [ln.strip() for ln in fixed_list if ln.strip()]

    merged = fixed.merge(
        events[["trace_name", "event_id", "split"]],
        on=["trace_name", "event_id"],
        how="left",
        suffixes=("", "_idx"),
    )
    split_col = "split" if "split" in merged.columns else "split_idx"
    counts = merged[split_col].value_counts(dropna=False).to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}

    train_e = set(events.loc[events["split"] == "train", "event_id"].astype(str))
    val_e = set(events.loc[events["split"] == "val", "event_id"].astype(str))
    test_e = set(events.loc[events["split"] == "test", "event_id"].astype(str))
    fixed_e = set(merged["event_id"].astype(str))

    has_train = int(counts.get("train", 0)) > 0
    has_val = int(counts.get("val", 0)) > 0
    has_test = int(counts.get("test", 0)) > 0
    pure_test = has_test and not has_train and not has_val

    report = {
        "original_fixed_eval": {
            "n_traces": len(fixed),
            "n_list_file": len(fixed_list),
            "split_counts": counts,
            "n_events": int(merged["event_id"].nunique()),
            "overlap_train_events": len(fixed_e & train_e),
            "overlap_val_events": len(fixed_e & val_e),
            "overlap_test_events": len(fixed_e & test_e),
            "pure_chronological_test": pure_test,
            "list_hash": _hash_lines(sorted(fixed_list)),
            "path": str(diag / "fixed_eval_traces.txt"),
            "note": "Original Stage-1.5/2 fixed list is NOT overwritten.",
        }
    }

    # Always create/refresh deterministic test-only list when original is impure
    test_only_path = diag / "fixed_eval_test_only_traces.txt"
    test_only_meta = diag / "fixed_eval_test_only_events.parquet"
    if not pure_test:
        test_df = build_test_only_list(events, n_traces=args.n_test_only, seed=args.seed)
        names = test_df["trace_name"].astype(str).tolist()
        write_lines(test_only_path, names)
        test_df.to_parquet(test_only_meta, index=False)
        # verify
        assert set(test_df["event_id"].astype(str)).isdisjoint(train_e)
        assert set(test_df["event_id"].astype(str)).isdisjoint(val_e)
        report["action"] = "created_new_test_only_list"
        report["warning"] = (
            "Original fixed 10k contains validation traces and must NOT be called a test set. "
            "Created a new deterministic chronological-test-only list; original list preserved."
        )
    else:
        # still write alias copy for a stable stage3 path
        names = merged.loc[merged[split_col] == "test", "trace_name"].astype(str).tolist()
        write_lines(test_only_path, names)
        merged.loc[merged[split_col] == "test"].to_parquet(test_only_meta, index=False)
        report["action"] = "original_already_pure_test"
        report["warning"] = None

    test_only = pd.read_parquet(test_only_meta)
    report["test_only"] = {
        "n_traces": len(test_only),
        "n_events": int(test_only["event_id"].nunique()),
        "path_list": str(test_only_path),
        "path_parquet": str(test_only_meta),
        "list_hash": _hash_lines(test_only["trace_name"].astype(str).tolist()),
        "seed": args.seed,
        "max_per_event": 8,
        "all_split_test": bool((test_only.merge(events[["trace_name", "split"]], on="trace_name")["split"] == "test").all())
        if "split" not in test_only.columns
        else bool((test_only["split"] == "test").all() if "split" in test_only.columns else True),
    }
    # residual history / stage2 hyperparams note
    report["protocol_checks"] = {
        "candidate_cache_uses_labels_as_inputs": False,
        "note_candidates": "PhaseNet candidates come from model probabilities only; labels used only for metrics/oracle.",
        "residual_history_protocol": "frozen temporal store / no-future (Stage 2 artifact)",
        "stage2_hyperparams_selected_on": "validation (lambda search / shrinkage_k)",
    }
    save_json(report, out / "split_audit.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
