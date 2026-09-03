#!/usr/bin/env python
"""Pre-confirm contamination audit (no waveforms)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage6" / "final_confirm")
    audit = load_json(artifacts_dir() / "results" / "stage6" / "full_split_audit.json")
    ce = set(load_full_event_ids("stage6_internal_confirm"))
    ct = set(load_full_trace_names("stage6_internal_confirm"))
    pe = set(load_full_event_ids("stage6_picker_train"))
    re = set(load_full_event_ids("stage6_ranker_train"))
    de = set(load_full_event_ids("stage6_dev"))

    # Stage1-5 prior used (from audit / never_used)
    never = set(Path(artifacts_dir() / "results" / "stage6" / "never_used_event_ids.txt").read_text().splitlines())
    # confirm should be subset of never_used
    not_never = sorted(ce - never)

    # history features must not contain confirm
    hist = pd.read_parquet(
        artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet",
        columns=["subset", "trace_name", "event_id"],
    )
    hist_ev = set(hist.event_id.astype(str))
    hist_tr = set(hist.trace_name.astype(str))
    hist_overlap_ev = sorted(ce & hist_ev)
    hist_overlap_tr = sorted(ct & hist_tr)

    # ID-A train meta
    train_meta = pd.read_parquet(artifacts_dir() / "models" / "stage6" / "phasenet_ida_full_seed42" / "train_meta.parquet", columns=["event_id", "trace_name"])
    ida_ev = set(train_meta.event_id.astype(str)) - {"NOISE"}
    ida_overlap = sorted(ce & ida_ev)

    # existing confirm caches?
    cache_hits = []
    for p in (artifacts_dir() / "cache" / "stage6").rglob("*confirm*"):
        cache_hits.append(str(p))

    # duplicates
    ev_list = load_full_event_ids("stage6_internal_confirm")
    tr_list = load_full_trace_names("stage6_internal_confirm")
    dup_ev = len(ev_list) - len(set(ev_list))
    dup_tr = len(tr_list) - len(set(tr_list))

    contaminated = (
        len(ce & pe)
        + len(ce & re)
        + len(ce & de)
        + len(not_never)
        + len(hist_overlap_ev)
        + len(hist_overlap_tr)
        + len(ida_overlap)
        + dup_ev
        + dup_tr
    )

    doc = {
        "n_confirm_events": len(ce),
        "n_confirm_traces": len(ct),
        "expected_events": 2700,
        "expected_traces": 74753,
        "sizes_match_full_split_audit": len(ce) == 2700 and len(ct) == 74753,
        "time_range": audit["manifests"]["stage6_internal_confirm"],
        "disjoint_picker_train": len(ce & pe) == 0,
        "disjoint_ranker_train": len(ce & re) == 0,
        "disjoint_dev": len(ce & de) == 0,
        "confirm_all_never_used_in_stage1_5": audit.get("confirm_all_never_used_in_stage1_5"),
        "confirm_contaminated_excluded_n_audit": audit.get("confirm_contaminated_excluded_n"),
        "not_in_never_used_list_n": len(not_never),
        "not_in_never_used_examples": not_never[:5],
        "history_store_overlap_events": len(hist_overlap_ev),
        "history_store_overlap_traces": len(hist_overlap_tr),
        "ida_train_overlap_events": len(ida_overlap),
        "duplicate_confirm_events": dup_ev,
        "duplicate_confirm_traces": dup_tr,
        "existing_confirm_named_caches": cache_hits,
        "existing_caches_note": "named *confirm* paths listed; must not have been used for method selection",
        "no_human_review_log_found": not (artifacts_dir() / "results" / "stage6" / "confirm_human_review.json").exists(),
        "contaminated_events": int(contaminated),
        "preconfirm_audit_passed": contaminated == 0 and len(ce) == 2700 and len(ct) == 74753 and audit.get("confirm_all_never_used_in_stage1_5") is True,
    }
    save_json(doc, out / "preconfirm_contamination_audit.json")
    md = f"""# Pre-confirm Contamination Audit

**passed:** `{doc['preconfirm_audit_passed']}`  
**contaminated_events:** {doc['contaminated_events']}  
**confirm events/traces:** {doc['n_confirm_events']} / {doc['n_confirm_traces']}

Disjoint vs picker/ranker/dev: {doc['disjoint_picker_train']}/{doc['disjoint_ranker_train']}/{doc['disjoint_dev']}  
History overlap events/traces: {doc['history_store_overlap_events']}/{doc['history_store_overlap_traces']}  
ID-A train overlap: {doc['ida_train_overlap_events']}
"""
    (ROOT / "reports/stage6/preconfirm_contamination_audit.md").write_text(md)
    if not doc["preconfirm_audit_passed"]:
        raise SystemExit("PRECONFIRM_FAILED")
    print(json.dumps({"preconfirm_audit_passed": True, "sha256": sha256_file(out / "preconfirm_contamination_audit.json")}, indent=2))


if __name__ == "__main__":
    main()
