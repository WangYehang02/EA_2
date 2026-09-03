"""Sanity-audit invariants for Stage 5.1 hierarchical residual."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/results/stage5_1_ustc/sanity_audit.json"
VERDICT = ROOT / "artifacts/results/stage5_1_ustc/final_verdict.json"


@pytest.mark.skipif(not AUDIT.exists(), reason="sanity_audit.json not generated yet")
def test_sanity_audit_revokes_hierarchical_recommendation():
    s = json.loads(AUDIT.read_text())
    assert s["hierarchical_prior_recommended"] is False
    assert s["final_classification"] == "implementation_bug"
    assert s["methods_metrics"]["stage5_1_fine_shrunk"]["e2e_p95"] > 10
    assert s["methods_metrics"]["frozen_catalog_rescore_stage2"]["e2e_p95"] < 5
    assert s["agreement"]["frozen_vs_s51_fine_shrunk"]["equivalent"] is False


@pytest.mark.skipif(not VERDICT.exists(), reason="final_verdict.json missing")
def test_final_verdict_hierarchical_false_after_sanity():
    v = json.loads(VERDICT.read_text())
    assert v["hierarchical_prior_recommended"] is False


def test_sanity_script_avoids_stage3_method_selection():
    text = (ROOT / "scripts/audit_stage5_1_hierarchical_sanity.py").read_text()
    assert "fixed_eval_test_only" not in text
    assert "learned_gate_picks_test" not in text
