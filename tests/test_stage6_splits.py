"""Stage 6 split / seal / BN tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.splits import (
    STAGE6_SUBSETS,
    assert_confirm_sealed,
    load_stage6_event_ids,
    load_stage6_trace_names,
    stage6_paths,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/results/stage6/split_audit.json"


@pytest.mark.skipif(not AUDIT.exists(), reason="stage6 splits not built")
def test_stage6_event_disjoint_and_hashes():
    audit = json.loads(AUDIT.read_text())
    assert audit["event_disjoint"] is True
    assert audit["trace_disjoint"] is True
    ids = {name: set(load_stage6_event_ids(name)) for name in STAGE6_SUBSETS}
    for a in STAGE6_SUBSETS:
        for b in STAGE6_SUBSETS:
            if a >= b:
                continue
            assert ids[a].isdisjoint(ids[b])
    traces = {name: set(load_stage6_trace_names(name)) for name in STAGE6_SUBSETS}
    for a in STAGE6_SUBSETS:
        for b in STAGE6_SUBSETS:
            if a >= b:
                continue
            assert traces[a].isdisjoint(traces[b])
    assert audit["sizes"]["stage6_internal_confirm"]["events"] >= 1500


@pytest.mark.skipif(not AUDIT.exists(), reason="stage6 splits not built")
def test_confirm_sealed_blocks_without_method_lock():
    paths = stage6_paths()
    seal = paths["confirm_seal"]
    assert seal.exists()
    assert seal.read_text().strip() == "SEALED"
    if paths["method_lock"].exists():
        pytest.skip("method already locked")
    with pytest.raises(RuntimeError):
        assert_confirm_sealed(allow_if_method_locked=False)


def test_bn_policy_sets_batchnorm_eval():
    m = torch.nn.Sequential(torch.nn.Conv1d(3, 4, 3, padding=1), torch.nn.BatchNorm1d(4), torch.nn.ReLU())
    set_train_bn_eval(m)
    bn = m[1]
    assert bn.training is False
    assert m[0].training is True


def test_stage6_train_script_refuses_instance_and_checks_confirm():
    text = (ROOT / "scripts/train_stage6_phasenet.py").read_text()
    assert "INSTANCE pretrained" in text
    assert "stage6_internal_confirm" in text
    assert "set_train_bn_eval" in text
