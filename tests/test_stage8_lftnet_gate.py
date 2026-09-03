"""Stage 8 gate tests — incomplete LFTNet drop must not unlock full eval / SOTA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_stage6_method_lock_unchanged():
    lock = ROOT / "artifacts/results/stage6/final_confirm/method_lock.json"
    assert sha256(lock) == "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"
    assert (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists()


def test_provenance_incomplete_and_gate_blocks():
    man = json.loads((ROOT / "artifacts/results/stage8/lftnet_release_manifest.json").read_text())
    assert man["provenance_class"].startswith("E.")
    assert man["gate"]["auto_continue_full_eval"] is False
    smoke = json.loads((ROOT / "artifacts/results/stage8/lftnet_smoke_metrics.json").read_text())
    assert smoke["checkpoint_loaded"] is False
    assert smoke["random_init_refused"] is True
    assert smoke["alignment_gate"].startswith("FAIL")


def test_no_checkpoint_under_drop():
    root = Path(json.loads((ROOT / "artifacts/results/stage8/lftnet_release_manifest.json").read_text())["resolved_primary_root"])
    ckpts = list(root.rglob("*.h5")) + list(root.rglob("*.pt")) + list(root.rglob("*.pth")) + list(root.rglob("*.ckpt"))
    assert ckpts == []


def test_utils_has_no_imports():
    root = Path(json.loads((ROOT / "artifacts/results/stage8/lftnet_release_manifest.json").read_text())["resolved_primary_root"])
    src = (root / "se-tcn-Eqt_utils.py").read_text(encoding="utf-8")
    assert "import " not in src.split("def ")[0] or True
    # stronger: first non-empty lines are def, not import
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    assert not any(lines[i].startswith("import ") or lines[i].startswith("from ") for i in range(min(5, len(lines))))


def test_final_verdict_no_sota_no_replace():
    v = json.loads((ROOT / "artifacts/results/stage8/lftnet_final_verdict.json").read_text())
    assert v["final_verdict"] == "implementation_not_reproducible"
    assert v["sota_claim_allowed"] is False
    assert v["replace_stage6_main_method"] is False
    assert v["post_confirm_comparator_ran"] is False


def test_confirm_not_used_for_threshold():
    lock = json.loads((ROOT / "artifacts/results/stage8/lftnet_method_lock.json").read_text())
    assert lock.get("locked") is False
    assert (ROOT / "artifacts/results/stage8/LFTNET_CONFIG_LOCKED").read_text().startswith("NOT_LOCKED")


def test_dev_confirm_event_disjoint_still_holds():
    import pandas as pd

    dev = pd.read_csv(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv", usecols=["event_id"])
    conf = pd.read_csv(
        ROOT / "artifacts/results/stage6/final_confirm/confirm_s_eval_manifest.csv",
        usecols=["event_id"],
    )
    assert len(set(dev.event_id.astype(str)) & set(conf.event_id.astype(str))) == 0
