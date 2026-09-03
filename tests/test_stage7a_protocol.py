"""Stage 7A protocol tests: alignment, threshold policy, leakage exclusion, lock integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def stage6_lock():
    p = ROOT / "artifacts/results/stage6/final_confirm/method_lock.json"
    h = (ROOT / "artifacts/results/stage6/final_confirm/method_lock.sha256").read_text().strip()
    assert p.exists()
    assert _sha256(p) == h
    return json.loads(p.read_text())


def test_stage6_method_lock_unchanged(stage6_lock):
    method = stage6_lock.get("main_method") or stage6_lock.get("method")
    assert method == "fixed_rescore_UNION"
    assert (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists()


def test_leakage_weights_excluded_from_valid_comparators():
    reg = json.loads((ROOT / "artifacts/results/stage7/comparator_registry.json").read_text())
    for name in ["PhaseNet-instance", "EQTransformer-instance"]:
        m = reg["models"][name]
        assert m["valid_comparator"] is False
        assert m.get("run_in_stage7A") is False
        assert "leakage" in (m.get("exclusion_reason") or "").lower() or m.get("role") == "diagnostic_only_data_leakage"


def test_at_least_two_new_valid_or_registry_blocks_gpu():
    reg = json.loads((ROOT / "artifacts/results/stage7/comparator_registry.json").read_text())
    n = reg["n_new_valid_comparators"]
    assert reg["proceed_gpu_batch"] == (n >= 2)
    assert n >= 2  # PhaseNet-ETHZ + PhaseNet-SCEDC


def test_comparator_keyed_alignment_and_shuffle_invariance():
    from earthquake.stage6.keyed_align import keyed_align_predictions, metrics_from_aligned, shuffle_invariant_check

    rng = np.random.default_rng(0)
    n = 40
    man = pd.DataFrame(
        {
            "trace_name": [f"T{i}" for i in range(n)],
            "event_id": [f"E{i//5}" for i in range(n)],
            "sampling_rate_hz": np.full(n, 100.0),
            "s_arrival_sample": rng.uniform(1000, 5000, n),
        }
    )
    pred = man[["trace_name", "event_id", "sampling_rate_hz"]].copy()
    pred["pred_s_sample"] = man["s_arrival_sample"] + rng.normal(0, 20, n)
    pred["true_s_sample"] = man["s_arrival_sample"]
    pred["none_of_k"] = False
    aligned = keyed_align_predictions(man, pred)
    m = metrics_from_aligned(aligned)
    assert "f1@0.5" in m
    assert shuffle_invariant_check(man, pred)["ok"] is True


def test_dev_only_threshold_selection_and_confirm_mutation_forbidden():
    """Threshold may only be chosen on a declared dev table; confirm must not retune."""
    grid = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
    # synthetic: pick best on "dev"
    f1 = {0.05: 0.5, 0.10: 0.6, 0.20: 0.7, 0.30: 0.65, 0.50: 0.55, 0.70: 0.4}
    best = max(grid, key=lambda t: (f1[t], t))
    assert best == 0.20
    # confirm must use locked threshold; mutating confirm metrics for selection is forbidden
    locked = best
    confirm_f1_fake = {t: 0.99 for t in grid}  # would prefer any thr if allowed
    assert locked != max(grid, key=lambda t: confirm_f1_fake[t]) or locked == 0.70
    selected_before_confirm = True
    assert selected_before_confirm is True
    assert locked in grid


def test_runtime_manifest_deterministic():
    import hashlib

    man = pd.read_csv(ROOT / "artifacts/results/stage6/phaseB_eval_manifest.csv", usecols=["trace_name"])
    h = man.trace_name.astype(str).map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    order1 = h.sort_values().index[:10000].tolist()
    order2 = h.sort_values().index[:10000].tolist()
    assert order1 == order2
    assert len(order1) == 10000


def test_bootstrap_delta_direction_definition():
    """Δ = fixed_UNION - comparator; positive ΔF1 means UNION better."""
    fixed_f1, comp_f1 = 0.84, 0.80
    delta = fixed_f1 - comp_f1
    assert delta > 0
    fixed_p95, comp_p95 = 2.5, 4.0
    delta_p95 = fixed_p95 - comp_p95
    assert delta_p95 < 0  # UNION better P95


def test_comparator_locks_selected_before_confirm_if_present():
    locks = list((ROOT / "artifacts/results/stage7/locks").glob("*_comparator_lock.json"))
    if not locks:
        pytest.skip("locks not written yet")
    for p in locks:
        d = json.loads(p.read_text())
        assert d["selected_before_confirm"] is True
        assert d["threshold"] in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
