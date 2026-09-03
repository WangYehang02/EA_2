"""Stage 10 protocol / leakage / blind-feature tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def test_stage10_frozen_hashes_unchanged():
    lock = ROOT / "artifacts/results/stage6/final_confirm/method_lock.json"
    assert sha256(lock) == "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303"
    assert (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists()


def test_stage10_train_dev_event_disjoint():
    from earthquake.stage10.protocol import assert_event_disjoint, load_event_ids

    pt = load_event_ids("picker_train")
    dv = load_event_ids("dev")
    cf = load_event_ids("internal_confirm")
    rt = load_event_ids("ranker_train")
    assert_event_disjoint(pt, dv, label="pt-dev")
    assert_event_disjoint(pt, cf, label="pt-cf")
    assert_event_disjoint(rt, dv, label="rt-dev")
    assert_event_disjoint(dv, cf, label="dev-cf")


def test_stage10_enz_zne_and_segphase_map():
    from earthquake.stage10.protocol import enz_to_ud_ns_ew, enz_to_zne

    enz = np.array([[1.0, 2], [3.0, 4], [5.0, 6]], dtype=np.float32)
    zne = enz_to_zne(enz)
    assert np.allclose(zne[0], [5.0, 6.0])  # Z
    assert np.allclose(zne[1], [3.0, 4.0])  # N
    assert np.allclose(zne[2], [1.0, 2.0])  # E
    assert np.allclose(enz_to_ud_ns_ew(enz), zne)


def test_stage10_trace_name_join_and_shuffle_invariant_metrics():
    from earthquake.stage10.protocol import join_predictions_by_trace_name
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    rng = np.random.default_rng(0)
    n = 50
    man = pd.DataFrame(
        {
            "trace_name": [f"T{i}" for i in range(n)],
            "event_id": [f"E{i//5}" for i in range(n)],
            "sampling_rate_hz": 100.0,
            "s_arrival_sample": rng.uniform(1000, 5000, n),
        }
    )
    preds = pd.DataFrame(
        {
            "trace_name": man.trace_name.values,
            "pred_s_sample": man.s_arrival_sample.values + rng.normal(0, 10, n),
        }
    )
    joined = join_predictions_by_trace_name(man, preds)
    m0 = comprehensive_pick_metrics(
        joined.pred_s_sample.to_numpy(float),
        joined.s_arrival_sample.to_numpy(float),
        joined.sampling_rate_hz.to_numpy(float),
    )
    man2 = man.sample(frac=1.0, random_state=1).reset_index(drop=True)
    preds2 = preds.sample(frac=1.0, random_state=2).reset_index(drop=True)
    joined2 = join_predictions_by_trace_name(man2, preds2)
    # metrics on man2 order
    m1 = comprehensive_pick_metrics(
        joined2.pred_s_sample.to_numpy(float),
        joined2.s_arrival_sample.to_numpy(float),
        joined2.sampling_rate_hz.to_numpy(float),
    )
    assert abs(m0["f1@0.5"] - m1["f1@0.5"]) < 1e-12


def test_stage10_p_only_masked_loss_not_noise():
    from earthquake.stage10.protocol import build_psn_targets

    t, mask = build_psn_targets(1000, 200.0, None, has_p=True, has_s=False)
    assert mask[0] == 1.0 and mask[1] == 0.0
    assert t[1].sum() == 0.0  # no S soft label
    # S channel not forced; noise derived from P only
    assert t[2, 200] < 0.5


def test_stage10_blind_forbidden_features():
    from earthquake.stage10.protocol import assert_blind_feature_frame

    ok = pd.DataFrame({"trace_name": ["a"], "peak_prob": [0.9], "source_model": ["pn"]})
    assert_blind_feature_frame(ok)
    bad = ok.assign(distance_km=10.0)
    with pytest.raises(RuntimeError):
        assert_blind_feature_frame(bad)


def test_stage10_confirm_not_for_tuning_flag():
    reg = Path("artifacts/results/stage10/stage10A_protocol_registry.json")
    if not reg.exists():
        pytest.skip("10A not written")
    import json

    d = json.loads(reg.read_text())
    assert d["confirm_usable_for_stage10_tuning"] is False
    assert d["exact_lftnet_protocol_reproducible"] is False
    assert d["sota_claim_allowed"] is False


def test_stage10_dkpn_official_excluded_from_main():
    reg = Path("artifacts/results/stage10/stage10A_protocol_registry.json")
    if not reg.exists():
        pytest.skip("10A not written")
    import json

    d = json.loads(reg.read_text())
    assert d["baselines"]["DKPN"]["official_weights_usable_for_main"] is False


def test_stage10_augmentation_label_shift_helper():
    """Time-shift must move labels with waveform (unit for train loop)."""
    wave = np.zeros((3, 200), np.float32)
    wave[:, 50] = 1.0
    p_sample = 50.0
    shift = 10
    wave2 = np.roll(wave, shift, axis=-1)
    p2 = p_sample + shift
    assert wave2[0, 60] == 1.0
    assert p2 == 60.0
