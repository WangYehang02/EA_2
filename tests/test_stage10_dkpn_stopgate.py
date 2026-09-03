"""Stage 10 DKPN loss/activation, threshold, bootstrap, confirm-guard tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_stage10_dkpn_masked_s_loss_p_only():
    from earthquake.stage10.dkpn_clean import masked_soft_ce

    b, t = 2, 64
    logits = torch.zeros(b, 3, t)
    logits[:, 0, 10] = 5.0
    logits[:, 2, :] = 1.0
    target = torch.zeros(b, 3, t)
    target[:, 0, 10] = 1.0
    target[:, 2, :] = 1.0
    target[:, 2, 10] = 0.0
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    # S logits huge should not affect loss when masked
    logits_bad_s = logits.clone()
    logits_bad_s[:, 1, :] = 99.0
    l0 = masked_soft_ce(logits, target, mask)
    l1 = masked_soft_ce(logits_bad_s, target, mask)
    assert torch.isfinite(l0) and abs(float(l0 - l1)) < 1e-5


def test_stage10_dkpn_rejects_softmax_as_logits():
    from earthquake.stage10.dkpn_clean import masked_soft_ce

    probs = torch.softmax(torch.randn(2, 3, 16), dim=1)
    target = torch.softmax(torch.randn(2, 3, 16), dim=1)
    mask = torch.ones(2, 3)
    with pytest.raises(RuntimeError, match="simplex"):
        masked_soft_ce(probs, target, mask)


def test_stage10_dkpn_channel_order_enz_zne():
    from earthquake.stage10.protocol import enz_to_zne

    enz = np.arange(9, dtype=np.float32).reshape(3, 3)
    zne = enz_to_zne(enz)
    assert np.allclose(zne[0], enz[2])
    assert np.allclose(zne[2], enz[0])


def test_stage10_threshold_sweep_monotone_miss():
    from earthquake.stage10.candidates import threshold_sweep_f1

    n = 40
    true = np.full(n, 1000.0)
    pred = true + np.linspace(0, 80, n)  # increasing error
    prob = np.linspace(0.99, 0.05, n)
    sr = np.full(n, 100.0)
    df = threshold_sweep_f1(prob, pred, true, sr, np.array([0.1, 0.5, 0.9]))
    assert list(df["threshold"]) == [0.1, 0.5, 0.9]
    # higher threshold → more abstention → miss not decrease
    assert df["miss_rate"].iloc[-1] >= df["miss_rate"].iloc[0] - 1e-12


def test_stage10_oracle_union_not_below_sources():
    from earthquake.stage10.candidates import oracle_hit_rate

    y = np.array([10.0, 20.0, 30.0, 40.0])
    a = np.array([10.0, np.nan, 99.0, np.nan])
    b = np.array([np.nan, 20.0, 30.0, 80.0])
    u = oracle_hit_rate(y, {"A": a, "B": b}, 1.0)
    assert u >= oracle_hit_rate(y, {"A": a}, 1.0) - 1e-12
    assert u >= oracle_hit_rate(y, {"B": b}, 1.0) - 1e-12


def test_stage10_event_bootstrap_resamples_events():
    from earthquake.stage10.candidates import event_bootstrap_delta

    # 3 events, 10 traces each; b is better only on E2
    eids = np.array(["E0"] * 10 + ["E1"] * 10 + ["E2"] * 10)
    a = np.zeros(30)
    b = np.zeros(30)
    b[20:] = 1.0
    out = event_bootstrap_delta(eids, a, b, n_boot=200, seed=0)
    assert "ci95_lo" in out and out["n_boot"] == 200
    assert out["mean_delta"] > 0


def test_stage10_confirm_consumed_guard_and_unused_in_verdict():
    assert (ROOT / "artifacts/results/stage6/final_confirm/CONFIRM.CONSUMED").exists()
    v = ROOT / "artifacts/results/stage10/dkpn/stop_gate_verdict.json"
    if not v.exists():
        pytest.skip("verdict not written yet")
    import json

    d = json.loads(v.read_text())
    assert d.get("confirm_read") is False
    assert d["verdict"] in {
        "implementation_bug",
        "keep_as_primary_picker_candidate",
        "keep_as_union_candidate_only",
        "rejected_candidate_source",
    }


def test_stage10_checkpoint_changes_parameters():
    import hashlib

    p = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean/train_seed42/checkpoints/last.pt")
    if not p.exists():
        pytest.skip("no last.pt")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    h = hashlib.sha256()
    for k in sorted(state):
        if torch.is_tensor(state[k]) and state[k].is_floating_point():
            h.update(state[k].cpu().numpy().tobytes())
    # not all zeros
    n = sum(float(state[k].abs().sum()) for k in state if torch.is_tensor(state[k]) and state[k].is_floating_point())
    assert n > 0
    audit = ROOT / "artifacts/results/stage10/dkpn/training_effectiveness_audit.json"
    if audit.exists():
        import json

        d = json.loads(audit.read_text())
        assert d["weight_update"]["hashes_differ"] is True
        assert d["weight_update"]["param_l2_vs_seed42_init"] > 1e-3


def test_stage10_prediction_join_trace_name_shuffle():
    from earthquake.stage10.protocol import join_predictions_by_trace_name
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    man = pd.DataFrame({"trace_name": ["a", "b", "c"], "s_arrival_sample": [1.0, 2.0, 3.0], "sampling_rate_hz": 100.0})
    pred = pd.DataFrame({"trace_name": ["c", "a", "b"], "pred_s_sample": [3.0, 1.0, 2.0]})
    j = join_predictions_by_trace_name(man, pred)
    m = comprehensive_pick_metrics(j.pred_s_sample.to_numpy(), j.s_arrival_sample.to_numpy(), j.sampling_rate_hz.to_numpy())
    j2 = join_predictions_by_trace_name(man.sample(frac=1, random_state=0).reset_index(drop=True), pred)
    m2 = comprehensive_pick_metrics(j2.pred_s_sample.to_numpy(), j2.s_arrival_sample.to_numpy(), j2.sampling_rate_hz.to_numpy())
    assert abs(m["f1@0.5"] - m2["f1@0.5"]) < 1e-12
