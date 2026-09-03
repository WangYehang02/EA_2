"""Threshold protocol, phase-balanced loss, checkpoint policy (no HDF5, no confirm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from earthquake.stage10.checkpoint_policy import is_valid_best
from earthquake.stage10.dataset_v2 import build_partial_weights
from earthquake.stage10.diag_splits import sha256_sorted_ids, split_heldout_by_event
from earthquake.stage10.frozen_cycle import FrozenCycleSampler
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.phase_balanced_loss import phase_balanced_partial_nll
from earthquake.stage10.threshold_calibration import (
    PROTOCOL_NAME,
    THRESH_GRID,
    pick_metrics_at_thr,
    select_threshold_on_calibration,
)


def test_threshold_grid_includes_required_values():
    need = {0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5}
    assert need.issubset(set(THRESH_GRID))


def test_cal_eval_events_disjoint_and_hash_stable():
    held = pd.DataFrame(
        {
            "event_id": [f"E{i//2}" for i in range(20)],
            "trace_name": [f"T{i}" for i in range(20)],
            "s_arrival_sample": np.arange(20, dtype=float),
        }
    )
    cal, evl = split_heldout_by_event(held, seed=42)
    assert set(cal.event_id) & set(evl.event_id) == set()
    cal2, evl2 = split_heldout_by_event(held, seed=42)
    assert sha256_sorted_ids(cal.event_id) == sha256_sorted_ids(cal2.event_id)
    assert sha256_sorted_ids(evl.event_id) == sha256_sorted_ids(evl2.event_id)


def test_same_protocol_init_and_trained_does_not_use_eval():
    rng = np.random.default_rng(0)
    t = np.full(8, 100.0)
    vis = np.ones(8, dtype=bool)
    cal_s = rng.random((8, 300))
    cal_s[:, 100] = 0.4
    eval_s = rng.random((8, 300))
    eval_s[:, 10] = 0.9
    proto = select_threshold_on_calibration(cal_s, t, vis)
    frozen = pick_metrics_at_thr(eval_s, t, vis, proto["selected_thr"])
    proto2 = select_threshold_on_calibration(cal_s, t, vis)
    assert proto["protocol"] == PROTOCOL_NAME == proto2["protocol"]
    assert proto["selected_thr"] == proto2["selected_thr"]
    assert frozen["thr"] == proto["selected_thr"]
    sneak = select_threshold_on_calibration(eval_s, t, vis)
    assert sneak["selected_thr"] != proto["selected_thr"] or sneak["calibration_metrics_at_selected"]["f1@0.5s"] != proto[
        "calibration_metrics_at_selected"
    ]["f1@0.5s"]


def test_phase_balanced_not_token_ce():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 64, requires_grad=True)
    y = torch.softmax(torch.randn(2, 3, 64), dim=1)
    pad = torch.ones(2, 64)
    z = torch.zeros(2, 64)
    is_noise = torch.zeros(2)
    a = partial_label_nll(logits, p_pos=y[:, 0], s_pos=y[:, 1], n_pos=y[:, 2], not_p=z, not_s=z, pad_mask=pad)
    b, stats, _ = phase_balanced_partial_nll(
        logits, p_pos=y[:, 0], s_pos=y[:, 1], n_pos=y[:, 2], not_p=z, not_s=z, pad_mask=pad, is_noise=is_noise
    )
    assert stats["not_official_token_ce"] is True
    assert abs(float(a - b)) > 1e-4


def test_unknown_s_still_log_p_plus_n():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 8, requires_grad=True)
    pad = torch.ones(1, 8)
    z = torch.zeros(1, 8)
    not_s = torch.ones(1, 8)
    is_noise = torch.zeros(1)
    loss, _, _ = phase_balanced_partial_nll(
        logits, p_pos=z, s_pos=z, n_pos=z, not_p=z, not_s=not_s, pad_mask=pad, is_noise=is_noise
    )
    logp = F.log_softmax(logits.float(), dim=1)
    expect = -torch.logsumexp(torch.stack([logp[:, 0], logp[:, 2]], dim=-1), dim=-1).mean()
    assert abs(float(loss - expect)) < 1e-5


def test_phase_balanced_padding_zero_grad():
    logits = torch.randn(1, 3, 32, requires_grad=True)
    pad = torch.ones(1, 32)
    pad[:, 20:] = 0
    p_pos = torch.zeros(1, 32)
    p_pos[:, 5] = 1.0
    z = torch.zeros(1, 32)
    loss, _, _ = phase_balanced_partial_nll(
        logits, p_pos=p_pos, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad, is_noise=torch.zeros(1)
    )
    loss.backward()
    assert float(logits.grad[:, :, 20:].abs().sum()) == 0.0
    assert float(logits.grad[:, :, 5].abs().sum()) > 0


def test_event_background_not_certified_n_in_weights():
    w = build_partial_weights(
        n=50, p_c=None, s_c=None, vis_p=False, vis_s=False, is_noise=False,
        pad_mask=np.ones(50, np.float32), sigma=10.0, has_p_label=True, has_s_label=True,
    )
    assert float(w["n_pos"].sum()) == 0.0


def test_f1_zero_cannot_mint_valid_best():
    assert is_valid_best(metric=0.0, init_metric=-1.0) is False
    assert is_valid_best(metric=0.0, init_metric=0.0) is False
    assert is_valid_best(metric=0.02, init_metric=0.02) is False
    assert is_valid_best(metric=0.05, init_metric=0.02) is True


def test_frozen_cycle_sampler_reproducible():
    a = list(FrozenCycleSampler(10, seed=42))[:10]
    b = list(FrozenCycleSampler(10, seed=42))[:10]
    assert a == b
    c = list(FrozenCycleSampler(10, seed=7))[:10]
    assert a != c


def test_confirm_guard_still_unread():
    from pathlib import Path
    import json

    p = Path("artifacts/results/stage10/dkpn_v2/confirm_guard.json")
    d = json.loads(p.read_text())
    assert d.get("confirm_read") is False
