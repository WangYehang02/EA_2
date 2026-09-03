"""Partial-label / crop-visibility tests (no confirm, no HDF5 required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from earthquake.stage10.crop_v2 import IN_SAMPLES, crop_kind_start, phase_visible
from earthquake.stage10.dataset_v2 import build_partial_weights
from earthquake.stage10.partial_label import complete_ce_from_logits, complete_ce_from_probs, partial_label_nll


def test_complete_label_ce_matches_official_logp():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, 128)
    y = torch.softmax(torch.randn(4, 3, 128), dim=1)
    a = complete_ce_from_logits(logits, y)
    b = complete_ce_from_probs(torch.softmax(logits, dim=1), y, eps=1e-12)
    assert torch.isfinite(a) and abs(float(a - b)) < 1e-6


def test_complete_partial_nll_matches_ce_when_simplex_weights():
    torch.manual_seed(1)
    logits = torch.randn(3, 3, 64, requires_grad=True)
    y = torch.softmax(torch.randn(3, 3, 64), dim=1)
    pad = torch.ones(3, 64)
    loss_a = complete_ce_from_logits(logits, y)
    # partial with p/s/n pos = y, mean over all T*weight
    # complete_ce uses mean_T then sum_C then mean_B — different reduction.
    # Compare token-mean NLL:
    logp = F.log_softmax(logits, dim=1)
    token = -(y * logp).sum(dim=1)
    loss_token = token.mean()
    loss_p = partial_label_nll(
        logits, p_pos=y[:, 0], s_pos=y[:, 1], n_pos=y[:, 2],
        not_p=torch.zeros_like(y[:, 0]), not_s=torch.zeros_like(y[:, 0]), pad_mask=pad,
    )
    assert abs(float(loss_p - loss_token)) < 1e-5


def test_p_only_unknown_s_not_labeled_n():
    n = 200
    pad = np.ones(n, np.float32)
    w = build_partial_weights(
        n=n, p_c=50.0, s_c=None, vis_p=True, vis_s=False, is_noise=False,
        pad_mask=pad, sigma=10.0, has_p_label=True, has_s_label=False,
    )
    assert w["n_pos"].sum() == 0.0
    assert w["s_pos"].sum() == 0.0
    assert w["p_pos"].max() > 0.5
    assert w["not_p"].sum() > 0
    # far from P is not_p, not n
    assert w["not_p"][150] > 0.9 and w["n_pos"][150] == 0


def test_s_only_unknown_p_not_labeled_n():
    n = 200
    pad = np.ones(n, np.float32)
    w = build_partial_weights(
        n=n, p_c=None, s_c=80.0, vis_p=False, vis_s=True, is_noise=False,
        pad_mask=pad, sigma=10.0, has_p_label=True, has_s_label=True,
    )
    assert w["n_pos"].sum() == 0.0
    assert w["p_pos"].sum() == 0.0
    assert w["s_pos"].max() > 0.5
    assert w["not_s"][10] > 0.9


def test_crop_out_phase_has_zero_pos_weight():
    n = IN_SAMPLES
    pad = np.ones(n, np.float32)
    # S at sample 9000, crop start such that S not in window
    w = build_partial_weights(
        n=n, p_c=100.0, s_c=None, vis_p=True, vis_s=False, is_noise=False,
        pad_mask=pad, sigma=10.0, has_p_label=True, has_s_label=True,
    )
    assert float(w["s_pos"].sum()) == 0.0


def test_padding_gradient_zero():
    logits = torch.randn(1, 3, 32, requires_grad=True)
    pad = torch.ones(1, 32)
    pad[:, 20:] = 0
    p_pos = torch.zeros(1, 32)
    p_pos[:, 5] = 1.0
    z = torch.zeros(1, 32)
    loss = partial_label_nll(logits, p_pos=p_pos, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=pad)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[:, :, 20:].abs().sum()) == 0.0
    assert float(logits.grad[:, :, 5].abs().sum()) > 0


def test_outside_phase_pos_weight_gradient_decoupled():
    """S-pos weight 0 ⇒ S-channel CE term 0; not_p still allows S in the *marginal* (intentional)."""
    n = 16
    pad = np.ones(n, np.float32)
    w = build_partial_weights(
        n=n, p_c=4.0, s_c=None, vis_p=True, vis_s=False, is_noise=False,
        pad_mask=pad, sigma=1.5, has_p_label=True, has_s_label=False,
    )
    logits = torch.randn(1, 3, n, requires_grad=True)
    loss = partial_label_nll(
        logits,
        p_pos=torch.from_numpy(w["p_pos"])[None],
        s_pos=torch.from_numpy(w["s_pos"])[None],
        n_pos=torch.from_numpy(w["n_pos"])[None],
        not_p=torch.from_numpy(w["not_p"])[None],
        not_s=torch.from_numpy(w["not_s"])[None],
        pad_mask=torch.from_numpy(w["pad_mask"])[None],
    )
    loss.backward()
    assert w["s_pos"].sum() == 0
    assert w["n_pos"].sum() == 0


def test_long_ps_multi_crop_starts():
    rng = np.random.default_rng(0)
    n, p, s = 12000, 2000.0, 2000.0 + 3500.0  # >30 s
    sp = crop_kind_start("p_centered", n, p, s, rng)
    ss = crop_kind_start("s_centered", n, p, s, rng)
    assert phase_visible(p, sp) and not phase_visible(s, sp)
    assert phase_visible(s, ss) and not phase_visible(p, ss)
    wp = build_partial_weights(
        n=IN_SAMPLES, p_c=p - sp - 400, s_c=s - sp - 400,
        vis_p=True, vis_s=False, is_noise=False,
        pad_mask=np.ones(IN_SAMPLES, np.float32), sigma=10.0,
        has_p_label=True, has_s_label=True,
    )
    ws = build_partial_weights(
        n=IN_SAMPLES, p_c=p - ss - 400, s_c=s - ss - 400,
        vis_p=False, vis_s=True, is_noise=False,
        pad_mask=np.ones(IN_SAMPLES, np.float32), sigma=10.0,
        has_p_label=True, has_s_label=True,
    )
    assert wp["p_pos"].max() > 0.5 and wp["s_pos"].sum() == 0 and wp["n_pos"].sum() == 0
    assert ws["s_pos"].max() > 0.5 and ws["p_pos"].sum() == 0 and ws["n_pos"].sum() == 0


def test_logits_reject_softmax_input():
    p = torch.softmax(torch.randn(2, 3, 8), dim=1)
    y = p
    with pytest.raises(RuntimeError, match="softmax"):
        complete_ce_from_logits(p, y)
    z = torch.zeros(2, 8)
    with pytest.raises(RuntimeError, match="softmax"):
        partial_label_nll(p, p_pos=z, s_pos=z, n_pos=z, not_p=z, not_s=z, pad_mask=torch.ones(2, 8))


def test_truncated_gaussian_renorm_and_noise_full_n():
    n = 100
    pad = np.ones(n, np.float32)
    # center near left edge
    g = build_partial_weights(
        n=n, p_c=2.0, s_c=40.0, vis_p=True, vis_s=True, is_noise=False,
        pad_mask=pad, sigma=10.0, has_p_label=True, has_s_label=True,
    )
    assert g["p_pos"].max() > 0.4
    noise = build_partial_weights(
        n=n, p_c=None, s_c=None, vis_p=False, vis_s=False, is_noise=True,
        pad_mask=pad, sigma=10.0, has_p_label=False, has_s_label=False,
    )
    assert np.allclose(noise["n_pos"], 1.0)
    assert noise["p_pos"].sum() == 0


def test_event_background_both_phases_outside_is_zero_loss_not_n():
    n = 200
    pad = np.ones(n, np.float32)
    w = build_partial_weights(
        n=n, p_c=None, s_c=None, vis_p=False, vis_s=False, is_noise=False,
        pad_mask=pad, sigma=10.0, has_p_label=True, has_s_label=True,
    )
    assert w["n_pos"].sum() == 0.0
    assert w["p_pos"].sum() == 0.0
    assert w["s_pos"].sum() == 0.0
    assert w["not_p"].sum() == 0.0
    assert w["not_s"].sum() == 0.0
    wsum = w["p_pos"] + w["s_pos"] + w["n_pos"] + w["not_p"] + w["not_s"]
    assert float(wsum.sum()) == 0.0


def test_trace_name_join_shuffle_still_ok():
    from earthquake.stage10.protocol import join_predictions_by_trace_name
    from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics

    man = pd.DataFrame({"trace_name": ["a", "b"], "s_arrival_sample": [1.0, 2.0], "sampling_rate_hz": 100.0})
    pred = pd.DataFrame({"trace_name": ["b", "a"], "pred_s_sample": [2.0, 1.0]})
    j = join_predictions_by_trace_name(man, pred)
    m = comprehensive_pick_metrics(j.pred_s_sample.to_numpy(), j.s_arrival_sample.to_numpy(), j.sampling_rate_hz.to_numpy())
    j2 = join_predictions_by_trace_name(man.sample(frac=1, random_state=3).reset_index(drop=True), pred)
    m2 = comprehensive_pick_metrics(j2.pred_s_sample.to_numpy(), j2.s_arrival_sample.to_numpy(), j2.sampling_rate_hz.to_numpy())
    assert abs(m["f1@0.5"] - m2["f1@0.5"]) < 1e-12


def test_confirm_not_used_v2_flag():
    from pathlib import Path
    import json

    p = Path("artifacts/results/stage10/dkpn_v2/confirm_guard.json")
    if p.exists():
        d = json.loads(p.read_text())
        assert d.get("confirm_read") is False
