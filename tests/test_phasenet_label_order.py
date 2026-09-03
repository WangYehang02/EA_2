"""Finetune soft labels must match SeisBench PhaseNet.labels channel order."""

from __future__ import annotations

import numpy as np

from earthquake.models.phasenet_finetune import soft_phase_targets


def test_soft_targets_psn_order_matches_stead():
    y = soft_phase_targets(1000, p_sample=200.0, s_sample=500.0, sampling_rate=100.0, label_order="PSN")
    assert y.shape == (3, 1000)
    assert y[:, 200].argmax() == 0  # P
    assert y[:, 500].argmax() == 1  # S
    assert y[:, 800].argmax() == 2  # N
    np.testing.assert_allclose(y.sum(axis=0), 1.0, atol=1e-5)


def test_soft_targets_nps_order():
    y = soft_phase_targets(1000, p_sample=200.0, s_sample=500.0, sampling_rate=100.0, label_order="NPS")
    assert y[:, 200].argmax() == 1  # P
    assert y[:, 500].argmax() == 2  # S
    assert y[:, 800].argmax() == 0  # N
