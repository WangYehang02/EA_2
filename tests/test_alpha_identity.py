from __future__ import annotations

import numpy as np

from earthquake.models.fixed_fusion import linear_fusion, log_space_fusion


def test_alpha_identity_linear():
    rng = np.random.default_rng(0)
    pn = rng.random(1200).astype(np.float32)
    prior = rng.random(1200).astype(np.float32)
    out = linear_fusion(pn, prior, alpha=1.0)
    assert np.max(np.abs(out - pn)) < 1e-6


def test_alpha_identity_log():
    rng = np.random.default_rng(1)
    pn = rng.random(1200).astype(np.float32)
    prior = rng.random(1200).astype(np.float32)
    out = log_space_fusion(pn, prior, alpha=1.0)
    # log-space with alpha=1 renormalizes; check argmax identity and high correlation
    assert int(out.argmax()) == int(pn.argmax())
    # renormalized pn
    pn_n = pn / pn.sum()
    assert np.max(np.abs(out - pn_n.astype(np.float32))) < 1e-5
