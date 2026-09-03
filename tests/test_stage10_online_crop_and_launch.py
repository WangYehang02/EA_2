"""Online crop scheduling, GPU launch policy, DDP shard tests (no HDF5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earthquake.stage10.crop_v2 import CROP_CYCLE, kinds_over_cycle, long_ps_sample_weight, scheduled_crop_kind, trace_schedule_hash
from earthquake.stage10.gpu_policy import MIN_GPUS, THROUGHPUT_GPUS, THROUGHPUT_STEPS, ddp_index_shard, launch_block_reason


def test_three_virtual_epochs_complete_ps_background_cycle():
    kinds = kinds_over_cycle("TRACE.A", has_p=True, has_s=True)
    assert set(kinds) == set(CROP_CYCLE)
    assert len(kinds) == 3
    # mixed schedule: not the old homogeneous epoch0=P,1=S,2=bg for every trace
    kinds_b = kinds_over_cycle("TRACE.B", has_p=True, has_s=True)
    assert set(kinds_b) == set(CROP_CYCLE)
    kinds2 = kinds_over_cycle("TRACE.A", has_p=True, has_s=True)
    assert kinds2 == kinds


def test_mixed_crop_each_epoch_has_all_kinds():
    names = [f"TR.{i:04d}" for i in range(300)]
    from collections import Counter

    for e in range(3):
        kinds = [
            scheduled_crop_kind(virtual_epoch=e, trace_name=n, is_noise=False, has_p_label=True, has_s_label=True)
            for n in names
        ]
        c = Counter(kinds)
        assert set(c) == set(CROP_CYCLE)
        for k in CROP_CYCLE:
            assert 60 <= c[k] <= 160, (e, k, c)


def test_ddp_crop_determined_globally_then_sharded():
    names = [f"TR.{i:04d}" for i in range(1000)]
    world = 4
    e = 1
    global_kinds = [
        scheduled_crop_kind(virtual_epoch=e, trace_name=n, is_noise=False, has_p_label=True, has_s_label=True)
        for n in names
    ]
    shards = []
    for r in range(world):
        idx = ddp_index_shard(len(names), r, world)
        shards.append([global_kinds[i] for i in idx])
        # rank must not re-roll: same as indexing global
        reroll = [
            scheduled_crop_kind(virtual_epoch=e, trace_name=names[i], is_noise=False, has_p_label=True, has_s_label=True)
            for i in idx
        ]
        assert reroll == shards[-1]
    # reconstruct
    recon = [None] * len(names)
    for r in range(world):
        for j, i in enumerate(ddp_index_shard(len(names), r, world)):
            recon[i] = shards[r][j]
    assert recon == global_kinds


def test_trace_hash_schedule_is_reproducible():
    a = trace_schedule_hash("IU.ANMO.00.BHZ")
    b = trace_schedule_hash("IU.ANMO.00.BHZ")
    assert a == b
    k1 = [scheduled_crop_kind(virtual_epoch=e, trace_name="PONLY.1", is_noise=False, has_p_label=True, has_s_label=False) for e in range(6)]
    k2 = [scheduled_crop_kind(virtual_epoch=e, trace_name="PONLY.1", is_noise=False, has_p_label=True, has_s_label=False) for e in range(6)]
    assert k1 == k2
    assert set(k1[:3]) <= {"p_centered", "background"}
    assert "p_centered" in k1[:3]


def test_long_ps_does_not_drop_p_or_s_supervision():
    p, s = 1000.0, 1000.0 + 45 * 100  # 45 s
    assert long_ps_sample_weight(p, s) == 2
    kinds = kinds_over_cycle("LONG.PS", has_p=True, has_s=True)
    assert "p_centered" in kinds and "s_centered" in kinds


def test_ddp_ranks_do_not_heavily_overlap():
    n = 10_000
    w = 8
    shards = [set(ddp_index_shard(n, r, w)) for r in range(w)]
    for i in range(w):
        for j in range(i + 1, w):
            assert shards[i].isdisjoint(shards[j])
    assert set.union(*shards) == set(range(n))


def test_full_train_refused_below_min_gpus():
    assert MIN_GPUS == 4
    assert THROUGHPUT_STEPS == 50
    assert THROUGHPUT_GPUS == 1
    assert launch_block_reason(mode="pilot", n_idle=1, already_training=False, train_running_flag=False) is not None
    assert launch_block_reason(mode="train", n_idle=3, already_training=False, train_running_flag=False) == "need_min_gpus_4_have_3"
    assert launch_block_reason(mode="pilot", n_idle=4, already_training=False, train_running_flag=False) is None
    assert launch_block_reason(mode="throughput", n_idle=1, already_training=False, train_running_flag=False) is None
    assert launch_block_reason(mode="throughput", n_idle=0, already_training=False, train_running_flag=False) is not None


def test_launcher_refuses_if_training_already_running():
    r = launch_block_reason(mode="pilot", n_idle=8, already_training=True, train_running_flag=False)
    assert r == "training_already_running"
    r2 = launch_block_reason(mode="train", n_idle=8, already_training=False, train_running_flag=True)
    assert r2 == "training_already_running"


def test_online_catalog_does_not_triple_expand():
    from earthquake.stage10.dataset_v2 import annotate_online_catalog, expand_crop_catalog

    meta = pd.DataFrame(
        {
            "trace_name": ["t0", "t1"],
            "event_id": ["e0", "e1"],
            "p_arrival_sample": [1000.0, 800.0],
            "s_arrival_sample": [1500.0, 800.0 + 4000.0],
            "is_noise": [False, False],
        }
    )
    on = annotate_online_catalog(meta)
    assert len(on) == 2
    assert "crop_kind" not in on.columns
    assert int(on.loc[1, "sample_weight"]) == 2
    ex = expand_crop_catalog(meta)
    assert len(ex) == 6
