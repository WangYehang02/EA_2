"""Watchdog unit tests with mocked GPU query (no real GPU required)."""

from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import watch_pairwise_fulldev_gpu as wd


def _gpu(i, free=8000, used=100, util=0, uuid=None):
    return wd.GPUInfo(
        index=i,
        uuid=uuid or f"GPU-{i}",
        total_mb=24000,
        free_mb=free,
        used_mb=used,
        util=util,
        mem_util=0,
        n_procs=0,
    )


def test_no_gpu_no_select():
    assert wd.select_gpu([], min_free_mb=4000, max_util=10, max_used_mb=1500) is None


def test_all_busy_no_select():
    gpus = [_gpu(0, free=20000, used=2000, util=50), _gpu(1, free=500, used=20000, util=0)]
    assert wd.select_gpu(gpus, min_free_mb=4000, max_util=10, max_used_mb=1500) is None


def test_one_idle_selected():
    gpus = [_gpu(0, free=1000, used=20000, util=0), _gpu(3, free=9000, used=100, util=2)]
    g = wd.select_gpu(gpus, min_free_mb=4000, max_util=10, max_used_mb=1500)
    assert g is not None and g.index == 3


def test_multi_idle_prefers_most_free():
    gpus = [_gpu(1, free=5000, used=100, util=0), _gpu(2, free=12000, used=50, util=1)]
    g = wd.select_gpu(gpus, min_free_mb=4000, max_util=10, max_used_mb=1500)
    assert g.index == 2


def test_passed_exits(tmp_path, monkeypatch):
    out = tmp_path / "pairwise_fulldev"
    out.mkdir()
    (out / "SCALAR_PAIRWISE.FULLDEV.PASSED").write_text("ok\n")
    monkeypatch.setattr(wd, "OUT", out)
    monkeypatch.setattr(wd, "LOG_DIR", out / "logs")
    monkeypatch.setattr(wd, "LOCK_PATH", out / "watchdog.lock")
    rc = wd.main(["--once", "--dry-run"])
    assert rc == 0


def test_failed_no_retry(tmp_path, monkeypatch):
    out = tmp_path / "pairwise_fulldev"
    out.mkdir()
    (out / "SCALAR_PAIRWISE.FULLDEV.FAILED").write_text("boom\n")
    monkeypatch.setattr(wd, "OUT", out)
    monkeypatch.setattr(wd, "LOG_DIR", out / "logs")
    monkeypatch.setattr(wd, "LOCK_PATH", out / "watchdog.lock")
    rc = wd.main(["--once"])
    assert rc == 2


def test_dry_run_does_not_launch(tmp_path, monkeypatch):
    out = tmp_path / "pairwise_fulldev"
    out.mkdir()
    # minimal lock for preflight
    lock = {
        "calibration_tau": 0.5,
        "checkpoint_path": str(tmp_path / "ckpt.pt"),
        "checkpoint_sha256": "abc",
        "lock_body_sha256": "x",
    }
    # create fake ckpt matching sha
    import hashlib

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"fake")
    lock["checkpoint_sha256"] = hashlib.sha256(b"fake").hexdigest()
    (out / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json").write_text(json.dumps(lock))
    (out / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256").write_text(lock["lock_body_sha256"] + "\n")
    # eval script path — point to real or create dummy
    monkeypatch.setattr(wd, "OUT", out)
    monkeypatch.setattr(wd, "LOG_DIR", out / "logs")
    monkeypatch.setattr(wd, "LOCK_PATH", out / "watchdog.lock")
    monkeypatch.setattr(wd, "EVAL_SCRIPT", ROOT / "scripts" / "eval_scalar_pairwise_fulldev.py")
    monkeypatch.setattr(wd, "query_gpus", lambda: [_gpu(7, free=20000, used=20, util=0)])
    launched = []

    def boom(*a, **k):
        launched.append(1)
        raise AssertionError("should not launch in dry-run")

    monkeypatch.setattr(wd, "launch_eval", boom)
    rc = wd.main(["--once", "--dry-run"])
    assert rc == 0
    assert launched == []


def test_flock_blocks_second(tmp_path, monkeypatch):
    out = tmp_path / "pairwise_fulldev"
    out.mkdir()
    lock_path = out / "watchdog.lock"
    monkeypatch.setattr(wd, "OUT", out)
    monkeypatch.setattr(wd, "LOG_DIR", out / "logs")
    monkeypatch.setattr(wd, "LOCK_PATH", lock_path)
    f = open(lock_path, "a+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc = wd.main(["--once", "--dry-run"])
        assert rc == 0  # second watchdog exits quietly
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def test_eval_command_has_no_confirm():
    cmd = " ".join(wd.EVAL_MODULE_CMD).lower()
    assert "confirm" not in cmd
    assert "eval_scalar_pairwise_fulldev" in cmd
