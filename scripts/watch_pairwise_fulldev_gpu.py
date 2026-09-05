#!/usr/bin/env python
"""GPU watchdog for frozen scalar_pairwise phaseB full-dev.

Hard-coded command only. Never launches confirm / waveform / arbitrary shell.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "results" / "pairwise_fulldev"
LOG_DIR = OUT / "logs"
LOCK_PATH = OUT / "watchdog.lock"
EVAL_SCRIPT = ROOT / "scripts" / "eval_scalar_pairwise_fulldev.py"

# Hard-coded — no user-injected command.
EVAL_MODULE_CMD = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--device",
    "auto",
]


@dataclass
class GPUInfo:
    index: int
    uuid: str
    total_mb: float
    free_mb: float
    used_mb: float
    util: float
    mem_util: float
    n_procs: int


def query_gpus() -> list[GPUInfo]:
    """Query nvidia-smi. Separated for unit-test mocking."""
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    # process counts
    try:
        procs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            text=True,
        )
        uuid_counts: dict[str, int] = {}
        for line in procs.strip().splitlines():
            u = line.strip()
            if u:
                uuid_counts[u] = uuid_counts.get(u, 0) + 1
    except Exception:
        uuid_counts = {}

    out: list[GPUInfo] = []
    for line in smi.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, uuid, tot, free, used, util, mem_util = parts[:7]
        out.append(
            GPUInfo(
                index=int(idx),
                uuid=uuid,
                total_mb=float(tot),
                free_mb=float(free),
                used_mb=float(used),
                util=float(util),
                mem_util=float(mem_util),
                n_procs=int(uuid_counts.get(uuid, 0)),
            )
        )
    return out


def select_gpu(
    gpus: list[GPUInfo],
    *,
    min_free_mb: float,
    max_util: float,
    max_used_mb: float,
) -> GPUInfo | None:
    cands = [
        g
        for g in gpus
        if g.util <= max_util and g.used_mb <= max_used_mb and g.free_mb >= min_free_mb
    ]
    if not cands:
        return None
    # free desc, util asc, index asc
    cands.sort(key=lambda g: (-g.free_mb, g.util, g.index))
    return cands[0]


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def completion_status() -> str | None:
    if (OUT / "SCALAR_PAIRWISE.FULLDEV.PASSED").exists():
        return "PASSED"
    if (OUT / "SCALAR_PAIRWISE.FULLDEV.NO_GO").exists():
        return "NO_GO"
    if (OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED").exists():
        return "FAILED"
    return None


def is_running_alive() -> bool:
    marker = OUT / "SCALAR_PAIRWISE.FULLDEV.RUNNING"
    if not marker.exists():
        return False
    try:
        meta = json.loads(marker.read_text())
        pid = int(meta.get("pid", -1))
        if pid > 0:
            os.kill(pid, 0)
            return True
    except Exception:
        # stale marker
        return False
    return False


def preflight() -> list[str]:
    errors = []
    lock = OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    sha = OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256"
    if not lock.exists():
        errors.append("METHOD_LOCK missing")
        return errors
    data = json.loads(lock.read_text())
    if abs(float(data.get("calibration_tau", -1)) - 0.50) > 1e-12:
        errors.append("tau != 0.50")
    ckpt = Path(data["checkpoint_path"])
    if not ckpt.exists():
        errors.append("checkpoint missing")
    else:
        import hashlib

        h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if h != data.get("checkpoint_sha256"):
            errors.append("checkpoint sha mismatch")
    if sha.exists():
        if data.get("lock_body_sha256") and data["lock_body_sha256"] != sha.read_text().strip():
            errors.append("lock sha file mismatch")
    # confirm must not appear in eval argv construction
    cmd_s = " ".join(EVAL_MODULE_CMD).lower()
    if "confirm" in cmd_s:
        errors.append("confirm leaked into eval command")
    if not EVAL_SCRIPT.exists():
        errors.append("eval script missing")
    return errors


def launch_eval(gpu: GPUInfo, run_id: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    stdout_p = LOG_DIR / f"fulldev_{run_id}.stdout.log"
    stderr_p = LOG_DIR / f"fulldev_{run_id}.stderr.log"
    _log(
        f"LAUNCH physical_gpu={gpu.index} uuid={gpu.uuid} "
        f"free_mb={gpu.free_mb:.0f} util={gpu.util} CUDA_VISIBLE_DEVICES={gpu.index} "
        f"cmd={' '.join(EVAL_MODULE_CMD)}"
    )
    with open(stdout_p, "w", encoding="utf-8") as out, open(stderr_p, "w", encoding="utf-8") as err:
        proc = subprocess.Popen(EVAL_MODULE_CMD, cwd=str(ROOT), env=env, stdout=out, stderr=err)
        _log(f"LAUNCH_PID={proc.pid}")
        rc = proc.wait()
    _log(f"EXIT code={rc} run_id={run_id}")
    return int(rc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-mb", type=float, default=4000.0)
    ap.add_argument("--max-util", type=float, default=10.0)
    ap.add_argument("--max-used-mb", type=float, default=1500.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed-once", action="store_true", default=False)
    ap.add_argument("--query-gpus-fn", default=None, help=argparse.SUPPRESS)  # tests inject via monkeypatch
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # OS-level flock
    lock_f = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log("another watchdog holds flock; exit")
        return 0

    retried = False
    idle_polls = 0
    try:
        while True:
            st = completion_status()
            if st in {"PASSED", "NO_GO"}:
                _log(f"already {st}; exit 0")
                return 0
            if st == "FAILED":
                if args.retry_failed_once and not retried:
                    _log("FAILED present; retry-failed-once enabled — clearing FAILED markers once")
                    (OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED").unlink(missing_ok=True)
                    (OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED.json").unlink(missing_ok=True)
                    retried = True
                else:
                    _log("FAILED present; manual intervention required; exit 2")
                    return 2

            if is_running_alive():
                _log("RUNNING alive; not launching duplicate")
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue

            errs = preflight()
            if errs:
                _log(f"preflight failed: {errs}")
                if not args.dry_run:
                    (OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED").write_text(
                        "preflight: " + "; ".join(errs) + "\n", encoding="utf-8"
                    )
                return 1

            gpus = query_gpus()
            chosen = select_gpu(
                gpus,
                min_free_mb=args.min_free_mb,
                max_util=args.max_util,
                max_used_mb=args.max_used_mb,
            )
            if chosen is None:
                idle_polls += 1
                if idle_polls == 1 or idle_polls % 10 == 0:
                    summary = [
                        f"gpu{g.index}: free={g.free_mb:.0f} used={g.used_mb:.0f} util={g.util}"
                        for g in gpus
                    ]
                    _log(f"no eligible GPU (polls={idle_polls}); " + " | ".join(summary))
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue

            # race: re-query
            gpus2 = query_gpus()
            chosen2 = select_gpu(
                gpus2,
                min_free_mb=args.min_free_mb,
                max_util=args.max_util,
                max_used_mb=args.max_used_mb,
            )
            if chosen2 is None or chosen2.index != chosen.index:
                _log("GPU race lost; retry")
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue

            if args.dry_run:
                _log(f"DRY-RUN would launch on GPU {chosen2.index} uuid={chosen2.uuid}")
                return 0

            run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            rc = launch_eval(chosen2, run_id)
            st2 = completion_status()
            _log(f"post-launch status={st2} rc={rc}")
            if st2 in {"PASSED", "NO_GO"}:
                return 0
            if st2 == "FAILED" or rc != 0:
                _log("eval failed; not auto-retrying (default)")
                return 1
            return rc
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
