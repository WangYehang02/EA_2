#!/usr/bin/env python
"""GPU watchdog for one-shot scalar_pairwise confirm evaluation.

Hard-coded command. Never launches full-dev, waveform, or arbitrary shell.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "results" / "pairwise_confirm"
LOG_DIR = OUT / "logs"
LOCK_PATH = OUT / "watchdog.lock"
EVAL_SCRIPT = ROOT / "scripts" / "eval_scalar_pairwise_confirm.py"
EVAL_CMD = [sys.executable, str(EVAL_SCRIPT), "--device", "auto", "--n-boot", "5000", "--workers", "8"]


@dataclass
class GPUInfo:
    index: int
    uuid: str
    free_mb: float
    used_mb: float
    util: float


def query_gpus() -> list[GPUInfo]:
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    out = []
    for line in smi.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        out.append(
            GPUInfo(index=int(parts[0]), uuid=parts[1], free_mb=float(parts[2]), used_mb=float(parts[3]), util=float(parts[4]))
        )
    return out


def select_gpu(gpus: list[GPUInfo], *, min_free_mb: float, max_util: float, max_used_mb: float) -> GPUInfo | None:
    cands = [g for g in gpus if g.util <= max_util and g.used_mb <= max_used_mb and g.free_mb >= min_free_mb]
    if not cands:
        return None
    cands.sort(key=lambda g: (-g.free_mb, g.util, g.index))
    return cands[0]


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def final_status() -> str | None:
    for s in ["CONFIRMED", "PARTIALLY_CONFIRMED", "NOT_CONFIRMED"]:
        if (OUT / f"SCALAR_PAIRWISE.CONFIRM.{s}").exists():
            return s
    if (OUT / "SCALAR_PAIRWISE.CONFIRM.FAILED").exists():
        return "FAILED"
    return None


def running_alive() -> bool:
    p = OUT / "SCALAR_PAIRWISE.CONFIRM.RUNNING"
    if not p.exists():
        return False
    try:
        pid = int(json.loads(p.read_text()).get("pid", -1))
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def preflight() -> list[str]:
    errs = []
    if not (OUT / "SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.json").exists():
        errs.append("execution lock missing")
    if not (ROOT / "artifacts/results/pairwise_fulldev/SCALAR_PAIRWISE.FULLDEV.PASSED").exists():
        errs.append("FULLDEV.PASSED missing")
    cmd = " ".join(EVAL_CMD).lower()
    if "fulldev" in cmd and "confirm" not in cmd:
        errs.append("wrong command")
    # must be confirm evaluator
    if "eval_scalar_pairwise_confirm" not in cmd:
        errs.append("confirm eval not in command")
    if "confirm" not in Path(EVAL_CMD[1]).name:
        errs.append("non-confirm script")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-mb", type=float, default=4000)
    ap.add_argument("--max-util", type=float, default=10)
    ap.add_argument("--max-used-mb", type=float, default=1500)
    ap.add_argument("--poll-seconds", type=float, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log("flock held; exit")
        return 0

    idle = 0
    try:
        while True:
            st = final_status()
            if st in {"CONFIRMED", "PARTIALLY_CONFIRMED", "NOT_CONFIRMED"}:
                _log(f"already {st}; exit 0")
                return 0
            if st == "FAILED":
                _log("FAILED present; manual intervention required")
                return 2
            if running_alive():
                _log("RUNNING alive; no duplicate")
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            errs = preflight()
            if errs:
                _log(f"preflight failed: {errs}")
                return 1
            gpus = query_gpus()
            g = select_gpu(gpus, min_free_mb=args.min_free_mb, max_util=args.max_util, max_used_mb=args.max_used_mb)
            if g is None:
                idle += 1
                if idle == 1 or idle % 10 == 0:
                    _log("no eligible GPU; " + " | ".join(f"{x.index}:free={x.free_mb:.0f},util={x.util}" for x in gpus))
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            # recheck race
            g2 = select_gpu(query_gpus(), min_free_mb=args.min_free_mb, max_util=args.max_util, max_used_mb=args.max_used_mb)
            if g2 is None or g2.index != g.index:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            if args.dry_run:
                _log(f"DRY-RUN would launch confirm on GPU {g2.index} uuid={g2.uuid}")
                return 0
            run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(g2.index)
            _log(f"LAUNCH GPU={g2.index} uuid={g2.uuid} cmd={' '.join(EVAL_CMD)}")
            out_p = LOG_DIR / f"confirm_{run_id}.stdout.log"
            err_p = LOG_DIR / f"confirm_{run_id}.stderr.log"
            with open(out_p, "w") as out, open(err_p, "w") as err:
                proc = subprocess.Popen(EVAL_CMD, cwd=str(ROOT), env=env, stdout=out, stderr=err)
                _log(f"LAUNCH_PID={proc.pid}")
                rc = proc.wait()
            _log(f"EXIT rc={rc} status={final_status()}")
            return 0 if final_status() in {"CONFIRMED", "PARTIALLY_CONFIRMED", "NOT_CONFIRMED"} else 1
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
