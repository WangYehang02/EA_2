"""GPU idle policy for DKPN v2. Never preempt or kill other jobs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_GPUS = 4
PREFERRED_GPUS = 8
THROUGHPUT_STEPS = 50
THROUGHPUT_GPUS = 1
MEM_USED_MAX_MIB = 500
UTIL_MAX_PCT = 2  # "接近 0"


@dataclass
class GpuSnap:
    index: int
    memory_used_mib: float
    utilization_pct: float
    n_compute_procs: int
    compute_users: list[str] = field(default_factory=list)
    name: str = ""

    @property
    def idle(self) -> bool:
        return (
            self.memory_used_mib < MEM_USED_MAX_MIB
            and self.utilization_pct <= UTIL_MAX_PCT
            and self.n_compute_procs == 0
        )


def ddp_index_shard(n: int, rank: int, world_size: int) -> list[int]:
    """Deterministic disjoint shard (no overlap across ranks)."""
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError((rank, world_size))
    return list(range(int(rank), int(n), int(world_size)))


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return ""
    return r.stdout


def snapshot_gpus() -> list[GpuSnap]:
    q = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    snaps: dict[int, GpuSnap] = {}
    if not q.strip():
        return []
    for line in q.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        snaps[idx] = GpuSnap(
            index=idx,
            name=parts[1],
            memory_used_mib=float(parts[2]),
            utilization_pct=float(parts[3]),
            n_compute_procs=0,
        )
    apps = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,used_memory,process_name",
            "--format=csv,noheader",
        ]
    )
    uuid_to_idx = {}
    uq = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    for line in uq.strip().splitlines():
        a, b = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[b] = int(a)
    if apps.strip():
        for line in apps.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            uuid = parts[0]
            idx = uuid_to_idx.get(uuid)
            if idx is None:
                continue
            snaps[idx].n_compute_procs += 1
            if len(parts) >= 5:
                snaps[idx].compute_users.append(parts[4])
    return [snaps[k] for k in sorted(snaps)]


def idle_gpu_indices(snaps: list[GpuSnap] | None = None) -> list[int]:
    snaps = snaps if snaps is not None else snapshot_gpus()
    return [g.index for g in snaps if g.idle]


def allocatable_gpu_indices(snaps: list[GpuSnap] | None = None) -> list[int]:
    """Idle cards, plus leftover display/compositor (<500 MiB, util≈0, not python/vLLM).

    Does not treat another user's training/vLLM job as free. Does not kill anyone.
    """
    snaps = snaps if snaps is not None else snapshot_gpus()
    out = []
    for g in snaps:
        if g.idle:
            out.append(g.index)
            continue
        if g.memory_used_mib >= MEM_USED_MAX_MIB or g.utilization_pct > UTIL_MAX_PCT:
            continue
        users = " ".join(g.compute_users).lower()
        if any(tok in users for tok in ("vllm", "pt_data_worker")):
            continue
        if "python" in users and "ptyxis" not in users:
            continue
        out.append(g.index)
    return out


def training_process_running(pattern: str = "train_stage10_dkpn_v") -> list[str]:
    """True training/pilot only — throughput/status/smoke/shell wrappers do not count.

    Matches v2 (`train_stage10_dkpn_v2.py`) and v3 (`train_stage10_dkpn_v3.py`).
    """
    out = _run(["ps", "-eo", "pid,args"])
    lines = []
    mypid = str(os.getpid())
    for line in out.splitlines():
        if pattern not in line:
            continue
        if "wait_and_launch" in line or "launch_dkpn_v2" in line:
            continue
        toks = line.strip().split()
        if not toks:
            continue
        if toks[0] == mypid:
            continue
        exe = toks[1] if len(toks) > 1 else ""
        if "python" not in exe:
            continue
        joined = " ".join(toks)
        if "--mode train" not in joined and "--mode pilot" not in joined and "--mode resume" not in joined:
            continue
        lines.append(line.strip())
    return lines


def launch_block_reason(
    *,
    mode: str,
    n_idle: int,
    already_training: bool,
    train_running_flag: bool,
    min_gpus: int = MIN_GPUS,
) -> str | None:
    """Return a refusal string, or None if launch is allowed."""
    if already_training or train_running_flag:
        return "training_already_running"
    if mode in {"train", "pilot", "full", "resume"}:
        if n_idle < min_gpus:
            return f"need_min_gpus_{min_gpus}_have_{n_idle}"
        return None
    if mode == "throughput":
        if n_idle < 1:
            return "need_min_gpus_1_have_0"
        return None
    return None


def status_dict(
    *,
    out_dir: Path,
    snaps: list[GpuSnap] | None = None,
    watcher_only: bool | None = None,
) -> dict[str, Any]:
    snaps = snaps if snaps is not None else snapshot_gpus()
    idle = [g.index for g in snaps if g.idle]
    train_lines = training_process_running()
    train_flag = (out_dir / "TRAIN.RUNNING").is_file()
    pid_file = out_dir / "TRAIN.PID"
    return {
        "min_gpus": MIN_GPUS,
        "preferred_gpus": PREFERRED_GPUS,
        "throughput_steps": THROUGHPUT_STEPS,
        "throughput_gpus": THROUGHPUT_GPUS,
        "mem_used_max_mib": MEM_USED_MAX_MIB,
        "util_max_pct": UTIL_MAX_PCT,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "memory_used_mib": g.memory_used_mib,
                "utilization_pct": g.utilization_pct,
                "n_compute_procs": g.n_compute_procs,
                "idle": g.idle,
            }
            for g in snaps
        ],
        "idle_gpu_indices": idle,
        "n_idle": len(idle),
        "launch_ok_full_or_pilot": len(idle) >= MIN_GPUS,
        "launch_ok_1gpu_throughput_only": len(idle) >= 1,
        "watcher_only": bool(watcher_only if watcher_only is not None else not train_lines),
        "training_subprocesses": train_lines,
        "training_subprocess_present": bool(train_lines) or train_flag,
        "TRAIN.RUNNING": train_flag,
        "TRAIN.PID": pid_file.read_text().strip() if pid_file.is_file() else None,
        "never_preempt": True,
    }
