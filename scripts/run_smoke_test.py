#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "scripts/inspect_instance.py"])
    run([py, "scripts/build_index.py", "--max-events", "200"])
    run([py, "scripts/analyze_history_coverage.py", "--max-events", "200"])
    run([py, "scripts/build_history.py", "--max-events", "200", "--protocol", "frozen"])
    run([py, "scripts/run_phasenet.py", "--config", "configs/debug.yaml"])
    run([py, "scripts/evaluate_phasenet.py", "--config", "configs/debug.yaml"])
    run([py, "scripts/evaluate_fixed_fusion.py", "--config", "configs/debug.yaml"])
    run([py, "-m", "pytest", "-q"])
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
