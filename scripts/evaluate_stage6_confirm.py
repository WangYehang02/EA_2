#!/usr/bin/env python
"""One-shot internal confirm eval — requires method_lock_stage6.json."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthquake.stage6.splits import stage6_paths

def main():
    lock = stage6_paths()["method_lock"]
    if not lock.exists():
        raise SystemExit("Refuse confirm eval: method_lock_stage6.json missing")
    print("Confirm eval not fully implemented yet — method is locked gate only for now.")

if __name__ == "__main__":
    main()
