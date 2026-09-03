"""Read-only audit of v3 checkpoints before exact resume."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

REQUIRED_KEYS = ("model", "opt", "optimizer_steps", "epoch", "virtual_epoch")
SEMANTIC_HASH_KEYS = (
    "partial_label_sha256",
    "dataset_v2_sha256",
    "crop_v2_sha256",
    "train_catalog_sha256",
    "val_subset_sha256",
    "config_sha256",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def audit_resume_checkpoint(path: Path, *, expect_epoch: int | None = 2) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "block": True, "reason": "checkpoint_missing", "path": str(path)}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return {"ok": False, "block": True, "reason": "checkpoint_not_dict", "path": str(path)}
    keys = sorted(ck.keys())
    missing = [k for k in REQUIRED_KEYS if k not in ck]
    has_model = isinstance(ck.get("model"), dict) and len(ck.get("model") or {}) > 0
    opt = ck.get("opt")
    has_opt = isinstance(opt, dict) and "state" in opt and "param_groups" in opt and len(opt.get("state") or {}) > 0
    epoch = ck.get("virtual_epoch", ck.get("epoch"))
    steps = ck.get("optimizer_steps")
    scaler_used = False  # pilot used autocast without GradScaler
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "keys": keys,
        "has_model": has_model,
        "has_optimizer": has_opt,
        "n_opt_state": int(len(opt["state"])) if has_opt else 0,
        "optimizer_steps": steps,
        "completed_virtual_epoch": epoch,
        "has_scheduler": "scheduler" in ck or "sch" in ck,
        "has_amp_scaler": "scaler" in ck,
        "amp_scaler_used_in_pilot": scaler_used,
        "has_rng": any("rng" in str(k).lower() for k in ck),
        "has_sampler_state": "sampler" in ck or "crop_cycle" in ck,
        "crop_cycle_is_function_of_virtual_epoch": True,
        "lr": None,
        "missing_required": missing,
    }
    if has_opt and opt["param_groups"]:
        audit["lr"] = float(opt["param_groups"][0].get("lr", float("nan")))
        st0 = next(iter(opt["state"].values())) if opt["state"] else {}
        audit["adam_step"] = float(st0["step"]) if torch.is_tensor(st0.get("step")) else st0.get("step")
    # Hard block: weights only / cannot restore optimizer
    if (not has_opt) or missing:
        audit["ok"] = False
        audit["block"] = True
        audit["reason"] = "weights_only_or_missing_optimizer"
        return audit
    if int(steps or -1) < 1:
        audit["ok"] = False
        audit["block"] = True
        audit["reason"] = f"missing_optimizer_steps epoch={epoch} steps={steps}"
        return audit
    if expect_epoch is not None and int(epoch) != int(expect_epoch):
        audit["ok"] = False
        audit["block"] = True
        audit["reason"] = f"unexpected_epoch epoch={epoch} expected={expect_epoch}"
        return audit
    if expect_epoch is None and (epoch is None or int(epoch) < 2):
        audit["ok"] = False
        audit["block"] = True
        audit["reason"] = f"epoch_lt_2 epoch={epoch}"
        return audit
    audit["ok"] = True
    audit["block"] = False
    audit["reason"] = None
    audit["reconstruct"] = {
        "scheduler": "replay ReduceLROnPlateau.step(val_loss) over train_history",
        "amp_scaler": "not_used_autocast_only",
        "rng": "per-epoch EventStationBalancedSampler(seed+ep) and dataset default_rng(seed+idx); crop=(hash+ep)%3",
        "sampler_crop_cycle": "ds.set_virtual_epoch(next_epoch) is exact",
    }
    return audit


def verify_semantic_hashes(run_hashes: dict[str, Any], files: dict[str, Path]) -> dict[str, Any]:
    mismatches = {}
    current = {}
    for name, path in files.items():
        cur = sha256_file(path)
        current[name] = cur
        expected = run_hashes.get(name)
        if expected is not None and expected != cur:
            mismatches[name] = {"expected": expected, "current": cur}
    return {"ok": len(mismatches) == 0, "current": current, "mismatches": mismatches}
