#!/usr/bin/env python
"""Forensic freeze / epoch-3 scan / A-B-C-D replay for v3 non-finite E3.

Does not overwrite epoch_2.pt, PILOT.PASSED, or existing train logs.
Does not read confirm. Does not start 30-epoch training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import load_yaml_config, resolve_instance_root, save_json
from earthquake.stage10.amp_forward import (
    batch_loss_amp_forward_fp32_nll,
    batch_loss_legacy_amp_path_a,
    shard_valid_counts,
    slice_batch,
)
from earthquake.stage10.crop_v2 import IN_SAMPLES
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.finite_hooks import install_finite_hooks, remove_hooks
from earthquake.stage10.partial_label import partial_nll_numerator, token_weights
from earthquake.stage10.v3_resume_audit import sha256_file
from earthquake.utils import ensure_dir

RUN = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
FORE = RUN / "forensic_nonfinite_e3"
CFG = ROOT / "configs/stage10/dkpn_clean_v3_seed42_mixedcrop.yaml"
N_RANKS = 4
BATCH = 32
SEED = 42
EP = 3


def _v2():
    spec = importlib.util.spec_from_file_location("train_dkpn_v2_helpers", ROOT / "scripts/train_stage10_dkpn_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_finite_report(t: torch.Tensor, name: str) -> dict:
    x = t.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    n_nan = int(torch.isnan(x).sum().cpu())
    n_inf = int(torch.isinf(x).sum().cpu())
    mx = float(x[finite].abs().max().cpu()) if finite.any() else None
    return {"name": name, "n": int(x.numel()), "n_nan": n_nan, "n_inf": n_inf, "max_abs": mx, "dtype": str(t.dtype)}


def audit_checkpoint(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {
        "path": str(path),
        "sha256": sha256_file(path),
        "keys": sorted(ck.keys()) if isinstance(ck, dict) else None,
        "has_scheduler": "scheduler" in ck or "sch" in ck,
        "has_grad_scaler": "scaler" in ck,
        "grad_scaler_scale": None,
        "optimizer_steps": ck.get("optimizer_steps"),
        "virtual_epoch": ck.get("virtual_epoch", ck.get("epoch")),
        "lr": None,
        "model": [],
        "buffers": [],
        "optimizer_exp_avg": [],
        "optimizer_exp_avg_sq": [],
        "n_nonfinite_model": 0,
        "n_nonfinite_opt": 0,
        "confirm_read": False,
    }
    model = ck.get("model") or {}
    for k, v in model.items():
        if torch.is_tensor(v):
            r = tensor_finite_report(v, k)
            bucket = "buffers" if (k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked")) else "model"
            out[bucket].append(r)
            if r["n_nan"] or r["n_inf"]:
                out["n_nonfinite_model"] += r["n_nan"] + r["n_inf"]
    opt = ck.get("opt") or {}
    if opt.get("param_groups"):
        out["lr"] = float(opt["param_groups"][0].get("lr", float("nan")))
    for pid, st in (opt.get("state") or {}).items():
        if not isinstance(st, dict):
            continue
        for fld in ("exp_avg", "exp_avg_sq"):
            if fld in st and torch.is_tensor(st[fld]):
                r = tensor_finite_report(st[fld], f"param{pid}.{fld}")
                out[f"optimizer_{fld}"].append(r)
                if r["n_nan"] or r["n_inf"]:
                    out["n_nonfinite_opt"] += r["n_nan"] + r["n_inf"]
    out["model_ok"] = out["n_nonfinite_model"] == 0
    out["optimizer_ok"] = out["n_nonfinite_opt"] == 0
    out["safe_to_resume_from"] = bool(out["model_ok"] and out["optimizer_ok"])
    # compact: keep only non-ok or summary counts
    out["model_n_tensors"] = len(out["model"]) + len(out["buffers"])
    out["opt_n_exp_avg"] = len(out["optimizer_exp_avg"])
    worst_m = max((r["max_abs"] or 0) for r in out["model"] + out["buffers"]) if out["model"] else None
    worst_o = max((r["max_abs"] or 0) for r in out["optimizer_exp_avg"] + out["optimizer_exp_avg_sq"]) if out["optimizer_exp_avg"] else None
    out["model_max_abs"] = worst_m
    out["optimizer_max_abs"] = worst_o
    out["model_nonfinite_names"] = [r["name"] for r in out["model"] + out["buffers"] if r["n_nan"] or r["n_inf"]]
    out["opt_nonfinite_names"] = [r["name"] for r in out["optimizer_exp_avg"] + out["optimizer_exp_avg_sq"] if r["n_nan"] or r["n_inf"]]
    # drop bulky per-tensor lists from disk json (keep summaries); write full to sidecar
    full = {"model": out["model"], "buffers": out["buffers"], "exp_avg": out["optimizer_exp_avg"], "exp_avg_sq": out["optimizer_exp_avg_sq"]}
    out.pop("model")
    out.pop("buffers")
    out.pop("optimizer_exp_avg")
    out.pop("optimizer_exp_avg_sq")
    return out, full


def batch_meta(batch, n_ranks=N_RANKS) -> dict:
    w = token_weights(
        p_pos=batch["p_pos"], s_pos=batch["s_pos"], n_pos=batch["n_pos"],
        not_p=batch["not_p"], not_s=batch["not_s"], pad_mask=batch["pad_mask"],
    )
    x = batch["x"]
    rows = []
    labels_ok = True
    crop_ok = True
    for i in range(x.shape[0]):
        pad = batch["pad_mask"][i]
        p_c = float(batch["p_c"][i]) if "p_c" in batch else -1.0
        s_c = float(batch["s_c"][i]) if "s_c" in batch else -1.0
        vis_p = float(batch["vis_p"][i])
        vis_s = float(batch["vis_s"][i])
        lab = torch.stack([batch[k][i] for k in ("p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask")])
        lab_finite = bool(torch.isfinite(lab).all())
        labels_ok = labels_ok and lab_finite
        p_ok = (p_c < 0) or (0.0 <= p_c < float(IN_SAMPLES))
        s_ok = (s_c < 0) or (0.0 <= s_c < float(IN_SAMPLES))
        vis_ok = (vis_p < 0.5 or p_ok) and (vis_s < 0.5 or s_ok)
        crop_ok = crop_ok and vis_ok
        rows.append(
            {
                "trace_name": str(batch["trace_name"][i]),
                "event_id": str(batch["event_id"][i]),
                "crop_kind": str(batch["crop_kind"][i]),
                "crop_start_not_in_batch": True,
                "certified_noise": bool(batch["is_noise"][i]),
                "is_noise": bool(batch["is_noise"][i]),
                "vis_p": vis_p,
                "vis_s": vis_s,
                "p_c": p_c,
                "s_c": s_c,
                "crop_boundary_ok": bool(vis_ok),
                "pad_frac": float(1.0 - pad.mean()),
                "valid_tokens": float(w[i].sum()),
                "x_min": float(x[i].min()),
                "x_max": float(x[i].max()),
                "label_min": float(lab.min()),
                "label_max": float(lab.max()),
                "mask_min": float(pad.min()),
                "mask_max": float(pad.max()),
                "x_finite": bool(torch.isfinite(x[i]).all()),
                "label_finite": lab_finite,
                "w_finite": bool(torch.isfinite(w[i]).all()),
                "rank": int(i // (x.shape[0] // n_ranks)) if x.shape[0] >= n_ranks else 0,
            }
        )
    xf = x.float().reshape(-1)
    absx = xf.abs()
    return {
        "n": int(x.shape[0]),
        "global_valid": float(w.sum()),
        "n_zero_loss_windows": int((w.sum(dim=-1) <= 0).sum()),
        "rank_valid": shard_valid_counts(batch, n_ranks),
        "x_min": float(xf.min()),
        "x_max": float(xf.max()),
        "x_p99": float(torch.quantile(absx, 0.99)),
        "x_p999": float(torch.quantile(absx, 0.999)),
        "x_finite": bool(torch.isfinite(x).all()),
        "labels_finite": labels_ok,
        "crop_boundary_ok": crop_ok,
        "rows": rows,
    }


def freeze() -> None:
    ensure_dir(FORE)
    ck = RUN / "checkpoints" / "epoch_2.pt"
    copy = FORE / "epoch_2.copy.pt"
    if not copy.is_file():
        shutil.copy2(ck, copy)
    audit, full = audit_checkpoint(ck)
    save_json(audit, FORE / "checkpoint_finite_audit.json")
    save_json(full, FORE / "checkpoint_finite_audit_tensors.json")
    hashes = {
        "epoch_2_pt_sha256": sha256_file(ck),
        "epoch_2_copy_sha256": sha256_file(copy),
        "config_sha256": sha256_file(CFG),
        "partial_label_sha256": sha256_file(ROOT / "src/earthquake/stage10/partial_label.py"),
        "dataset_v2_sha256": sha256_file(ROOT / "src/earthquake/stage10/dataset_v2.py"),
        "crop_v2_sha256": sha256_file(ROOT / "src/earthquake/stage10/crop_v2.py"),
        "train_v3_sha256": sha256_file(ROOT / "scripts/train_stage10_dkpn_v3.py"),
        "run_hashes": json.loads((RUN / "run_hashes.json").read_text()) if (RUN / "run_hashes.json").is_file() else None,
    }
    save_json(hashes, FORE / "code_config_hashes.json")
    failed = json.loads((RUN / "TRAIN.FAILED").read_text()) if (RUN / "TRAIN.FAILED").is_file() else {}
    scene = {
        "flag": "TRAIN.FAILED_NONFINITE_E3",
        "failure_utc": "2026-08-31T16:53:00Z",
        "failure_local": "2026-09-01T00:53:41+08:00",
        "source": "4gpu_resume_TRAIN.FAILED",
        "exception": failed.get("error") or "RuntimeError('non-finite loss')",
        "reconstructed_traceback": "".join(
            traceback.format_exception(RuntimeError, RuntimeError("non-finite loss"), None)
        )
        + "  File scripts/train_stage10_dkpn_v2.py, batch_loss, raise RuntimeError('non-finite loss')\n"
        "  File src/earthquake/stage10/v3_continue.py, run_resume, epoch=3\n"
        "NOTE: live traceback was not captured at crash; this is reconstructed from TRAIN.FAILED.json.\n",
        "last_completed_optimizer_step": 71076,
        "last_completed_virtual_epoch": 2,
        "failed_virtual_epoch": 3,
        "epoch3_batch_index": None,
        "grad_scaler_used": False,
        "grad_scaler_scale": None,
        "lr": 0.001,
        "nvis": failed.get("nvis", 4),
        "global_batch": failed.get("batch_size", 32),
        "per_rank_batch": 8,
        "parallelism": "DataParallel_not_DDP",
        "gpus": "3,4,5,6",
        "amp_wrapped_loss": True,
        "legacy_denom_clamp_min_1": True,
        "confirm_read": False,
        "pilot_passed_preserved": True,
        "epoch_2_not_overwritten": True,
        "sha256_epoch_2": hashes["epoch_2_pt_sha256"],
        "checkpoint_safe": audit["safe_to_resume_from"],
    }
    save_json(scene, FORE / "failure_scene.json")
    (RUN / "TRAIN.FAILED_NONFINITE_E3").write_text(
        datetime.now(timezone.utc).isoformat() + "\n" + json.dumps(scene, indent=2) + "\n"
    )
    print(json.dumps({"freeze": True, "safe_ckpt": audit["safe_to_resume_from"], "sha": hashes["epoch_2_pt_sha256"]}, indent=2))


def _loader(v2h, wave, *, augment: bool):
    import pandas as pd

    cat = v2h.annotate_online_catalog(pd.read_parquet(RUN / "train_online_catalog.parquet"))
    ld, ds = v2h.make_loader(
        cat, wave, batch=BATCH, workers=4, augment=augment, seed=SEED + EP,
        shuffle=False, balanced=True, num_samples=len(cat), drop_last=True,
    )
    ds.set_virtual_epoch(EP)
    return ld, ds, cat


def scan(max_batches: int | None = None) -> None:
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    v2h = _v2()
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    ld, ds, cat = _loader(v2h, wave, augment=True)
    stats = {
        "n_batches": 0,
        "n_windows": 0,
        "n_x_nonfinite": 0,
        "n_label_nonfinite": 0,
        "n_w_nonfinite": 0,
        "n_crop_illegal": 0,
        "n_zero_loss": 0,
        "n_bg_pseudo_n": 0,
        "n_rank_local_valid_zero": 0,
        "x_max": 0.0,
        "x_p99_max": 0.0,
        "x_p999_max": 0.0,
        "extreme_x_batches": [],
        "bad_traces": [],
        "kinds": {},
        "no_nan_to_num": True,
        "confirm_read": False,
    }
    for bi, batch in enumerate(ld):
        meta = batch_meta(batch)
        stats["n_batches"] += 1
        stats["n_windows"] += meta["n"]
        stats["n_zero_loss"] += meta["n_zero_loss_windows"]
        stats["x_max"] = max(stats["x_max"], abs(meta["x_max"]), abs(meta["x_min"]))
        stats["x_p99_max"] = max(stats["x_p99_max"], meta["x_p99"])
        stats["x_p999_max"] = max(stats["x_p999_max"], meta.get("x_p999", 0.0))
        if any(v == 0.0 for v in meta["rank_valid"]):
            stats["n_rank_local_valid_zero"] += 1
        if abs(meta["x_max"]) > 100 or abs(meta["x_min"]) > 100:
            stats["extreme_x_batches"].append(
                {"bi": bi, "x_min": meta["x_min"], "x_max": meta["x_max"], "p99": meta["x_p99"], "p999": meta["x_p999"]}
            )
        if not meta["x_finite"]:
            stats["n_x_nonfinite"] += 1
            stats["bad_traces"].extend([r["trace_name"] for r in meta["rows"] if not r["x_finite"]])
        if not meta.get("labels_finite", True):
            stats["n_label_nonfinite"] += 1
            stats["bad_traces"].extend([r["trace_name"] for r in meta["rows"] if not r.get("label_finite", True)])
        if not meta.get("crop_boundary_ok", True):
            stats["n_crop_illegal"] += 1
        for r in meta["rows"]:
            stats["kinds"][r["crop_kind"]] = stats["kinds"].get(r["crop_kind"], 0) + 1
            idx = meta["rows"].index(r)
            if r["crop_kind"] == "background" and (not r["is_noise"]) and float(batch["n_pos"][idx].sum()) > 1e-6:
                stats["n_bg_pseudo_n"] += 1
        if max_batches is not None and stats["n_batches"] >= max_batches:
            break
        if stats["n_batches"] % 200 == 0:
            print(json.dumps({"scan_batches": stats["n_batches"], "x_max": stats["x_max"]}), flush=True)
    stats["truncated"] = max_batches is not None
    save_json(stats, FORE / "epoch3_data_scan.json")
    print(json.dumps({k: stats[k] for k in stats if k != "bad_traces"}, indent=2))


def _load_model(device):
    ck = torch.load(FORE / "epoch_2.copy.pt", map_location="cpu", weights_only=False)
    m = build_dkpn_random().to(device)
    m.load_state_dict(ck["model"])
    m.train()
    return m, ck


def _bn_batch_stats_only(model: torch.nn.Module) -> None:
    """Match DataParallel replica forward: use per-shard batch stats, do not mutate running_var."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False


def replay_dp_shards(device, max_steps: int | None = None) -> dict:
    """Reproduce 4-GPU DataParallel AMP by running four independent train-mode bs=8 shards.

    Does not update parameters. Restores BN buffers between shards so running_var is not
    sequentially polluted (true DP replicas see the same frozen buffers and local batch stats).
    """
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    v2h = _v2()
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    ld, ds, cat = _loader(v2h, wave, augment=True)
    model, ck = _load_model(device)
    model.train()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last32 = {r: deque(maxlen=32) for r in range(N_RANKS)}
    found = None
    for bi, batch in enumerate(ld):
        meta = batch_meta(batch)
        for r in range(N_RANKS):
            last32[r].append(
                {
                    "batch_index": bi,
                    "rank": r,
                    "valid": meta["rank_valid"][r],
                    "global_valid": meta["global_valid"],
                    "x_max": meta["x_max"],
                }
            )
        shard_ok = []
        fail_rank = None
        bs = batch["x"].shape[0] // N_RANKS
        for r in range(N_RANKS):
            model.load_state_dict(frozen)
            model.train()
            _bn_batch_stats_only(model)
            sl = slice(r * bs, (r + 1) * bs)
            sub = slice_batch(batch, sl)
            try:
                loss, logits = batch_loss_legacy_amp_path_a(model, sub, device)
                shard_ok.append(
                    {
                        "rank": r,
                        "ok": True,
                        "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else None,
                        "logits_max": float(logits.detach().float().max().cpu()),
                        "valid": meta["rank_valid"][r],
                    }
                )
            except RuntimeError as e:
                fail_rank = r
                shard_ok.append({"rank": r, "ok": False, "error": repr(e), "valid": meta["rank_valid"][r]})
                found = {
                    "batch_index": bi,
                    "fail_rank": r,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "meta": {k: meta[k] for k in meta if k != "rows"},
                    "rank_valid": meta["rank_valid"],
                    "rows": meta["rows"],
                    "shard_results": shard_ok,
                    "optimizer_step_if_updated": 71076 + bi,
                    "mode": "dp_shard_bs8_train_amp_path_a",
                }
                cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
                torch.save({"batch": cpu_batch, "batch_index": bi, "meta": found}, FORE / "offending_batch.pt")
                save_json(found, FORE / "offending_batch_meta.json")
                save_json({str(rr): list(last32[rr]) for rr in last32}, FORE / "last32_batches_per_rank.json")
                print("FOUND_DP_SHARD", json.dumps({k: found[k] for k in found if k != "rows"}), flush=True)
                break
        if found is not None:
            break
        if max_steps is not None and bi + 1 >= max_steps:
            break
        if bi % 50 == 0:
            print(
                json.dumps(
                    {
                        "dp_shard_bi": bi,
                        "rank_valid": meta["rank_valid"],
                        "logits_max": [s.get("logits_max") for s in shard_ok],
                    }
                ),
                flush=True,
            )
    if found is None:
        save_json({"found": False, "scanned_batches": bi + 1, "mode": "dp_shards"}, FORE / "dp_shards_not_found.json")
    return found or {}


def replay_train_opt(device, max_steps: int | None = None) -> dict:
    """Replay epoch 3 with optimizer updates on epoch_2 COPY (legacy AMP path A).

    Does not write original checkpoints. Stops at first non-finite.
    """
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    v2h = _v2()
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    ld, ds, cat = _loader(v2h, wave, augment=True)
    model, ck = _load_model(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.load_state_dict(ck["opt"])
    last32 = {r: deque(maxlen=32) for r in range(N_RANKS)}
    found = None
    for bi, batch in enumerate(ld):
        meta = batch_meta(batch)
        for r in range(N_RANKS):
            last32[r].append(
                {
                    "batch_index": bi,
                    "rank": r,
                    "valid": meta["rank_valid"][r],
                    "global_valid": meta["global_valid"],
                    "x_max": meta["x_max"],
                    "n_zero": meta["n_zero_loss_windows"],
                }
            )
        opt.zero_grad(set_to_none=True)
        try:
            loss, logits = batch_loss_legacy_amp_path_a(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            g = torch.sqrt(
                sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None)
            )
            if not torch.isfinite(g):
                raise RuntimeError("non-finite grad")
            opt.step()
        except RuntimeError as e:
            found = {
                "batch_index": bi,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "meta": {k: meta[k] for k in meta if k != "rows"},
                "rank_valid": meta["rank_valid"],
                "rows": meta["rows"],
                "optimizer_step_if_updated": 71076 + bi,
                "mode": "replay_train_opt_path_a",
            }
            cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.save({"batch": cpu_batch, "batch_index": bi, "meta": found}, FORE / "offending_batch.pt")
            save_json(found, FORE / "offending_batch_meta.json")
            save_json({str(rr): list(last32[rr]) for rr in last32}, FORE / "last32_batches_per_rank.json")
            print("FOUND_TRAIN_OPT", json.dumps({k: found[k] for k in found if k != "rows"}), flush=True)
            break
        if max_steps is not None and bi + 1 >= max_steps:
            break
        if bi % 50 == 0:
            print(
                json.dumps(
                    {
                        "train_opt_bi": bi,
                        "loss": float(loss.detach().cpu()),
                        "logits_max": float(logits.detach().float().max().cpu()),
                        "step": 71076 + bi + 1,
                    }
                ),
                flush=True,
            )
    if found is None:
        save_json(
            {"found": False, "scanned_batches": bi + 1, "mode": "replay_train_opt"},
            FORE / "offending_not_found.json",
        )
    return found or {}


def replay_find_fail(device, max_steps: int | None = None) -> dict:
    """Frozen-weight train-mode replay (no optimizer). Legacy AMP path A."""
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    v2h = _v2()
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    ld, ds, cat = _loader(v2h, wave, augment=True)
    model, ck = _load_model(device)
    last32 = {r: deque(maxlen=32) for r in range(N_RANKS)}
    found = None
    for bi, batch in enumerate(ld):
        meta = batch_meta(batch)
        for r in range(N_RANKS):
            last32[r].append({"batch_index": bi, "rank": r, "valid": meta["rank_valid"][r], "global_valid": meta["global_valid"]})
        try:
            loss, logits = batch_loss_legacy_amp_path_a(model, batch, device)
        except RuntimeError as e:
            found = {
                "batch_index": bi,
                "error": repr(e),
                "meta": {k: meta[k] for k in meta if k != "rows"},
                "rank_valid": meta["rank_valid"],
                "rows": meta["rows"],
                "optimizer_step_if_updated": 71076 + bi,
            }
            cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.save({"batch": cpu_batch, "batch_index": bi, "meta": found}, FORE / "offending_batch.pt")
            save_json(found, FORE / "offending_batch_meta.json")
            save_json({str(r): list(last32[r]) for r in last32}, FORE / "last32_batches_per_rank.json")
            print("FOUND", json.dumps({k: found[k] for k in found if k != "rows"}), flush=True)
            break
        if max_steps is not None and bi + 1 >= max_steps:
            break
        if bi % 100 == 0:
            print(json.dumps({"replay_bi": bi, "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else None}), flush=True)
    if found is None:
        save_json({"found": False, "scanned_batches": bi + 1}, FORE / "offending_not_found.json")
    return found or {}


def _run_path(name, fn, model, batch, device) -> dict:
    model.zero_grad(set_to_none=True)
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    first, handles = install_finite_hooks(raw)
    out = {"path": name, "forward_ok": False, "loss_ok": False, "grad_ok": False}
    try:
        loss, logits = fn()
        out["forward_ok"] = True
        out["loss_ok"] = bool(torch.isfinite(loss))
        out["loss"] = float(loss.detach().cpu()) if torch.isfinite(loss) else None
        lf = logits.detach().float()
        out["logits_min"] = float(lf.min().cpu())
        out["logits_max"] = float(lf.max().cpu())
        out["logits_finite"] = bool(torch.isfinite(logits).all())
        nll, w, terms = partial_nll_numerator(
            logits.float(),
            p_pos=batch["p_pos"].to(device),
            s_pos=batch["s_pos"].to(device),
            n_pos=batch["n_pos"].to(device),
            not_p=batch["not_p"].to(device),
            not_s=batch["not_s"].to(device),
            pad_mask=batch["pad_mask"].to(device),
        )
        out["numerator"] = float(nll.sum().cpu())
        out["denominator"] = float(w.sum().cpu())
        out["terms"] = {k: float(v.detach().cpu()) for k, v in terms.items()}
        out["rank_valid"] = shard_valid_counts(batch, N_RANKS)
        if out["loss_ok"]:
            loss.backward()
            gs = [p.grad for p in raw.parameters() if p.grad is not None]
            out["grad_ok"] = all(torch.isfinite(g).all() for g in gs) if gs else False
    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    out["first_nonfinite_module"] = first.get("module")
    out["first_nonfinite"] = dict(first) if first else None
    remove_hooks(handles)
    return out


def abcd(device) -> None:
    blob_p = FORE / "offending_batch.pt"
    if not blob_p.is_file():
        raise SystemExit("run --mode replay first")
    blob = torch.load(blob_p, map_location="cpu", weights_only=False)
    batch = blob["batch"]
    model, _ = _load_model(device)
    model.train()
    def to_dev(b):
        out = {}
        for k, v in b.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out
    b = to_dev(batch)
    paths = {}
    paths["A_fp16_fwd_fp16_loss_legacy"] = _run_path(
        "A", lambda: batch_loss_legacy_amp_path_a(model, b, device), model, b, device
    )
    model, _ = _load_model(device)
    model.train()
    paths["B_fp16_fwd_fp32_loss"] = _run_path(
        "B", lambda: batch_loss_amp_forward_fp32_nll(model, b, device, amp=True, dtype=torch.float16), model, b, device
    )
    model, _ = _load_model(device)
    model.train()
    bf = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    paths["C_bf16_fwd_fp32_loss"] = _run_path(
        "C", lambda: batch_loss_amp_forward_fp32_nll(model, b, device, amp=True, dtype=bf), model, b, device
    )
    model, _ = _load_model(device)
    model.train()
    paths["D_fp32"] = _run_path(
        "D", lambda: batch_loss_amp_forward_fp32_nll(model, b, device, amp=False), model, b, device
    )
    verdict = {
        "A_fail": not paths["A_fp16_fwd_fp16_loss_legacy"].get("loss_ok"),
        "B_fail": not paths["B_fp16_fwd_fp32_loss"].get("loss_ok"),
        "C_fail": not paths["C_bf16_fwd_fp32_loss"].get("loss_ok"),
        "D_fail": not paths["D_fp32"].get("loss_ok"),
    }
    if verdict["A_fail"] and not verdict["B_fail"]:
        root = "loss_precision_fp16_nll"
    elif verdict["A_fail"] and verdict["B_fail"] and not verdict["D_fail"]:
        root = "fp16_activation_overflow"
    elif verdict["D_fail"]:
        root = "data_or_state_or_loss_logic"
    elif paths["A_fp16_fwd_fp16_loss_legacy"].get("rank_valid") and min(paths["A_fp16_fwd_fp16_loss_legacy"]["rank_valid"]) == 0:
        root = "local_valid_zero_ddp_style"
    else:
        root = "inconclusive_or_all_ok"
    save_json({"paths": paths, "verdict": verdict, "root_cause_class": root, "confirm_read": False}, FORE / "abcd_replay.json")
    print(json.dumps({"verdict": verdict, "root": root}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["freeze", "scan", "replay", "replay_train_opt", "abcd", "dp_shards", "all"], required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--scan-max-batches", type=int, default=None)
    ap.add_argument("--replay-max-steps", type=int, default=None)
    args = ap.parse_args()
    ensure_dir(FORE)
    if args.mode in {"freeze", "all"}:
        freeze()
    if args.mode in {"scan", "all"}:
        scan(args.scan_max_batches)
    if args.mode in {"replay", "replay_train_opt", "abcd", "dp_shards", "all"}:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise SystemExit("cuda required for replay")
        if args.mode in {"replay", "all"}:
            replay_find_fail(device, args.replay_max_steps)
        if args.mode == "replay_train_opt":
            replay_train_opt(device, args.replay_max_steps)
        if args.mode == "dp_shards":
            replay_dp_shards(device, args.replay_max_steps)
        if args.mode in {"abcd", "all"} and (FORE / "offending_batch.pt").is_file():
            abcd(device)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
