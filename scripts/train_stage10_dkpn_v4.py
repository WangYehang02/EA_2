#!/usr/bin/env python
"""DKPN v4 full FP32. Random init. A-loss. Pilot hard-stops at 3 virtual epochs.

Never loads v3 epoch_2 or F-smoke weights. Never auto-continues to 30 epochs.
No --mode resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import load_yaml_config, resolve_instance_root, save_json
from earthquake.stage10.checkpoint_policy import (
    FIXED_0P2_METRIC,
    OFFICIAL_PROBABILITY_THRESHOLD,
    best_metric_filename,
    checkpoint_metadata,
    epoch_checkpoint_name,
    is_valid_best,
)
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.fp32_guard import (
    Fp32ActivationGuard,
    PrecisionMismatch,
    assert_no_autocast,
    batch_loss_fp32,
    eval_dev_plain_fp32,
    loss_terms_fp32,
    state_dict_tensor_sha256,
)
from earthquake.stage10.gpu_policy import MIN_GPUS, idle_gpu_indices, launch_block_reason, status_dict, training_process_running
from earthquake.stage10.nonfinite import NonfiniteError
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.phase_balanced_loss import partial_label_group_terms
from earthquake.stage10.train_monitor import divergence_report, psn_means, snapshot_norms
from earthquake.stage10.diag_infer import collapse_flag, psn_distribution as psn_dist
from earthquake.stage10.v4_pilot import pilot_gate_v4
from earthquake.utils import ensure_dir

CFG_PATH = ROOT / "configs/stage10/dkpn_clean_v4_seed42_fp32.yaml"
V3_DIR = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
V2_DIR = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42")
V3_EPOCH2 = V3_DIR / "checkpoints" / "epoch_2.pt"
SMOKE_F = V3_DIR / "forensic_nonfinite_e3" / "smoke_ef" / "plan_F" / "last_state.forensic.pt"


def _load_v2_helpers():
    spec = importlib.util.spec_from_file_location("train_dkpn_v2_helpers", ROOT / "scripts/train_stage10_dkpn_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def group_losses_fp32(model, batch, device) -> dict:
    assert_no_autocast()
    model.train()
    model.zero_grad(set_to_none=True)
    x = batch["x"].to(device).float()
    kw = {k: batch[k].to(device) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    is_noise = batch["is_noise"].to(device)
    logits = dkpn_logits(model, x)
    groups = partial_label_group_terms(logits, is_noise=is_noise, **kw)
    denom = sum(groups[k]["w"].sum() for k in groups).clamp_min(1.0)
    terms = {k: (groups[k]["nll"] * groups[k]["w"]).sum() / denom for k in groups}
    stats = {f"L_{k}": float(terms[k].detach().cpu()) for k in terms}
    stats["L_total"] = float(partial_label_nll(logits, **kw).detach().cpu())
    params = [p for p in model.parameters() if p.requires_grad]
    norms = {}
    for name, t in terms.items():
        grads = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        acc = 0.0
        for g in grads:
            if g is not None:
                acc += float(g.detach().float().pow(2).sum().cpu())
        norms[name] = acc ** 0.5
    g_phase = norms.get("p", 0.0) + norms.get("s", 0.0)
    g_n = norms.get("certified_noise", 0.0) + norms.get("complete_n", 0.0)
    stats["group_grad_norms"] = norms
    stats["phase_over_n_grad"] = g_phase / max(g_n, 1e-12)
    model.zero_grad(set_to_none=True)
    return stats


def _block_precision(out: Path, detail: dict) -> None:
    payload = {"reason": "precision_mismatch", "precision_mode": "fp32", "confirm_read": False, **detail}
    save_json(payload, out / "TRAIN_BLOCKED_PRECISION_MISMATCH")
    (out / "TRAIN_BLOCKED_PRECISION_MISMATCH.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "PILOT.FAILED").write_text(json.dumps(payload, indent=2, default=str))


def _copy_v3_splits(out: Path) -> dict:
    src_tr = V3_DIR / "train_online_catalog.parquet"
    src_va = V3_DIR / "val_subset.parquet"
    if not src_tr.is_file() or not src_va.is_file():
        raise SystemExit("v3 split files missing; refuse to regenerate a different split")
    shutil.copy2(src_tr, out / "train_online_catalog.parquet")
    shutil.copy2(src_va, out / "val_subset.parquet")
    return {
        "parent_train_catalog_sha256": sha256_file(src_tr),
        "parent_val_subset_sha256": sha256_file(src_va),
        "copied_train_catalog_sha256": sha256_file(out / "train_online_catalog.parquet"),
        "copied_val_subset_sha256": sha256_file(out / "val_subset.parquet"),
        "split_source": str(V3_DIR),
        "splits_identical_to_v3": sha256_file(src_tr) == sha256_file(out / "train_online_catalog.parquet")
        and sha256_file(src_va) == sha256_file(out / "val_subset.parquet"),
    }


def compare_init_with_v3(v4_sd: dict) -> dict:
    v3_p = V3_DIR / "checkpoints" / "init.pt"
    v3 = torch.load(v3_p, map_location="cpu", weights_only=False)
    v3_sd = v3["model"] if isinstance(v3, dict) and "model" in v3 else v3
    h3 = state_dict_tensor_sha256(v3_sd)
    h4 = state_dict_tensor_sha256(v4_sd)
    file3 = sha256_file(v3_p)
    keys3, keys4 = set(v3_sd), set(v4_sd)
    return {
        "v3_init_pt": str(v3_p),
        "v3_init_file_sha256": file3,
        "v3_param_tensor_sha256": h3,
        "v4_param_tensor_sha256": h4,
        "parameter_tensors_identical": h3 == h4,
        "key_diff_only_in_v3": sorted(keys3 - keys4),
        "key_diff_only_in_v4": sorted(keys4 - keys3),
        "note": None
        if h3 == h4
        else "parameter tensors differ from v3 init; not silently ignored (seed/build path/PyTorch RNG)",
    }


def run_pilot(cfg, wave, device, seed, out, workers, root: Path, v2h) -> None:
    if bool(cfg.get("amp", False)) or bool(cfg.get("autocast", False)) or cfg.get("precision_mode") != "fp32":
        raise SystemExit("v4 requires precision_mode=fp32 amp=false autocast=false")
    split_meta = _copy_v3_splits(out)
    cat = v2h.annotate_online_catalog(pd.read_parquet(out / "train_online_catalog.parquet"))
    val_meta = pd.read_parquet(out / "val_subset.parquet")
    val_cat = v2h.annotate_online_catalog(val_meta)
    val_cat = val_cat.copy()
    val_cat["crop_kind"] = "s_centered"
    hashes = {
        "run": cfg["run_name"],
        "seed": seed,
        "init": "random",
        "precision_mode": "fp32",
        "amp": False,
        "autocast": False,
        "grad_scaler": False,
        "loss": "current_partial_label_nll",
        "phase_balanced_loss": False,
        "mixed_crop": "(stable_hash(trace_id)+virtual_epoch)%3",
        "online_crop": True,
        "windows_per_virtual_epoch": int(len(cat)),
        "crop_cycle_virtual_epochs": 3,
        "hard_stop_after_pilot": True,
        "never_auto_continue_30": True,
        "official_height": 0.2,
        "v3_epoch2_loaded": False,
        "f_smoke_weights_loaded": False,
        "confirm_read": False,
        "val_crop": "s_centered",
        "config_sha256": sha256_file(root / "configs/stage10/dkpn_clean_v4_seed42_fp32.yaml"),
        "train_script_sha256": sha256_file(Path(__file__)),
        "partial_label_sha256": sha256_file(root / "src/earthquake/stage10/partial_label.py"),
        "dataset_v2_sha256": sha256_file(root / "src/earthquake/stage10/dataset_v2.py"),
        "crop_v2_sha256": sha256_file(root / "src/earthquake/stage10/crop_v2.py"),
        **split_meta,
        "train_catalog_sha256": sha256_file(out / "train_online_catalog.parquet"),
        "val_subset_sha256": sha256_file(out / "val_subset.parquet"),
    }
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_dkpn_random().to(device)
    ensure_dir(out / "checkpoints")
    init_payload = {
        "model": model.state_dict(),
        "seed": seed,
        "epoch": -1,
        "init": "random",
        "v2_loaded": False,
        "v3_epoch2_loaded": False,
        "f_smoke_loaded": False,
        "precision_mode": "fp32",
    }
    torch.save(init_payload, out / "checkpoints" / "init.pt")
    hashes["init_pt_file_sha256"] = sha256_file(out / "checkpoints" / "init.pt")
    hashes["init_param_tensor_sha256"] = state_dict_tensor_sha256(model.state_dict())
    hashes["init_vs_v3"] = compare_init_with_v3(model.state_dict())
    nvis = torch.cuda.device_count() if device.type == "cuda" else 1
    hashes["data_parallel_ngpu"] = int(nvis)
    hashes["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    hashes["per_rank_batch_size"] = int(cfg["batch_size"]) // max(nvis, 1)
    hashes["global_batch_size"] = int(cfg["batch_size"])
    save_json(hashes, out / "run_hashes.json")
    save_json(hashes["init_vs_v3"], out / "init_vs_v3.json")
    if nvis != 4:
        raise SystemExit(f"v4 pilot requires 4 GPUs, have {nvis}")
    if int(cfg["batch_size"]) != 32 or (32 % nvis) != 0:
        raise SystemExit("v4 requires global batch 32 divisible by nvis")
    model = torch.nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(cfg["lr_factor"]), patience=int(cfg["lr_patience"])
    )
    batch_sz = 32
    guard = Fp32ActivationGuard()
    guard.attach(model)
    val_loader, _ = v2h.make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
    try:
        init_metrics = eval_dev_plain_fp32(model, val_loader, device, psn_dist=psn_dist, collapse_flag=collapse_flag)
    except PrecisionMismatch as e:
        _block_precision(out, {"when": "init_eval", "error": repr(e)})
        return
    save_json({"init_eval_height_0p2": init_metrics, "confirm_read": False, "precision_mode": "fp32"}, out / "init_eval.json")
    epoch_eta_s = 7550.0
    launch = {
        "pid": os.getpid(),
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(sys.argv),
        "out_dir": str(out),
        "run_name": cfg["run_name"],
        "precision_mode": "fp32",
        "amp": False,
        "autocast": False,
        "grad_scaler": False,
        "per_rank_batch": 8,
        "global_batch": 32,
        "nvis": 4,
        "init_param_tensor_sha256": hashes["init_param_tensor_sha256"],
        "init_vs_v3_identical": hashes["init_vs_v3"]["parameter_tensors_identical"],
        "config_sha256": hashes["config_sha256"],
        "train_script_sha256": hashes["train_script_sha256"],
        "train_catalog_sha256": hashes["train_catalog_sha256"],
        "val_subset_sha256": hashes["val_subset_sha256"],
        "eta_sec_per_epoch": epoch_eta_s,
        "eta_sec_total": epoch_eta_s * 3,
        "confirm_read": False,
        "continued_to_30": False,
        "v3_epoch2_loaded": False,
        "f_smoke_weights_loaded": False,
    }
    save_json(launch, out / "LAUNCH.json")
    print(json.dumps({"LAUNCH": launch}, indent=2), flush=True)
    history = []
    best_loss = 1e9
    best_metric = -1.0
    ever_valid_best = False
    optimizer_steps = 0
    windows_seen = 0
    first500: list[float] = []
    kinds_per_epoch: list[set[str]] = []
    bg_pseudo_n = 0
    n_skip = 0
    nan_inf = False
    last_batch = None
    series: list[dict] = []
    (out / "TRAIN.RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    (out / "TRAIN.PID").write_text(str(os.getpid()))
    max_ep = 3
    jsonl = (out / "monitor.jsonl").open("w", encoding="utf-8")
    try:
        for ep in range(max_ep):
            train_loader, ds = v2h.make_loader(
                cat, wave, batch=batch_sz, workers=workers, augment=True, seed=seed + ep,
                shuffle=False, balanced=True, num_samples=len(cat), drop_last=True,
            )
            ds.set_virtual_epoch(ep)
            model.train()
            tr_losses, gnorms, pmeans = [], [], []
            vis_p = vis_s = n_eff = n_zero = 0
            kinds: dict[str, int] = {}
            t0 = __import__("time").time()
            for bi, batch in enumerate(train_loader):
                opt.zero_grad(set_to_none=True)
                want = bi == 0 or (bi % 100 == 0)
                gsum = None
                try:
                    loss, logits, gsum = batch_loss_fp32(
                        model, batch, device, optimizer_step=optimizer_steps, guard=guard if want else None
                    )
                except PrecisionMismatch as e:
                    _block_precision(out, {"epoch": ep, "bi": bi, "error": repr(e), "guard": gsum})
                    return
                except NonfiniteError as e:
                    nan_inf = True
                    save_json({"ok": False, **e.to_dict(), "epoch": ep, "bi": bi, "confirm_read": False}, out / "TRAIN.FAILED.json")
                    (out / "PILOT.FAILED").write_text(json.dumps({"ok": False, "reasons": ["nan_inf"], **e.to_dict()}, indent=2, default=str))
                    return
                loss.backward()
                g = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
                if not torch.isfinite(g):
                    nan_inf = True
                    (out / "PILOT.FAILED").write_text(json.dumps({"ok": False, "reasons": ["nonfinite_grad"], "epoch": ep, "bi": bi}, indent=2))
                    return
                gnorms.append(float(g.detach().cpu()))
                opt.step()
                optimizer_steps += 1
                windows_seen += int(batch["x"].shape[0])
                tr_losses.append(float(loss.detach().cpu()))
                if optimizer_steps <= 500:
                    first500.append(tr_losses[-1])
                vis_p += int(batch["vis_p"].sum())
                vis_s += int(batch["vis_s"].sum())
                wsum = (batch["p_pos"] + batch["s_pos"] + batch["n_pos"] + batch["not_p"] + batch["not_s"]).sum(dim=-1)
                n_eff += int((wsum > 0).sum().item())
                n_zero += int((wsum <= 0).sum().item())
                for i, k in enumerate(batch["crop_kind"]):
                    kinds[k] = kinds.get(k, 0) + 1
                    if k == "background" and not bool(batch["is_noise"][i]) and float(batch["n_pos"][i].sum()) > 1e-6:
                        bg_pseudo_n += 1
                with torch.no_grad():
                    pr = torch.softmax(logits, dim=1)
                    pmeans.append([float(pr[:, i].mean().cpu()) for i in range(3)])
                last_batch = batch
                if want:
                    snap = snapshot_norms(model, opt)
                    row = {
                        "epoch": ep,
                        "bi": bi,
                        "loss": tr_losses[-1],
                        "finite": True,
                        "grad_norm": gnorms[-1],
                        "act_abs_max": (gsum or {}).get("worst_abs_max"),
                        "act_p99_abs": (gsum or {}).get("worst_p99_abs"),
                        "act_p999_abs": (gsum or {}).get("worst_p999_abs"),
                        "act_worst_module": (gsum or {}).get("worst_module"),
                        "loss_terms": loss_terms_fp32(logits, batch, device),
                        **psn_means(logits),
                        **snap,
                        "param_abs_max": snap.get("param_abs_max"),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "nan_inf": False,
                    }
                    series.append(row)
                    jsonl.write(json.dumps(row) + "\n")
                    jsonl.flush()
                    if bi % 100 == 0:
                        print(json.dumps({"ep": ep, "bi": bi, "loss": row["loss"], "act_abs_max": row["act_abs_max"]}), flush=True)
            kinds_per_epoch.append(set(kinds))
            val_loader, _ = v2h.make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
            try:
                vm = eval_dev_plain_fp32(model, val_loader, device, psn_dist=psn_dist, collapse_flag=collapse_flag)
            except PrecisionMismatch as e:
                _block_precision(out, {"when": f"eval_epoch_{ep}", "error": repr(e)})
                return
            except NonfiniteError as e:
                nan_inf = True
                (out / "PILOT.FAILED").write_text(json.dumps({"ok": False, "reasons": ["nan_inf_eval"], **e.to_dict()}, indent=2, default=str))
                return
            finally:
                model.train()
            diag = {}
            if last_batch is not None:
                try:
                    diag = group_losses_fp32(model, last_batch, device)
                except Exception as e:
                    diag = {"phase_over_n_grad": None, "diag_error": repr(e)}
            valid = is_valid_best(metric=vm["f1"], init_metric=init_metrics["f1"])
            rec = {
                "virtual_epoch": ep,
                "crop_cycles": (ep + 1) / 3.0,
                "optimizer_steps": optimizer_steps,
                "windows_seen": windows_seen,
                "train_loss": float(np.mean(tr_losses)),
                "val_loss": vm["val_loss"],
                "val_s_f1_0p5": vm["f1"],
                "val_s_f1_fixed0p2": vm["f1"],
                "val_s_f1_calibrated": None,
                "init_s_f1_fixed0p2": init_metrics["f1"],
                "init_oracle_acc@0.5s": init_metrics.get("oracle_argmax_accuracy@0.5s"),
                "metrics_height_0p2": vm["metrics_height_0p2"],
                "precision@0.5s_height0p2": vm["metrics_height_0p2"]["precision@0.5s"],
                "recall@0.5s_height0p2": vm["metrics_height_0p2"]["recall@0.5s"],
                "f1@0.5s_height0p2": vm["metrics_height_0p2"]["f1@0.5s"],
                "picks_per_trace_height0p2": vm["metrics_height_0p2"]["picks_per_trace"],
                "frac_no_s_peak_height0p2": vm["metrics_height_0p2"]["frac_no_s_peak"],
                "oracle_acc@0.5s": vm.get("oracle_argmax_accuracy@0.5s"),
                "true_s_probability": vm.get("true_s_probability"),
                "local_s_peak_probability": vm.get("local_s_peak_probability"),
                "psn_val": vm.get("psn"),
                "prob_mean_psn": list(np.mean(pmeans, axis=0)) if pmeans else None,
                "all_n_collapse": vm.get("all_n_collapse"),
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": float(np.mean(gnorms) if gnorms else 0.0),
                "n_vis_p": vis_p,
                "n_vis_s": vis_s,
                "kinds": kinds,
                "n_effective_loss_windows": n_eff,
                "n_zero_loss_windows": n_zero,
                "bg_pseudo_n_windows": bg_pseudo_n,
                "silent_skips": n_skip,
                "loss_groups": {k: diag[k] for k in diag if str(k).startswith("L_")},
                "phase_over_n_grad": diag.get("phase_over_n_grad"),
                "act_abs_max_last": (series[-1].get("act_abs_max") if series else None),
                "param_abs_max": (series[-1].get("param_abs_max") if series else None),
                "bn_running_var_max": (series[-1].get("bn_running_var_max") if series else None),
                "adam_abs_max": (series[-1].get("adam_abs_max") if series else None),
                "logits_max_last": (series[-1].get("logits_max") if series else None),
                "sec": __import__("time").time() - t0,
                "utc": datetime.now(timezone.utc).isoformat(),
                "confirm_read": False,
                "valid_best_fixed0p2": valid,
                "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                "metric_name_fixed0p2": FIXED_0P2_METRIC,
                "nan_inf": nan_inf,
                "precision_mode": "fp32",
                "continued_to_30": False,
            }
            history.append(rec)
            save_json({"history": history, "hashes": hashes, "init_eval": init_metrics}, out / "train_history.json")
            raw = _raw_model(model)
            payload = {
                "model": raw.state_dict(),
                "epoch": ep,
                "opt": opt.state_dict(),
                "scheduler": sch.state_dict(),
                "seed": seed,
                "optimizer_steps": optimizer_steps,
                "windows_seen": windows_seen,
                "v2_loaded": False,
                "v3_epoch2_loaded": False,
                "f_smoke_loaded": False,
                "precision_mode": "fp32",
                "virtual_epoch": ep,
                **checkpoint_metadata(
                    virtual_epoch=ep,
                    metric_name=FIXED_0P2_METRIC,
                    metric_value=vm["f1"],
                    init_metric=init_metrics["f1"],
                    probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                    valid_best=False,
                    extra={"kind": "epoch_snapshot", "val_loss": vm["val_loss"]},
                ),
            }
            torch.save(payload, out / "checkpoints" / epoch_checkpoint_name(ep))
            torch.save(payload, out / "checkpoints" / "last.pt")
            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]
                torch.save({"model": raw.state_dict(), "epoch": ep, "val_loss": best_loss, "kind": "best_loss", "precision_mode": "fp32"}, out / "checkpoints" / "best_loss.pt")
            if valid:
                ever_valid_best = True
                best_metric = vm["f1"]
                best_payload = {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "val_s_f1": best_metric,
                    "val_s_f1_fixed0p2": best_metric,
                    "precision_mode": "fp32",
                    **checkpoint_metadata(
                        virtual_epoch=ep,
                        metric_name=FIXED_0P2_METRIC,
                        metric_value=best_metric,
                        init_metric=init_metrics["f1"],
                        probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                        valid_best=True,
                    ),
                }
                torch.save(best_payload, out / "checkpoints" / "best_metric.pt")
                torch.save(best_payload, out / "checkpoints" / best_metric_filename("fixed0p2"))
            save_json(
                {
                    "valid_best": ever_valid_best,
                    "best_metric_fixed0p2": None if best_metric < 0 else best_metric,
                    "init_s_f1_fixed0p2": init_metrics["f1"],
                    "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                    "f1_all_zero_cannot_mint_valid_best": True,
                    "calibrated_best_not_written": True,
                },
                out / "checkpoints" / "best_metric_status.json",
            )
            sch.step(vm["val_loss"])
            print(json.dumps({k: rec[k] for k in rec if k != "metrics_height_0p2"}, default=str), flush=True)
        if not ever_valid_best:
            torch.save(
                {
                    "model": torch.load(out / "checkpoints" / "init.pt", map_location="cpu", weights_only=False)["model"],
                    "epoch": -1,
                    "val_s_f1_fixed0p2": init_metrics["f1"],
                    "precision_mode": "fp32",
                    **checkpoint_metadata(
                        virtual_epoch=-1,
                        metric_name=FIXED_0P2_METRIC,
                        metric_value=init_metrics["f1"],
                        init_metric=init_metrics["f1"],
                        probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                        valid_best=False,
                        extra={"reason": "never_beat_init_plus_min_delta"},
                    ),
                },
                out / "checkpoints" / "best_metric.pt",
            )
        f500 = float(np.mean(first500)) if first500 else None
        div = divergence_report(series)
        save_json(div, out / "divergence.json")
        ok, reasons = pilot_gate_v4(
            history, f500, kinds_per_epoch, bg_pseudo_n, nan_inf, n_skip, ever_valid_best, div, False
        )
        gate = {
            "ok": ok,
            "reasons": reasons,
            "first500_loss": f500,
            "official_height": 0.2,
            "never_used_calibrated_threshold": True,
            "hard_stop": True,
            "continued_to_30": False,
            "confirm_read": False,
            "precision_mode": "fp32",
            "silent_skips": n_skip,
            "bg_pseudo_n": bg_pseudo_n,
            "divergence": div,
            "ever_valid_best": ever_valid_best,
        }
        save_json(gate, out / "pilot_gate.json")
        if ok:
            (out / "PILOT.PASSED").write_text(
                datetime.now(timezone.utc).isoformat() + "\nwait_for_explicit_approval_before_any_30_epoch\ncontinued_to_30=false\n"
            )
        else:
            (out / "PILOT.FAILED").write_text(json.dumps(gate, indent=2, default=str))
        print("PILOT", json.dumps(gate, indent=2, default=str), flush=True)
        print("HARD_STOP continued_to_30=false", flush=True)
    except PrecisionMismatch as e:
        _block_precision(out, {"error": repr(e)})
    except Exception as e:
        (out / "TRAIN.FAILED").write_text(repr(e))
        raise
    finally:
        jsonl.close()
        guard.remove()
        if (out / "TRAIN.RUNNING").exists():
            (out / "TRAIN.RUNNING").unlink()
        if (out / "TRAIN.PID").exists():
            (out / "TRAIN.PID").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "status"], required=True)
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--gpus", default=None, help="comma-separated physical GPU ids (all must be idle)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    if args.mode == "resume" or "30" in args.mode:
        raise SystemExit("v4 has no resume/30-epoch mode")
    seed = int(cfg["seed"])
    out = ensure_dir(Path(cfg["out_dir"]))
    if args.mode == "status":
        print(json.dumps(status_dict(out_dir=out), indent=2))
        return
    forbidden = {V2_DIR.resolve(), V3_DIR.resolve()}
    if out.resolve() in forbidden:
        raise SystemExit("refusing to write into v2/v3 directory")
    if SMOKE_F.is_file():
        # presence is fine; loading it is forbidden and never done
        pass
    if (out / "PILOT.PASSED").is_file() or (out / "PILOT.FAILED").is_file():
        raise SystemExit("v4 pilot already finished; will not relaunch")
    if (out / "checkpoints" / "last.pt").is_file() or (out / "checkpoints" / "init.pt").is_file():
        raise SystemExit("v4 checkpoints exist; refusing overwrite")
    idle = idle_gpu_indices()
    already = bool(training_process_running())
    flag = (out / "TRAIN.RUNNING").is_file() or (V3_DIR / "TRAIN.RUNNING").is_file()
    if args.gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        req = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
        occ = [i for i in req if i not in idle]
        if already or flag:
            reason = "training_already_running"
        elif occ:
            reason = f"requested_gpus_not_idle_{occ}"
        elif len(req) < MIN_GPUS:
            reason = f"need_min_gpus_{MIN_GPUS}_have_{len(req)}"
        else:
            reason = None
    else:
        reason = launch_block_reason(mode="pilot", n_idle=len(idle), already_training=already, train_running_flag=flag)
        if reason is None:
            pick = idle[:4]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in pick)
    if reason:
        save_json({"ok": False, "reason": reason, "idle": idle, "mode": args.mode, "confirm_read": False}, ROOT / "artifacts/results/stage10/dkpn_v4_last_launch_refuse.json")
        print(json.dumps({"refused": reason, "n_idle": len(idle), "idle": idle}))
        raise SystemExit(3 if "need_min_gpus" in str(reason) else 4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda required")
    frozen = dict(cfg)
    frozen["runtime_gpus"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    frozen["runtime_global_batch_size"] = 32
    frozen["hard_stop_after_pilot"] = True
    frozen["never_auto_continue_30"] = True
    frozen["confirm_read"] = False
    frozen["continued_to_30"] = False
    frozen["precision_mode"] = "fp32"
    (out / "frozen_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    shutil.copy2(ROOT / args.config, out / "source_config.yaml")
    inst = resolve_instance_root()
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))
    v2h = _load_v2_helpers()
    run_pilot(cfg, wave, device, seed, out, workers, ROOT, v2h)


if __name__ == "__main__":
    raise SystemExit(main())
