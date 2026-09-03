#!/usr/bin/env python
"""DKPN v3 mixed-crop. A-loss only. Pilot hard-stops at 3 epochs.
Resume to max 30 requires --mode resume after PILOT.PASSED (explicit approval). Never load v2.
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
    MIN_DELTA_F1,
    OFFICIAL_PROBABILITY_THRESHOLD,
    best_metric_filename,
    checkpoint_metadata,
    epoch_checkpoint_name,
    is_valid_best,
)
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.gpu_policy import MIN_GPUS, allocatable_gpu_indices, idle_gpu_indices, launch_block_reason, status_dict, training_process_running
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.stage10.phase_balanced_loss import partial_label_group_terms
from earthquake.stage10.threshold_calibration import OFFICIAL_HEIGHT, pick_metrics_at_thr, window_prob_stats
from earthquake.stage10.eval_fp32 import eval_dev_fp32
from earthquake.stage10.nonfinite import NonfiniteError
from earthquake.utils import ensure_dir

from earthquake.stage10.diag_infer import collapse_flag, psn_distribution as psn_dist

CFG_PATH = ROOT / "configs/stage10/dkpn_clean_v3_seed42_mixedcrop.yaml"
V2_DIR = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42")
V2_FROZEN = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42_failed_homogeneous_crop_cycle")
BUGGY = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean/train_seed42")


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


@torch.no_grad()
def eval_dev_fixed0p2(model, loader, device) -> dict:
    """S-centered val only. Official height=0.2. Eval forward/loss/metrics are FP32, autocast off."""
    return eval_dev_fp32(
        model,
        loader,
        device,
        pick_metrics=pick_metrics_at_thr,
        psn_dist=psn_dist,
        collapse_flag=collapse_flag,
        height=OFFICIAL_HEIGHT,
    )


def group_losses_and_phase_n_grad(model, batch, device) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    x = batch["x"].to(device)
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


def pilot_gate_v3(history, first500, kinds_per_epoch, bg_pseudo_n, nan_inf) -> tuple[bool, list[str]]:
    reasons = []
    if len(history) < 3:
        return False, ["need_3_virtual_epochs"]
    h0, h1, h2 = history[0], history[1], history[2]
    losses = [h["train_loss"] for h in history]
    if first500 is None or not np.isfinite(first500):
        reasons.append("no_first500")
    elif losses[-1] >= first500 * 0.98:
        reasons.append("train_loss_not_down_vs_first500")
    init_f1 = float(h0.get("init_s_f1_fixed0p2", 0))
    f1s = [float(h["val_s_f1_fixed0p2"]) for h in history]
    if f1s[2] <= init_f1 + MIN_DELTA_F1:
        reasons.append("epoch2_dev_f1_0p2_not_above_init")
    if f1s[2] < f1s[1]:
        reasons.append("epoch2_below_epoch1")
    ora0 = float(h0.get("init_oracle_acc@0.5s", 0))
    ora2 = float(h2.get("oracle_acc@0.5s", 0))
    if ora2 <= ora0:
        reasons.append("oracle_not_above_init")
    if f1s[2] <= init_f1:
        reasons.append("thresholded_f1_not_above_init")
    picks = float(h2.get("picks_per_trace_height0p2", 99))
    if picks > 3.0:
        reasons.append("picks_per_trace_gt_3")
    psn = h2.get("prob_mean_psn") or [0, 0, 1]
    if psn[0] < 1e-6 or psn[1] < 1e-6 or h2.get("all_n_collapse"):
        reasons.append("ps_collapsed")
    need = {"p_centered", "s_centered", "background"}
    for ep, kinds in enumerate(kinds_per_epoch):
        if not need.issubset(set(kinds)):
            reasons.append(f"missing_crop_kinds_epoch{ep}:{sorted(need - set(kinds))}")
    if bg_pseudo_n > 0:
        reasons.append("event_background_pseudo_n")
    if h2.get("grad_norm", 0) <= 0 or not np.isfinite(h2.get("grad_norm", 0)):
        reasons.append("bad_grad")
    if h2.get("lr", 0) <= 0:
        reasons.append("bad_lr")
    if nan_inf or any(not np.isfinite(h["train_loss"]) for h in history):
        reasons.append("nan_inf")
    return (len(reasons) == 0), reasons


def run_pilot(cfg, wave, device, seed, out, workers, root: Path, v2h) -> None:
    assert Path(cfg["out_dir"]).resolve() not in {V2_DIR.resolve(), V2_FROZEN.resolve(), BUGGY.resolve()}
    meta = v2h._meta_picker(True, float(cfg["noise_ratio"]), seed)
    cat = v2h.annotate_online_catalog(meta)
    cat.to_parquet(out / "train_online_catalog.parquet", index=False)
    val_meta = v2h.freeze_val_subset(int(cfg["val_traces"]), seed, out)
    val_cat = v2h.annotate_online_catalog(val_meta)
    val_cat = val_cat.copy()
    val_cat["crop_kind"] = "s_centered"  # mixed-crop must NOT be used for S F1 eval
    hashes = {
        "run": cfg["run_name"],
        "seed": seed,
        "init": "random",
        "loss": "current_partial_label_nll",
        "phase_balanced_loss": False,
        "mixed_crop": "(stable_hash(trace_id)+virtual_epoch)%3",
        "online_crop": True,
        "windows_per_virtual_epoch": int(len(cat)),
        "crop_cycle_virtual_epochs": 3,
        "hard_stop_after_pilot": True,
        "never_auto_continue_30": True,
        "official_height": 0.2,
        "never_substitute_calibrated_threshold": True,
        "config_sha256": sha256_file(root / "configs/stage10/dkpn_clean_v3_seed42_mixedcrop.yaml"),
        "train_script_sha256": sha256_file(Path(__file__)),
        "partial_label_sha256": sha256_file(root / "src/earthquake/stage10/partial_label.py"),
        "dataset_v2_sha256": sha256_file(root / "src/earthquake/stage10/dataset_v2.py"),
        "crop_v2_sha256": sha256_file(root / "src/earthquake/stage10/crop_v2.py"),
        "v2_checkpoint_loaded": False,
        "confirm_read": False,
        "val_crop": "s_centered",
        "train_catalog_sha256": sha256_file(out / "train_online_catalog.parquet"),
        "val_subset_sha256": sha256_file(out / "val_subset.parquet") if (out / "val_subset.parquet").is_file() else None,
    }
    save_json(hashes, out / "run_hashes.json")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_dkpn_random().to(device)
    ensure_dir(out / "checkpoints")
    init_payload = {"model": model.state_dict(), "seed": seed, "epoch": -1, "init": "random", "v2_loaded": False}
    torch.save(init_payload, out / "checkpoints" / "init.pt")
    hashes["init_pt_sha256"] = sha256_file(out / "checkpoints" / "init.pt")
    save_json(hashes, out / "run_hashes.json")
    nvis = torch.cuda.device_count() if device.type == "cuda" else 1
    hashes["data_parallel_ngpu"] = int(nvis)
    hashes["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    hashes["per_rank_batch_size"] = int(cfg["batch_size"]) // max(nvis, 1)
    hashes["global_batch_size"] = int(cfg["batch_size"])
    save_json(hashes, out / "run_hashes.json")
    if nvis >= 2:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(cfg["lr_factor"]), patience=int(cfg["lr_patience"])
    )
    batch_sz = int(cfg["batch_size"])
    val_loader, val_ds = v2h.make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
    init_metrics = eval_dev_fixed0p2(model, val_loader, device)
    save_json({"init_eval_height_0p2": init_metrics, "confirm_read": False}, out / "init_eval.json")
    history = []
    best_loss = 1e9
    best_metric = -1.0
    ever_valid_best = False
    optimizer_steps = 0
    windows_seen = 0
    first500: list[float] = []
    kinds_per_epoch: list[set[str]] = []
    bg_pseudo_n = 0
    nan_inf = False
    last_batch = None
    (out / "TRAIN.RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    (out / "TRAIN.PID").write_text(str(os.getpid()))
    max_ep = 3  # HARD STOP. Never 30.
    try:
        for ep in range(max_ep):
            train_loader, ds = v2h.make_loader(
                cat, wave, batch=batch_sz, workers=workers, augment=True, seed=seed + ep, shuffle=False, balanced=True, num_samples=len(cat), drop_last=True
            )
            ds.set_virtual_epoch(ep)
            model.train()
            tr_losses = []
            gnorms = []
            vis_p = vis_s = 0
            kinds: dict[str, int] = {}
            pmeans = []
            n_eff = 0
            t0 = __import__("time").time()
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                try:
                    loss, logits = v2h.batch_loss(model, batch, device, amp=bool(cfg.get("amp", True)))
                except RuntimeError as e:
                    nan_inf = True
                    raise
                if not torch.isfinite(loss):
                    nan_inf = True
                    raise RuntimeError("non-finite loss")
                loss.backward()
                g = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
                if not torch.isfinite(g):
                    nan_inf = True
                    raise RuntimeError("non-finite grad")
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
                for i, k in enumerate(batch["crop_kind"]):
                    kinds[k] = kinds.get(k, 0) + 1
                    if k == "background" and not bool(batch["is_noise"][i]):
                        if float(batch["n_pos"][i].sum()) > 1e-6:
                            bg_pseudo_n += 1
                with torch.no_grad():
                    pr = torch.softmax(logits.float(), dim=1)
                    pmeans.append([float(pr[:, i].mean().cpu()) for i in range(3)])
                last_batch = batch
            kinds_per_epoch.append(set(kinds))
            val_loader, val_ds = v2h.make_loader(val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False)
            vm = eval_dev_fixed0p2(model, val_loader, device)
            diag = {}
            if last_batch is not None:
                try:
                    diag = group_losses_and_phase_n_grad(model, last_batch, device)
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
                "precision@0.1s_height0p2": vm["metrics_height_0p2"]["precision@0.1s"],
                "recall@0.1s_height0p2": vm["metrics_height_0p2"]["recall@0.1s"],
                "f1@0.1s_height0p2": vm["metrics_height_0p2"]["f1@0.1s"],
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
                "bg_pseudo_n_windows": bg_pseudo_n,
                "loss_groups": {k: diag[k] for k in diag if k.startswith("L_")},
                "phase_over_n_grad": diag.get("phase_over_n_grad"),
                "sec": __import__("time").time() - t0,
                "utc": datetime.now(timezone.utc).isoformat(),
                "confirm_read": False,
                "valid_best_fixed0p2": valid,
                "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                "metric_name_fixed0p2": FIXED_0P2_METRIC,
                "metric_name_calibrated": "val_s_f1_calibrated",
                "nan_inf": nan_inf,
            }
            history.append(rec)
            save_json({"history": history, "hashes": hashes, "init_eval": init_metrics}, out / "train_history.json")
            raw = _raw_model(model)
            epoch_meta = checkpoint_metadata(
                virtual_epoch=ep,
                metric_name=FIXED_0P2_METRIC,
                metric_value=vm["f1"],
                init_metric=init_metrics["f1"],
                probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                valid_best=False,
                extra={"kind": "epoch_snapshot", "val_loss": vm["val_loss"]},
            )
            payload = {
                "model": raw.state_dict(),
                "epoch": ep,
                "opt": opt.state_dict(),
                "seed": seed,
                "optimizer_steps": optimizer_steps,
                "windows_seen": windows_seen,
                "v2_loaded": False,
                **epoch_meta,
            }
            torch.save(payload, out / "checkpoints" / epoch_checkpoint_name(ep))
            torch.save(payload, out / "checkpoints" / "last.pt")
            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]
                torch.save({"model": raw.state_dict(), "epoch": ep, "val_loss": best_loss, "valid_best": False, "kind": "best_loss"}, out / "checkpoints" / "best_loss.pt")
            if valid:
                ever_valid_best = True
                best_metric = vm["f1"]
                best_payload = {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "val_s_f1": best_metric,
                    "val_s_f1_fixed0p2": best_metric,
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
                    "model": torch.load(out / "checkpoints" / "init.pt", map_location="cpu")["model"],
                    "epoch": -1,
                    "val_s_f1_fixed0p2": init_metrics["f1"],
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
        ok, reasons = pilot_gate_v3(history, f500, kinds_per_epoch, bg_pseudo_n, nan_inf)
        gate = {
            "ok": ok,
            "reasons": reasons,
            "first500_loss": f500,
            "official_height": 0.2,
            "never_used_calibrated_threshold": True,
            "hard_stop": True,
            "continued_to_30": False,
            "confirm_read": False,
        }
        save_json(gate, out / "pilot_gate.json")
        if ok:
            (out / "PILOT.PASSED").write_text(datetime.now(timezone.utc).isoformat() + "\nwait_for_explicit_approval_before_30_epochs\n")
        else:
            (out / "PILOT.FAILED").write_text(json.dumps(gate, indent=2))
        print("PILOT", json.dumps(gate, indent=2), flush=True)
    except Exception as e:
        (out / "TRAIN.FAILED").write_text(repr(e))
        raise
    finally:
        if (out / "TRAIN.RUNNING").exists():
            (out / "TRAIN.RUNNING").unlink()
        if (out / "TRAIN.PID").exists():
            (out / "TRAIN.PID").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "status", "resume"], required=True)
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--gpus", default=None, help="comma-separated physical GPU ids (all must be idle)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.mode not in {"pilot", "status", "resume"}:
        raise SystemExit("v3 supports --mode pilot|status|resume")
    cfg = load_yaml_config(ROOT / args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    seed = int(cfg["seed"])
    out = ensure_dir(Path(cfg["out_dir"]))
    if args.mode == "status":
        print(json.dumps(status_dict(out_dir=out), indent=2))
        return
    if out.resolve() in {V2_DIR.resolve(), V2_FROZEN.resolve(), BUGGY.resolve()}:
        raise SystemExit("refusing to write into v2/buggy directory")
    if args.mode == "pilot":
        if (out / "PILOT.PASSED").is_file() or (out / "PILOT.FAILED").is_file():
            raise SystemExit("v3 pilot already finished; will not relaunch")
        if (out / "checkpoints" / "last.pt").is_file():
            raise SystemExit("v3 checkpoints exist; refusing overwrite")
    elif args.mode == "resume":
        if not (out / "PILOT.PASSED").is_file():
            raise SystemExit("resume requires PILOT.PASSED")
        if (out / "PILOT.FAILED").is_file():
            raise SystemExit("resume refused: PILOT.FAILED")
        if not (out / "checkpoints" / "epoch_2.pt").is_file() and not (out / "checkpoints" / "last.pt").is_file():
            raise SystemExit("resume requires epoch_2.pt or last.pt")
    idle = idle_gpu_indices()
    usable = allocatable_gpu_indices()
    already = bool(training_process_running())
    flag = (out / "TRAIN.RUNNING").is_file() or (V2_DIR / "TRAIN.RUNNING").is_file()
    launch_mode = args.mode
    if args.gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        req = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
        occ = [i for i in req if i not in usable]
        if already or flag:
            reason = "training_already_running"
        elif occ:
            reason = f"requested_gpus_occupied_{occ}"
        elif len(req) < MIN_GPUS:
            reason = f"need_min_gpus_{MIN_GPUS}_have_{len(req)}"
        else:
            reason = None
    else:
        reason = launch_block_reason(mode=launch_mode, n_idle=len(idle), already_training=already, train_running_flag=flag)
        if reason is None:
            if len(idle) < MIN_GPUS:
                reason = f"need_min_gpus_{MIN_GPUS}_have_{len(idle)}"
            else:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in idle[:4])
    if reason:
        save_json({"ok": False, "reason": reason, "idle": idle, "mode": args.mode}, ROOT / "artifacts/results/stage10/dkpn_v3_last_launch_refuse.json")
        print(json.dumps({"refused": reason, "n_idle": len(idle), "idle": idle}))
        raise SystemExit(3 if "need_min_gpus" in str(reason) else 4)
    if args.mode == "pilot":
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda required")
    if args.mode == "pilot":
        frozen = dict(cfg)
        frozen["runtime_gpus"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        frozen["runtime_global_batch_size"] = int(cfg["batch_size"])
        frozen["hard_stop_after_pilot"] = True
        frozen["never_auto_continue_30"] = True
        frozen["confirm_read"] = False
        (out / "frozen_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
        shutil.copy2(ROOT / args.config, out / "source_config.yaml")
    inst = resolve_instance_root()
    from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource

    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))
    save_json(
        {"confirm_read": False, "confirm_waveforms": False, "run": cfg["run_name"], "mode": args.mode},
        ROOT / "artifacts/results/stage10/dkpn_v3/confirm_guard.json",
    )
    v2h = _load_v2_helpers()
    if args.mode == "pilot":
        run_pilot(cfg, wave, device, seed, out, workers, ROOT, v2h)
        return
    from earthquake.stage10.v3_continue import run_resume

    run_resume(
        cfg,
        wave,
        device,
        seed,
        out,
        workers,
        ROOT,
        v2h,
        eval_dev_fixed0p2=eval_dev_fixed0p2,
        group_losses_and_phase_n_grad=group_losses_and_phase_n_grad,
        raw_model=_raw_model,
    )


if __name__ == "__main__":
    main()
