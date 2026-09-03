"""Exact resume of v3 mixed-crop from completed virtual epoch 2 → max 30.

Does not re-init, reset optimizer/LR, change global batch, loss, or crop schedule.
Does not overwrite PILOT.PASSED or epoch_0/1/2.pt.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from earthquake.config import artifacts_dir, save_json
from earthquake.stage10.checkpoint_policy import (
    FIXED_0P2_METRIC,
    OFFICIAL_PROBABILITY_THRESHOLD,
    best_metric_filename,
    checkpoint_metadata,
    epoch_checkpoint_name,
    is_valid_best,
)
from earthquake.stage10.dkpn_clean import build_dkpn_random
from earthquake.stage10.nonfinite import NonfiniteError, check_finite
from earthquake.stage10.v3_resume_audit import audit_resume_checkpoint, verify_semantic_hashes
from earthquake.utils import ensure_dir

PILOT_EPOCH_FILES = {"epoch_0.pt", "epoch_1.pt", "epoch_2.pt", "init.pt"}


GATE_KEYS = (
    "scan_complete_finite",
    "old_fp16_eval_reproduced",
    "new_fp32_eval_same_batch_ok",
    "smoke_4gpu_lifecycle_past_fail_plus_1000",
    "eval_passed_twice",
    "train_loss_act_grad_finite",
    "no_catastrophic_f1_oracle_drop",
    "no_silent_skip",
    "pytest_passed",
    "confirm_waveforms_unread",
)


OVERFLOW_FLAG = "V3.FAILED_FP16_ACTIVATION_OVERFLOW"


def resume_blocked_nonfinite_e3(out: Path) -> str | None:
    """Refuse official resume on the frozen v3 dir.

    FP16 train-forward overflow is confirmed. Changing train dtype (BF16/FP32)
    is not a math-equivalent eval/loss fix; do not resume epoch_2 as the paper run.
    """
    if (out / OVERFLOW_FLAG).is_file():
        return "V3.FAILED_FP16_ACTIVATION_OVERFLOW_do_not_resume_epoch2"
    if not (out / "TRAIN.FAILED_NONFINITE_E3").is_file():
        return None
    gate_p = out / "forensic_nonfinite_e3" / "RESUME.GATE.json"
    if not gate_p.is_file():
        return "TRAIN.FAILED_NONFINITE_E3_gate_incomplete"
    try:
        gate = json.loads(gate_p.read_text())
    except Exception:
        return "TRAIN.FAILED_NONFINITE_E3_gate_incomplete"
    missing = [k for k in GATE_KEYS if not gate.get(k)]
    if missing:
        return "TRAIN.FAILED_NONFINITE_E3_gate_incomplete"
    return None


def _fail_train(out: Path, reason: str, extra: dict | None = None) -> None:
    payload = {"reason": reason, "confirm_read": False, **(extra or {})}
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        payload["traceback"] = tb
    else:
        payload["traceback"] = "".join(traceback.format_stack())
    save_json(payload, out / "TRAIN.FAILED.json")
    (out / "TRAIN.FAILED").write_text(json.dumps(payload, indent=2))
    print("TRAIN.FAILED", json.dumps(payload, default=str), flush=True)


def batch_divisible_by_ngpu(batch_sz: int, nvis: int) -> bool:
    """DataParallel + BatchNorm1d requires per-rank batch >= 2 and even split."""
    if int(nvis) < 2:
        return True
    return int(batch_sz) % int(nvis) == 0 and (int(batch_sz) // int(nvis)) >= 2


def _archive_failed_flag(out: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in ("TRAIN.FAILED", "TRAIN.FAILED.json"):
        p = out / name
        if p.is_file():
            p.rename(out / f"{name}.archived_{ts}")


def tag_oscillation(prev: dict | None, cur_f1: float, cur_oracle: float | None) -> str | None:
    if prev is None:
        return None
    f1_down = float(cur_f1) < float(prev["val_s_f1_fixed0p2"]) - 1e-6
    ora_c = float(cur_oracle if cur_oracle is not None else 0)
    ora_p = float(prev.get("oracle_acc@0.5s") or 0)
    ora_down = ora_c < ora_p - 0.02
    if f1_down and ora_down:
        return "representation_regression"
    if f1_down and not ora_down:
        return "confidence_oscillation"
    return None


def consecutive_collapse_fail(history: list[dict]) -> bool:
    """Two consecutive epochs: no-S>0.95, F1~0, and oracle clearly dropping."""
    if len(history) < 2:
        return False

    def bad(h: dict) -> bool:
        return float(h.get("frac_no_s_peak_height0p2") or 0) > 0.95 and float(h.get("val_s_f1_fixed0p2") or 1) < 0.02

    a, b = history[-2], history[-1]
    if not (bad(a) and bad(b)):
        return False
    oa, ob = float(a.get("oracle_acc@0.5s") or 0), float(b.get("oracle_acc@0.5s") or 0)
    if len(history) >= 3:
        op = float(history[-3].get("oracle_acc@0.5s") or 0)
        return (oa < op - 0.05) and (ob < oa - 0.05)
    return ob < oa - 0.05


def freeze_large_val(n_traces: int, seed: int, out: Path) -> pd.DataFrame:
    """Event-balanced S-labelled Stage6 dev. Never writes val_subset.parquet."""
    from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names

    path = out / "val_large.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    dev_tr = set(load_full_trace_names("stage6_dev"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    ev = events[events["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK large-val ∩ confirm")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ev = ev[s.notna()].copy()
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in ev.groupby("event_id", sort=False):
        picks.append(g.iloc[int(rng.integers(0, len(g)))])
    sub = pd.DataFrame(picks)
    if len(sub) > n_traces:
        sub = sub.sample(n=n_traces, random_state=seed)
    sub = sub.reset_index(drop=True)
    sub["is_noise"] = False
    sub.to_parquet(path, index=False)
    save_json(
        {
            "n_traces": int(len(sub)),
            "n_events": int(sub["event_id"].nunique()),
            "source": "stage6_dev_S_labelled_event_balanced",
            "confirm_read": False,
            "not_used_for_checkpoint_selection": True,
        },
        out / "val_large_manifest.json",
    )
    return sub


def confirm_was_read(out: Path, val_cat: pd.DataFrame) -> bool:
    from earthquake.stage6.full_splits import load_full_event_ids

    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    if set(val_cat["event_id"].astype(str)) & confirm:
        return True
    train_p = out / "train_online_catalog.parquet"
    if train_p.is_file():
        cols = list(pd.read_parquet(train_p).columns)
        if "event_id" in cols:
            ids = set(pd.read_parquet(train_p, columns=["event_id"])["event_id"].astype(str))
            if ids & confirm:
                return True
    return False


def _replay_scheduler(opt, cfg, history: list[dict]):
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(cfg["lr_factor"]), patience=int(cfg["lr_patience"])
    )
    for h in history:
        sch.step(float(h["val_loss"]))
    return sch


def run_resume(
    cfg: dict,
    wave,
    device,
    seed: int,
    out: Path,
    workers: int,
    root: Path,
    v2h,
    *,
    eval_dev_fixed0p2: Callable,
    group_losses_and_phase_n_grad: Callable,
    raw_model: Callable,
) -> None:
    if not (out / "PILOT.PASSED").is_file():
        raise SystemExit("resume requires PILOT.PASSED")
    if (out / "PILOT.FAILED").is_file():
        raise SystemExit("resume refused: PILOT.FAILED")
    e3_block = resume_blocked_nonfinite_e3(out)
    if e3_block:
        (out / "RESUME.BLOCKED").write_text(
            json.dumps(
                {
                    "reason": e3_block,
                    "confirm_read": False,
                    "note": "frozen v3 scene must not be overwritten; 4-GPU smoke required before any resume",
                },
                indent=2,
            )
        )
        raise SystemExit(f"RESUME.BLOCKED {e3_block}")
    _archive_failed_flag(out)
    ck_path = out / "checkpoints" / "epoch_2.pt"
    if not ck_path.is_file():
        ck_path = out / "checkpoints" / "last.pt"
    audit = audit_resume_checkpoint(ck_path)
    save_json(audit, out / "RESUME.AUDIT.json")
    if audit.get("block"):
        (out / "RESUME.BLOCKED").write_text(json.dumps(audit, indent=2))
        raise SystemExit("RESUME.BLOCKED")
    run_hashes = json.loads((out / "run_hashes.json").read_text())
    sem = verify_semantic_hashes(
        run_hashes,
        {
            "partial_label_sha256": root / "src/earthquake/stage10/partial_label.py",
            "dataset_v2_sha256": root / "src/earthquake/stage10/dataset_v2.py",
            "crop_v2_sha256": root / "src/earthquake/stage10/crop_v2.py",
            "config_sha256": root / "configs/stage10/dkpn_clean_v3_seed42_mixedcrop.yaml",
            "train_catalog_sha256": out / "train_online_catalog.parquet",
            "val_subset_sha256": out / "val_subset.parquet",
        },
    )
    save_json(sem, out / "RESUME.HASH_CHECK.json")
    if not sem["ok"]:
        (out / "RESUME.BLOCKED").write_text(json.dumps({"reason": "semantic_hash_mismatch", **sem}, indent=2))
        raise SystemExit("RESUME.BLOCKED hash mismatch")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    hist_doc = json.loads((out / "train_history.json").read_text())
    history = list(hist_doc["history"])
    if len(history) < 3 or int(history[2]["virtual_epoch"]) != 2:
        (out / "RESUME.BLOCKED").write_text("history does not contain completed epoch 2\n")
        raise SystemExit("RESUME.BLOCKED history")
    # If a later epoch_N already exists (crash restart), continue after the last complete one.
    existing_eps = []
    for p in (out / "checkpoints").glob("epoch_*.pt"):
        try:
            existing_eps.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    completed = max(existing_eps) if existing_eps else int(ck.get("virtual_epoch", 2))
    start_ep = completed + 1
    if start_ep < 3:
        start_ep = 3
    if completed > 2:
        later = out / "checkpoints" / epoch_checkpoint_name(completed)
        if later.is_file():
            later_audit = audit_resume_checkpoint(later, expect_epoch=completed)
            if later_audit.get("ok"):
                ck_path = later
                audit = later_audit
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                history = [h for h in history if int(h["virtual_epoch"]) <= completed]
            else:
                ck_path = out / "checkpoints" / "epoch_2.pt"
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                start_ep = 3
    cat = v2h.annotate_online_catalog(pd.read_parquet(out / "train_online_catalog.parquet"))
    val_meta = pd.read_parquet(out / "val_subset.parquet")
    val_cat = v2h.annotate_online_catalog(val_meta)
    val_cat = val_cat.copy()
    val_cat["crop_kind"] = "s_centered"
    if confirm_was_read(out, val_cat):
        _fail_train(out, "confirm_read")
        return
    init_eval = json.loads((out / "init_eval.json").read_text())["init_eval_height_0p2"]
    init_f1 = float(init_eval["f1"])
    torch.manual_seed(seed + start_ep)
    np.random.seed(seed + start_ep)
    torch.cuda.manual_seed_all(seed + start_ep)
    model = build_dkpn_random().to(device)
    model.load_state_dict(ck["model"])
    nvis = torch.cuda.device_count() if device.type == "cuda" else 1
    pilot_n = int(run_hashes.get("data_parallel_ngpu", nvis))
    if nvis != pilot_n:
        save_json(
            {
                "pilot_data_parallel_ngpu": pilot_n,
                "resume_data_parallel_ngpu": nvis,
                "global_batch_size_unchanged": int(cfg["batch_size"]) == int(run_hashes.get("global_batch_size", cfg["batch_size"])),
                "optimizer_restored": True,
                "note": "DataParallel device count changed by explicit request; module param order unchanged so Adam state still loads",
            },
            out / "RESUME.GPU_WIDTH.json",
        )
        if nvis < 2:
            (out / "RESUME.BLOCKED").write_text(
                json.dumps({"reason": "ngpu_too_small", "now": nvis, "pilot": pilot_n})
            )
            raise SystemExit("RESUME.BLOCKED ngpu")
    batch_sz = int(cfg["batch_size"])
    if not batch_divisible_by_ngpu(batch_sz, nvis):
        (out / "RESUME.BLOCKED").write_text(
            json.dumps(
                {
                    "reason": "batch_not_divisible_or_per_rank_lt_2",
                    "global_batch_size": batch_sz,
                    "nvis": nvis,
                    "note": "DKPN BatchNorm1d+AMP is unstable when DataParallel splits 32 across 6 GPUs",
                },
                indent=2,
            )
        )
        raise SystemExit("RESUME.BLOCKED ngpu_batch")
    if nvis >= 2:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    try:
        opt.load_state_dict(ck["opt"])
    except Exception as e:
        (out / "RESUME.BLOCKED").write_text(f"optimizer_load_failed: {e!r}\n")
        raise SystemExit("RESUME.BLOCKED optimizer")
    sch = _replay_scheduler(opt, cfg, history)
    if "scheduler" in ck:
        try:
            sch.load_state_dict(ck["scheduler"])
        except Exception:
            pass  # keep replay
    if batch_sz != int(run_hashes.get("global_batch_size", batch_sz)):
        (out / "RESUME.BLOCKED").write_text("global batch changed\n")
        raise SystemExit("RESUME.BLOCKED batch")
    optimizer_steps = int(ck["optimizer_steps"])
    windows_seen = int(ck["windows_seen"])
    best_loss = min(float(h["val_loss"]) for h in history)
    best_metric = max(float(h["val_s_f1_fixed0p2"]) for h in history)
    bad = 0
    # count consecutive non-improvements already in history after the best epoch
    best_ep = max(history, key=lambda h: float(h["val_s_f1_fixed0p2"]))["virtual_epoch"]
    for h in history:
        if int(h["virtual_epoch"]) > int(best_ep) and float(h["val_s_f1_fixed0p2"]) <= best_metric + 1e-12:
            bad += 1
    patience = int(cfg.get("patience", 8))
    bg_pseudo_n = 0
    nan_inf = False
    max_ep = 30
    ensure_dir(out / "checkpoints")
    (out / "TRAIN.RUNNING").write_text(datetime.now(timezone.utc).isoformat())
    (out / "TRAIN.PID").write_text(str(os.getpid()))
    save_json(
        {
            "resume_from": str(ck_path),
            "resume_sha256": audit.get("sha256"),
            "completed_epoch": int(ck.get("virtual_epoch", completed)),
            "next_epoch": start_ep,
            "optimizer_restored": True,
            "optimizer_steps": optimizer_steps,
            "adam_step": audit.get("adam_step"),
            "lr": float(opt.param_groups[0]["lr"]),
            "global_batch_size": batch_sz,
            "nvis": nvis,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "confirm_read": False,
            "pilot_passed_preserved": True,
            "scheduler_replayed_from_history": True,
            "amp_scaler": "not_used_autocast_only",
            "rng_not_in_pilot_ckpt": True,
            "sampler_crop_cycle": "set_virtual_epoch(ep); crop=(hash+ep)%3",
            "semantic_hashes_ok": True,
        },
        out / "RESUME.OK.json",
    )
    try:
        for ep in range(start_ep, max_ep):
            if epoch_checkpoint_name(ep) in PILOT_EPOCH_FILES:
                _fail_train(out, "refusing_overwrite_pilot_epoch_file", {"epoch": ep})
                return
            train_loader, ds = v2h.make_loader(
                cat,
                wave,
                batch=batch_sz,
                workers=workers,
                augment=True,
                seed=seed + ep,
                shuffle=False,
                balanced=True,
                num_samples=len(cat),
                drop_last=True,
            )
            ds.set_virtual_epoch(ep)
            model.train()
            tr_losses, gnorms, pmeans = [], [], []
            vis_p = vis_s = n_eff = n_zero = 0
            kinds: dict[str, int] = {}
            last_batch = None
            t0 = __import__("time").time()
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                try:
                    loss, logits = v2h.batch_loss(
                        model, batch, device, amp=bool(cfg.get("amp", True)), optimizer_step=optimizer_steps
                    )
                except NonfiniteError as e:
                    _fail_train(
                        out,
                        "nonfinite",
                        {"epoch": ep, "nvis": nvis, "batch_size": batch_sz, **e.to_dict()},
                    )
                    return
                except RuntimeError as e:
                    msg = repr(e)
                    nan_inf = "nan" in msg.lower() or "inf" in msg.lower() or "non-finite" in msg.lower()
                    _fail_train(
                        out,
                        "nan_inf_forward" if nan_inf else "forward_runtime_error",
                        {
                            "epoch": ep,
                            "error": msg,
                            "traceback": traceback.format_exc(),
                            "nvis": nvis,
                            "batch_size": batch_sz,
                            "phase": "train_forward",
                            "model.training": True,
                        },
                    )
                    return
                try:
                    loss.backward()
                except RuntimeError:
                    _fail_train(out, "nonfinite", {"epoch": ep, "phase": "backward", "traceback": traceback.format_exc()})
                    return
                g = torch.sqrt(
                    sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None)
                )
                try:
                    check_finite(g, "backward", "non-finite grad", model=model, device=device, optimizer_step=optimizer_steps, batch=batch)
                except NonfiniteError as e:
                    _fail_train(out, "nonfinite", {"epoch": ep, **e.to_dict()})
                    return
                gnorms.append(float(g.detach().cpu()))
                try:
                    opt.step()
                except RuntimeError:
                    _fail_train(out, "nonfinite", {"epoch": ep, "phase": "optimizer_step", "traceback": traceback.format_exc()})
                    return
                optimizer_steps += 1
                windows_seen += int(batch["x"].shape[0])
                tr_losses.append(float(loss.detach().cpu()))
                vis_p += int(batch["vis_p"].sum())
                vis_s += int(batch["vis_s"].sum())
                wsum = (batch["p_pos"] + batch["s_pos"] + batch["n_pos"] + batch["not_p"] + batch["not_s"]).sum(
                    dim=-1
                )
                n_eff += int((wsum > 0).sum().item())
                n_zero += int((wsum <= 0).sum().item())
                for i, k in enumerate(batch["crop_kind"]):
                    kinds[k] = kinds.get(k, 0) + 1
                    if k == "background" and not bool(batch["is_noise"][i]) and float(batch["n_pos"][i].sum()) > 1e-6:
                        bg_pseudo_n += 1
                with torch.no_grad():
                    pr = torch.softmax(logits.float(), dim=1)
                    pmeans.append([float(pr[:, i].mean().cpu()) for i in range(3)])
                last_batch = batch
            if bg_pseudo_n > 0:
                _fail_train(out, "bg_pseudo_n", {"bg_pseudo_n": bg_pseudo_n, "epoch": ep})
                return
            need = {"p_centered", "s_centered", "background"}
            if not need.issubset(set(kinds)):
                _fail_train(out, "missing_crop_kinds", {"kinds": kinds, "epoch": ep})
                return
            val_loader, _ = v2h.make_loader(
                val_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False
            )
            try:
                vm = eval_dev_fixed0p2(model, val_loader, device)
            except NonfiniteError as e:
                _fail_train(out, "nonfinite", {"epoch": ep, **e.to_dict()})
                return
            finally:
                model.train()
            diag: dict[str, Any] = {}
            if last_batch is not None:
                try:
                    diag = group_losses_and_phase_n_grad(model, last_batch, device)
                except Exception as e:
                    diag = {"phase_over_n_grad": None, "diag_error": repr(e)}
            prev = history[-1]
            ora = vm.get("oracle_argmax_accuracy@0.5s")
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
                "init_s_f1_fixed0p2": init_f1,
                "metrics_height_0p2": vm["metrics_height_0p2"],
                "precision@0.1s_height0p2": vm["metrics_height_0p2"]["precision@0.1s"],
                "recall@0.1s_height0p2": vm["metrics_height_0p2"]["recall@0.1s"],
                "f1@0.1s_height0p2": vm["metrics_height_0p2"]["f1@0.1s"],
                "precision@0.5s_height0p2": vm["metrics_height_0p2"]["precision@0.5s"],
                "recall@0.5s_height0p2": vm["metrics_height_0p2"]["recall@0.5s"],
                "f1@0.5s_height0p2": vm["metrics_height_0p2"]["f1@0.5s"],
                "picks_per_trace_height0p2": vm["metrics_height_0p2"]["picks_per_trace"],
                "frac_no_s_peak_height0p2": vm["metrics_height_0p2"]["frac_no_s_peak"],
                "oracle_acc@0.5s": ora,
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
                "loss_groups": {k: diag[k] for k in diag if str(k).startswith("L_")},
                "phase_over_n_grad": diag.get("phase_over_n_grad"),
                "confidence_tag": tag_oscillation(prev, vm["f1"], ora),
                "sec": __import__("time").time() - t0,
                "utc": datetime.now(timezone.utc).isoformat(),
                "confirm_read": False,
                "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                "metric_name_fixed0p2": FIXED_0P2_METRIC,
                "nan_inf": nan_inf,
                "resumed": True,
                "val_large_s_f1_fixed0p2": None,
            }
            rec["valid_best_vs_init"] = is_valid_best(metric=vm["f1"], init_metric=init_f1)
            rec["minted_new_best_metric"] = bool(rec["valid_best_vs_init"] and vm["f1"] > best_metric)
            rec["valid_best_fixed0p2"] = rec["minted_new_best_metric"]
            if (ep + 1) % 3 == 0:
                large = freeze_large_val(2048, seed, out)
                large_cat = v2h.annotate_online_catalog(large)
                large_cat = large_cat.copy()
                large_cat["crop_kind"] = "s_centered"
                ll, _ = v2h.make_loader(
                    large_cat, wave, batch=batch_sz, workers=0, augment=False, seed=seed, shuffle=False, balanced=False
                )
                try:
                    lvm = eval_dev_fixed0p2(model, ll, device)
                except NonfiniteError as e:
                    _fail_train(out, "nonfinite", {"epoch": ep, "eval": "val_large", **e.to_dict()})
                    return
                finally:
                    model.train()
                rec["val_large_s_f1_fixed0p2"] = lvm["f1"]
                rec["val_large_oracle_acc@0.5s"] = lvm.get("oracle_argmax_accuracy@0.5s")
                rec["val_large_not_used_for_checkpoint"] = True
            history.append(rec)
            save_json(
                {
                    "history": history,
                    "hashes": hist_doc.get("hashes", run_hashes),
                    "init_eval": init_eval,
                    "resumed_from_epoch": 2,
                },
                out / "train_history.json",
            )
            if rec["all_n_collapse"] and (ep + 1) % 3 == 0:
                _fail_train(out, "all_n_collapse_at_crop_cycle_end", {"epoch": ep})
                return
            if consecutive_collapse_fail(history):
                _fail_train(out, "two_consecutive_noS_and_f1_and_oracle_drop", {"epoch": ep})
                return
            if confirm_was_read(out, val_cat):
                _fail_train(out, "confirm_read", {"epoch": ep})
                return
            raw = raw_model(model)
            payload = {
                "model": raw.state_dict(),
                "epoch": ep,
                "opt": opt.state_dict(),
                "scheduler": sch.state_dict(),
                "optimizer_steps": optimizer_steps,
                "windows_seen": windows_seen,
                "seed": seed,
                "v2_loaded": False,
                "rng_torch": torch.get_rng_state(),
                "virtual_epoch": ep,
                "probability_threshold": OFFICIAL_PROBABILITY_THRESHOLD,
                **checkpoint_metadata(
                    virtual_epoch=ep,
                    metric_name=FIXED_0P2_METRIC,
                    metric_value=vm["f1"],
                    init_metric=init_f1,
                    probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                    valid_best=False,
                    extra={
                        "kind": "epoch_snapshot",
                        "val_loss": vm["val_loss"],
                        "oracle_acc@0.5s": rec["oracle_acc@0.5s"],
                    },
                ),
            }
            dest = out / "checkpoints" / epoch_checkpoint_name(ep)
            if dest.name in PILOT_EPOCH_FILES:
                _fail_train(out, "refusing_overwrite_pilot_checkpoint", {"path": str(dest)})
                return
            torch.save(payload, dest)
            torch.save(payload, out / "checkpoints" / "last.pt")
            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]
                torch.save(
                    {"model": raw.state_dict(), "epoch": ep, "val_loss": best_loss, "kind": "best_loss"},
                    out / "checkpoints" / "best_loss.pt",
                )
            if rec["minted_new_best_metric"]:
                best_metric = vm["f1"]
                bad = 0
                best_payload = {
                    "model": raw.state_dict(),
                    "epoch": ep,
                    "val_s_f1": best_metric,
                    "val_s_f1_fixed0p2": best_metric,
                    "oracle_acc@0.5s": rec["oracle_acc@0.5s"],
                    **checkpoint_metadata(
                        virtual_epoch=ep,
                        metric_name=FIXED_0P2_METRIC,
                        metric_value=best_metric,
                        init_metric=init_f1,
                        probability_threshold=OFFICIAL_PROBABILITY_THRESHOLD,
                        valid_best=True,
                        extra={
                            "height": 0.2,
                            "oracle_acc@0.5s": rec["oracle_acc@0.5s"],
                            "resume_from_sha256": audit.get("sha256"),
                        },
                    ),
                }
                torch.save(best_payload, out / "checkpoints" / "best_metric.pt")
                torch.save(best_payload, out / "checkpoints" / best_metric_filename("fixed0p2"))
            else:
                if vm["f1"] <= best_metric + 1e-12:
                    bad += 1
            save_json(
                {
                    "valid_best": True,
                    "best_metric_fixed0p2": best_metric,
                    "init_s_f1_fixed0p2": init_f1,
                    "probability_threshold": 0.2,
                    "early_stop_bad": bad,
                    "patience": patience,
                    "f1_all_zero_cannot_mint_valid_best": True,
                    "calibrated_best_not_written": True,
                },
                out / "checkpoints" / "best_metric_status.json",
            )
            sch.step(vm["val_loss"])
            print(json.dumps({k: rec[k] for k in rec if k != "metrics_height_0p2"}, default=str), flush=True)
            if bad >= patience:
                rec["early_stop"] = True
                save_json(
                    {"history": history, "early_stop": True, "stopped_epoch": ep, "hashes": hist_doc.get("hashes")},
                    out / "train_history.json",
                )
                break
        (out / "TRAIN.DONE").write_text(datetime.now(timezone.utc).isoformat())
        print("TRAIN.DONE calling full-dev stop gate", flush=True)
        import subprocess

        sg = subprocess.run(
            [sys.executable, str(root / "scripts/run_stage10_v3_full_dev_stop_gate.py"), "--out-dir", str(out)],
            check=False,
        )
        save_json({"stop_gate_returncode": int(sg.returncode), "confirm_read": False}, out / "STOP_GATE.INVOKE.json")
    except Exception as e:
        (out / "TRAIN.FAILED").write_text(repr(e))
        raise
    finally:
        if (out / "TRAIN.RUNNING").exists():
            (out / "TRAIN.RUNNING").unlink()
        if (out / "TRAIN.PID").exists():
            (out / "TRAIN.PID").unlink(missing_ok=True)
