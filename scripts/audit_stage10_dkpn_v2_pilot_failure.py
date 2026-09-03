#!/usr/bin/env python
"""v2 pilot failure audit. Does not resume last.pt, does not read confirm waveforms."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage10.crop_v2 import CROP_CYCLE, FSTAB, IN_SAMPLES, WIN_RAW, crop_kind_start, label_coord, phase_visible
from earthquake.stage10.dataset_v2 import DKPNPartialCropDataset, annotate_online_catalog, build_partial_weights
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits, dkpn_probs
from earthquake.stage10.dkpn_picks import extract_picks, s_f1_at_tolerance
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.utils import ensure_dir

RUN = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42")
ART = ROOT / "artifacts/results/stage10"
PLOT = ART / "v2_pilot_failure_plots"
THRS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def _sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_model(path: Path, device) -> torch.nn.Module:
    m = build_dkpn_random().to(device)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(obj["model"])
    m.eval()
    return m, obj


def collate(batch):
    keys_t = ["x", "p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask", "vis_p", "vis_s", "p_c", "s_c"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_t}
    out["crop_kind"] = [b["crop_kind"] for b in batch]
    out["trace_name"] = [b["trace_name"] for b in batch]
    return out


@torch.no_grad()
def infer_split(model, loader, device) -> dict:
    rows = []
    psn_sum = np.zeros(3, dtype=np.float64)
    n_tok = 0
    for batch in loader:
        x = batch["x"].to(device)
        logits = dkpn_logits(model, x)
        # one softmax only
        pr = torch.softmax(logits.float(), dim=1).cpu().numpy()
        assert pr.shape[1] == 3
        pad = batch["pad_mask"].numpy()
        for i in range(pr.shape[0]):
            s = pr[i, 1]
            valid = pad[i] > 0.5
            s_v = s.copy()
            s_v[~valid] = 0.0
            true = float(batch["s_c"][i])
            vis = bool(batch["vis_s"][i] > 0.5 and true >= 0)
            tmax = int(np.argmax(s_v)) if valid.any() else -1
            ptrue = float(s[int(np.clip(true, 0, len(s) - 1))]) if vis else float("nan")
            peaks02, _, amp02, _ = extract_picks(s_v, thr=0.2)
            rows.append(
                {
                    "trace": batch["trace_name"][i],
                    "kind": batch["crop_kind"][i],
                    "vis_s": vis,
                    "true_s": true,
                    "s_true_loc_p": ptrue,
                    "s_max_p": float(s_v.max()) if valid.any() else 0.0,
                    "s_argmax": tmax,
                    "s_argmax_err": (abs(tmax - true) if vis and tmax >= 0 else float("nan")),
                    "n_peaks_0p2": int(len(peaks02)),
                    "p_mean": float(pr[i, 0, valid].mean()) if valid.any() else 0.0,
                    "s_mean": float(pr[i, 1, valid].mean()) if valid.any() else 0.0,
                    "n_mean": float(pr[i, 2, valid].mean()) if valid.any() else 0.0,
                    "probs": pr[i],
                    "pad": pad[i],
                    "p_c": float(batch["p_c"][i]),
                    "vis_p": bool(batch["vis_p"][i] > 0.5),
                }
            )
            psn_sum += pr[i][:, valid].sum(axis=-1) if valid.any() else np.zeros(3)
            n_tok += int(valid.sum())
    vis = [r for r in rows if r["vis_s"]]
    n_vis = max(len(vis), 1)
    oracle_ok = sum(1 for r in vis if np.isfinite(r["s_argmax_err"]) and r["s_argmax_err"] <= 50)
    oracle_ok01 = sum(1 for r in vis if np.isfinite(r["s_argmax_err"]) and r["s_argmax_err"] <= 10)
    all_n = sum(1 for r in rows if r["s_max_p"] < 0.05 and r["n_mean"] > 0.9)
    no_peak = sum(1 for r in vis if r["n_peaks_0p2"] == 0)
    # wrong peak: has peak(s) but none within 50 samples
    wrong = 0
    for r in vis:
        if r["n_peaks_0p2"] == 0:
            continue
        pk, _, _, _ = extract_picks(r["probs"][1] * (r["pad"] > 0.5), thr=0.2)
        if not any(abs(int(p) - r["true_s"]) <= 50 for p in pk):
            wrong += 1
    sweep = {}
    for thr in THRS:
        preds, trues, vmask = [], [], []
        for r in vis:
            pk, _, amp, _ = extract_picks(r["probs"][1] * (r["pad"] > 0.5), thr=thr)
            if len(pk) == 0:
                preds.append(np.nan)
            else:
                preds.append(float(pk[int(np.argmax(amp))]))
            trues.append(r["true_s"])
            vmask.append(True)
        sweep[str(thr)] = {
            "f1@0.1s": s_f1_at_tolerance(np.asarray(preds), np.asarray(trues), np.asarray(vmask), tol_samples=10)["f1"],
            "f1@0.5s": s_f1_at_tolerance(np.asarray(preds), np.asarray(trues), np.asarray(vmask), tol_samples=50)["f1"],
            "n_with_peak": int(np.isfinite(preds).sum()),
        }
    offset = np.array([r["s_argmax_err"] if np.isfinite(r["s_argmax_err"]) else np.nan for r in vis])
    med_off = float(np.nanmedian(np.array([r["s_argmax"] - r["true_s"] for r in vis if np.isfinite(r["s_argmax_err"])] or [np.nan])))
    return {
        "n": len(rows),
        "n_vis_s": len(vis),
        "s_true_loc_p_mean": float(np.nanmean([r["s_true_loc_p"] for r in vis])) if vis else None,
        "s_max_p_mean": float(np.mean([r["s_max_p"] for r in vis])) if vis else None,
        "s_argmax_err_median": float(np.nanmedian(offset)) if vis else None,
        "n_peaks_0p2_mean": float(np.mean([r["n_peaks_0p2"] for r in vis])) if vis else None,
        "oracle_acc@0.5s": oracle_ok / n_vis,
        "oracle_acc@0.1s": oracle_ok01 / n_vis,
        "frac_allN_like": all_n / max(len(rows), 1),
        "frac_no_s_peak_0p2": no_peak / n_vis,
        "frac_wrong_peak_0p2": wrong / n_vis,
        "psn_token_mean": (psn_sum / max(n_tok, 1)).tolist(),
        "median_signed_argmax_offset_samples": med_off,
        "threshold_sweep": sweep,
        "_rows": rows,
    }


def old_homogeneous_kind(epoch, is_noise, has_p, has_s):
    if is_noise:
        return "noise"
    if has_p and has_s:
        return CROP_CYCLE[int(epoch) % 3]
    if has_p:
        return "p_centered" if epoch % 2 == 0 else "background"
    if has_s:
        return "s_centered" if epoch % 2 == 0 else "background"
    return "background"


def supervision_report(cat: pd.DataFrame) -> dict:
    rng = np.random.default_rng(0)
    n = 12000
    out = {}
    for ep in range(3):
        kinds = {"p_centered": 0, "s_centered": 0, "background": 0, "noise": 0}
        zero = n_as_n = n_noise = gmass_p = gmass_s = nmass = 0
        wsum_phase = wsum_non = 0.0
        for _, r in cat.iterrows():
            is_noise = bool(r.get("is_noise", False))
            p = r.get("p_arrival_sample", np.nan)
            s = r.get("s_arrival_sample", np.nan)
            hp = (not is_noise) and pd.notna(p) and np.isfinite(float(p))
            hs = (not is_noise) and pd.notna(s) and np.isfinite(float(s))
            kind = old_homogeneous_kind(ep, is_noise, hp, hs)
            kinds[kind] = kinds.get(kind, 0) + 1
            pv = float(p) if hp else None
            sv = float(s) if hs else None
            st = crop_kind_start(kind, n, pv, sv, rng)
            vis_p = phase_visible(pv, st)
            vis_s = phase_visible(sv, st)
            # v2 trained weights (old: P+S neither → N)
            pad = np.ones(IN_SAMPLES, np.float32)
            w_new = build_partial_weights(
                n=IN_SAMPLES, p_c=label_coord(pv, st), s_c=label_coord(sv, st),
                vis_p=vis_p, vis_s=vis_s, is_noise=is_noise, pad_mask=pad, sigma=10.0,
                has_p_label=hp, has_s_label=hs,
            )
            wsum = w_new["p_pos"] + w_new["s_pos"] + w_new["n_pos"] + w_new["not_p"] + w_new["not_s"]
            if float(wsum.sum()) <= 0:
                zero += 1
            if (not vis_p) and (not vis_s) and hp and hs and (not is_noise):
                n_as_n += 1  # v2 old rule would have been full-N; new rule is zero
            if is_noise:
                n_noise += 1
            gp = w_new["p_pos"] + w_new["s_pos"]
            gn = w_new["n_pos"]
            gmass_p += float(w_new["p_pos"].sum())
            gmass_s += float(w_new["s_pos"].sum())
            nmass += float(gn.sum())
            wsum_phase += float(gp.sum())
            wsum_non += float((wsum - gp).sum())
        out[f"epoch{ep}"] = {
            "kind_counts": kinds,
            "zero_loss_windows_under_NEW_rule": zero,
            "p_plus_s_background_windows": n_as_n,
            "noise_windows": n_noise,
            "frac_zero_loss_new": zero / max(len(cat), 1),
            "frac_ps_background_of_event": n_as_n / max(int((~cat.is_noise.astype(bool)).sum()), 1),
            "weight_mass_p": gmass_p,
            "weight_mass_s": gmass_s,
            "weight_mass_n": nmass,
            "phase_vs_nonphase_weight_ratio": (wsum_phase / max(wsum_non, 1e-12)),
            "note_v2_training": "v2 used homogeneous crops and treated P+S-out-of-window as full N; NEW rule zeros those windows",
        }
    return out


def plot20(rows, wave_by_name, outdir: Path):
    ensure_dir(outdir)
    vis = [r for r in rows if r["vis_s"]][:20]
    for i, r in enumerate(vis):
        fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        t = np.arange(IN_SAMPLES)
        enz = wave_by_name.get(r["trace"])
        if enz is not None:
            ax[0].plot(enz[0, : min(IN_SAMPLES, enz.shape[-1])], lw=0.5)
        ax[0].set_ylabel("Z (crop approx)")
        ax[1].plot(r["probs"][0], label="P", lw=0.8)
        ax[1].plot(r["probs"][1], label="S", lw=0.8)
        ax[1].plot(r["probs"][2], label="N", lw=0.8)
        ax[1].legend(fontsize=7)
        if r["p_c"] >= 0:
            ax[1].axvline(r["p_c"], color="C0", ls="--", lw=0.7)
        if r["true_s"] >= 0:
            ax[1].axvline(r["true_s"], color="C1", ls="--", lw=0.7)
        pk, _, _, _ = extract_picks(r["probs"][1], thr=0.2)
        for p in pk:
            ax[1].axvline(p, color="red", ls=":", lw=0.6)
        ax[2].plot(r["probs"][1])
        ax[2].set_ylabel("S prob")
        ax[3].plot(r["pad"])
        ax[3].set_ylabel("pad")
        fig.suptitle(f"{r['trace']} kind={r['kind']} visS={r['vis_s']}")
        fig.tight_layout()
        fig.savefig(outdir / f"dev_{i:02d}.png", dpi=100)
        plt.close(fig)


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    idle = idle_gpu_indices()
    if idle:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idle[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confirm_ids = set(load_full_event_ids("stage6_internal_confirm"))  # IDs only, no waveforms
    val = pd.read_parquet(RUN / "val_subset.parquet")
    assert set(val["event_id"].astype(str)) & confirm_ids == set()
    train_cat = pd.read_parquet(RUN / "train_online_catalog.parquet")
    assert set(train_cat.loc[~train_cat.is_noise.astype(bool), "event_id"].astype(str)) & confirm_ids == set()

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")

    def read_fn(name, is_noise=False):
        return wave.read(name, is_noise=is_noise)

    # localization crops: S-centered so S is in-window
    s_lab = train_cat[(~train_cat.is_noise.astype(bool)) & train_cat.get("has_s_label", True)]
    if "s_arrival_sample" in train_cat.columns:
        s_lab = train_cat[(~train_cat.is_noise.astype(bool)) & pd.to_numeric(train_cat.s_arrival_sample, errors="coerce").notna()]
    rng = np.random.default_rng(42)
    tr_diag = s_lab.sample(n=min(256, len(s_lab)), random_state=42).copy()
    tr_diag["crop_kind"] = "s_centered"
    va = val.copy()
    va["is_noise"] = False
    va["crop_kind"] = "s_centered"
    va = annotate_online_catalog(va) if "has_p_label" not in va.columns else va
    va["crop_kind"] = "s_centered"

    ds_tr = DKPNPartialCropDataset(tr_diag, read_fn, augment=False, seed=42)
    ds_va = DKPNPartialCropDataset(va, read_fn, augment=False, seed=42)
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)

    ckpt_files = {
        "init": RUN / "checkpoints/init.pt",
        "best_metric": RUN / "checkpoints/best_metric.pt",
        "best_loss": RUN / "checkpoints/best_loss.pt",
        "last": RUN / "checkpoints/last.pt",
    }
    diags = {}
    protocol = {
        "cf_5ch": True,
        "zne_order": True,
        "sr_100_len_3001": True,
        "psn_index_0P_1S_2N": True,
        "eval_softmax_once": True,
        "preproc_shared_compute_cf_5ch_normalize_cf_window": True,
        "crop_start_maps_via_label_coord": True,
        "padding_zeroed_in_peak_search": True,
        "extract_picks_height_0p2_distance_50_smooth_box3": True,
        "f1_0p5_is_time_tolerance_50_samples_not_prob_thr": True,
        "prob_threshold_swept_separately": THRS,
        "confirm_waveforms_read": False,
        "confirm_ids_used_only_for_disjoint_assert": True,
    }

    last_va_rows = None
    for name, path in ckpt_files.items():
        model, meta = load_model(path, device)
        tr = infer_split(model, ld_tr, device)
        va_m = infer_split(model, ld_va, device)
        last_va_rows = va_m["_rows"]
        tr.pop("_rows")
        va_m.pop("_rows")
        # collapse / generalization / pick / offset
        verdict = []
        if tr["oracle_acc@0.5s"] < 0.05 and va_m["oracle_acc@0.5s"] < 0.05:
            verdict.append("train_and_dev_near_zero_collapse")
        elif tr["oracle_acc@0.5s"] >= 0.2 and va_m["oracle_acc@0.5s"] < 0.05:
            verdict.append("train_ok_dev_fail_generalization")
        if va_m["oracle_acc@0.5s"] >= 0.2 and va_m["threshold_sweep"]["0.2"]["f1@0.5s"] < 0.05:
            verdict.append("oracle_ok_extract_picks_fail")
        off = va_m["median_signed_argmax_offset_samples"]
        if off is not None and np.isfinite(off) and abs(off) >= 20 and va_m["oracle_acc@0.5s"] < 0.2:
            verdict.append("possible_fixed_time_offset")
        diags[name] = {
            "ckpt_epoch": meta.get("epoch"),
            "train": tr,
            "dev": va_m,
            "verdict": verdict,
            "invalid_best_metric": name == "best_metric",
        }
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # 20 plots from last ckpt rows (reload last for probs already in last_va_rows)
    wave_preview = {}
    vis_rows = [r for r in (last_va_rows or []) if r["vis_s"]][:20]
    for r in vis_rows:
        try:
            enz = np.asarray(wave.read(r["trace"], is_noise=False), dtype=np.float32)
            wave_preview[r["trace"]] = enz
        except Exception:
            pass
    plot20(last_va_rows or [], wave_preview, PLOT)

    # grads on a tiny batch from last.pt
    model, _ = load_model(RUN / "checkpoints/last.pt", device)
    model.train()
    batch = next(iter(ld_tr))
    x = batch["x"].to(device).requires_grad_(True)
    kw = {k: batch[k].to(device) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    logits = dkpn_logits(model, x)
    loss = partial_label_nll(logits, **kw)
    g = torch.autograd.grad(loss, logits, retain_graph=False)[0]
    s_idx = batch["s_c"].long().clamp(0, IN_SAMPLES - 1)
    vis = batch["vis_s"] > 0.5
    g_s_true = []
    g_n_reg = []
    for i in range(len(s_idx)):
        if not bool(vis[i]):
            continue
        t = int(s_idx[i])
        g_s_true.append(float(g[i, :, t].abs().mean()))
        # N-ish region: far from s
        sl = list(range(0, max(0, t - 200))) + list(range(min(IN_SAMPLES, t + 200), IN_SAMPLES))
        if sl:
            g_n_reg.append(float(g[i, :, sl].abs().mean()))
    grad_rep = {
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(loss)),
        "mean_abs_grad_at_true_S": float(np.mean(g_s_true) if g_s_true else 0.0),
        "mean_abs_grad_N_region": float(np.mean(g_n_reg) if g_n_reg else 0.0),
        "S_over_N_grad_ratio": float((np.mean(g_s_true) if g_s_true else 0) / max(np.mean(g_n_reg) if g_n_reg else 1e-12, 1e-12)),
    }

    # epoch2 vs epoch1 capability: best_loss is epoch1, last is epoch2
    cover = {
        "best_loss_is_epoch": 1,
        "last_is_epoch": 2,
        "dev_oracle_epoch1_best_loss": diags["best_loss"]["dev"]["oracle_acc@0.5s"],
        "dev_oracle_epoch2_last": diags["last"]["dev"]["oracle_acc@0.5s"],
        "epoch2_covered_epoch1": diags["last"]["dev"]["oracle_acc@0.5s"] + 1e-9 < diags["best_loss"]["dev"]["oracle_acc@0.5s"],
        "note": "per-epoch ckpts missing; epoch1≈best_loss.pt, epoch2=last.pt, epoch0≈best_metric.pt (F1=0 invalid best)",
    }

    sup = supervision_report(train_cat.sample(n=min(8000, len(train_cat)), random_state=0))

    sweep_out = {k: {"train": diags[k]["train"]["threshold_sweep"], "dev": diags[k]["dev"]["threshold_sweep"]} for k in diags}
    save_json(sweep_out, ART / "dkpn_v2_threshold_sweep.json")
    payload = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "frozen_as": "dkpn_clean_v2_seed42_failed_homogeneous_crop_cycle",
        "device": str(device),
        "confirm_waveforms_read": False,
        "confirm_id_disjoint_ok": True,
        "missing_ckpts": ["epoch0.pt", "epoch1.pt", "epoch2.pt"],
        "best_metric_invalid": True,
        "best_metric_epoch": 0,
        "best_metric_val_s_f1": 0.0,
        "protocol": protocol,
        "diagnostics": {k: {kk: vv for kk, vv in d.items()} for k, d in diags.items()},
        "epoch2_vs_epoch1": cover,
        "supervision_sample8000": sup,
        "grad_last_on_train_s_centered_batch": grad_rep,
        "plots_dir": str(PLOT),
        "n_plots": 20,
    }
    save_json(payload, ART / "dkpn_v2_checkpoint_diagnostics.json")
    print(json.dumps({k: payload["diagnostics"][k]["verdict"] for k in payload["diagnostics"]}, indent=2))
    print("wrote", ART / "dkpn_v2_checkpoint_diagnostics.json")


if __name__ == "__main__":
    main()
