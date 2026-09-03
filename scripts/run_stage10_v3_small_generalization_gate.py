#!/usr/bin/env python
"""Small event-disjoint mixed-crop generalization gate. Does not start v3 full pilot."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage10.crop_v2 import CROP_CYCLE, scheduled_crop_kind
from earthquake.stage10.dataset_v2 import DKPNPartialCropDataset, annotate_online_catalog
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.dkpn_picks import extract_picks, s_f1_at_tolerance
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage10.partial_label import partial_label_nll
from earthquake.utils import ensure_dir

N_STEPS = 2000
BATCH = 16
N_TRAIN = 10000
N_DEV = 2000


def collate(batch):
    keys_t = ["x", "p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask", "vis_p", "vis_s", "p_c", "s_c"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_t}
    out["crop_kind"] = [b["crop_kind"] for b in batch]
    return out


def build_sets(seed=42):
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    picker = set(load_full_trace_names("stage6_picker_train"))
    dev_tr = set(load_full_trace_names("stage6_dev"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    ev = events[events["trace_name"].astype(str).isin(picker)].copy()
    dv = events[events["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK train∩confirm")
    if set(dv["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK dev∩confirm")
    if set(ev["event_id"].astype(str)) & set(dv["event_id"].astype(str)):
        raise SystemExit("LEAK train∩dev events")
    p = pd.to_numeric(ev["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ev["is_noise"] = False
    dv["is_noise"] = False
    ps = ev[p.notna() & s.notna()]
    po = ev[p.notna() & s.isna()]
    dps = (ps.s_arrival_sample.astype(float) - ps.p_arrival_sample.astype(float)) / 100.0
    short = ps[dps <= 30]
    longp = ps[dps > 30]
    rng = np.random.default_rng(seed)
    parts = []
    # mix short/long/p-only
    for df, n in [(short, 5000), (longp, min(500, len(longp))), (po, 2500)]:
        if len(df) == 0:
            continue
        parts.append(df.sample(n=min(n, len(df)), random_state=seed))
    tr = pd.concat(parts, ignore_index=True)
    if len(tr) < N_TRAIN:
        extra = ev[~ev.trace_name.isin(tr.trace_name)].sample(n=min(N_TRAIN - len(tr), len(ev)), random_state=seed)
        tr = pd.concat([tr, extra], ignore_index=True)
    tr = tr.drop_duplicates("trace_name").head(N_TRAIN)
    n_noise = int(round(len(tr) * 0.2))
    nsel = noise.sample(n=min(n_noise, len(noise)), random_state=seed).copy()
    nsel["is_noise"] = True
    nsel["event_id"] = "NOISE"
    for c in tr.columns:
        if c not in nsel.columns:
            nsel[c] = np.nan
    tr = pd.concat([tr, nsel[tr.columns]], ignore_index=True)
    sdev = pd.to_numeric(dv["s_arrival_sample"], errors="coerce")
    dv_s = dv[sdev.notna()]
    # include some p-only + we don't add noise to eval loc
    de = dv_s.sample(n=min(N_DEV, len(dv_s)), random_state=seed)
    assert set(tr.loc[~tr.is_noise.astype(bool), "event_id"].astype(str)) & confirm == set()
    assert set(de["event_id"].astype(str)) & confirm == set()
    return annotate_online_catalog(tr), annotate_online_catalog(de)


@torch.no_grad()
def s_metrics(model, ds, device, crop="s_centered"):
    cat = ds.cat.copy()
    cat["crop_kind"] = crop
    ds2 = DKPNPartialCropDataset(cat, ds.read_fn, augment=False, seed=0)
    ld = DataLoader(ds2, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    preds_thr, preds_or, trues, vis = [], [], [], []
    smax = []
    kinds = []
    for batch in ld:
        pr = torch.softmax(dkpn_logits(model, batch["x"].to(device)).float(), dim=1).cpu().numpy()
        for i in range(pr.shape[0]):
            s = pr[i, 1]
            kinds.append(batch["crop_kind"][i])
            t = float(batch["s_c"][i])
            v = bool(batch["vis_s"][i] > 0.5 and t >= 0)
            vis.append(v)
            trues.append(t)
            smax.append(float(s.max()))
            preds_or.append(int(np.argmax(s)))
            pk, _, amp, _ = extract_picks(s, thr=0.2)
            preds_thr.append(np.nan if len(pk) == 0 else float(pk[int(np.argmax(amp))]))
    v = np.asarray(vis)
    f1 = s_f1_at_tolerance(np.asarray(preds_thr), np.asarray(trues), v, tol_samples=50)["f1"]
    ora = s_f1_at_tolerance(np.asarray(preds_or, dtype=float), np.asarray(trues), v, tol_samples=50)["f1"]
    # oracle acc
    acc = float(np.mean([abs(o - t) <= 50 for o, t, vv in zip(preds_or, trues, vis) if vv] or [0]))
    return {
        "f1@0.5_thr0.2": f1,
        "f1@0.5_oracle_argmax": ora,
        "oracle_acc@0.5s": acc,
        "s_max_mean": float(np.mean(smax)),
        "n": len(trues),
        "n_vis": int(sum(vis)),
    }


def main() -> None:
    idle = idle_gpu_indices()
    # prefer a free card; leave at least one unused overall by using only 1
    if not idle:
        raise SystemExit("no idle GPU for small gate")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda required")
    tr, de = build_sets(42)
    # mix-crop presence in epoch 0
    kinds0 = [
        scheduled_crop_kind(virtual_epoch=0, trace_name=str(n), is_noise=False, has_p_label=True, has_s_label=True)
        for n in tr.loc[(tr.has_p_label.astype(bool)) & (tr.has_s_label.astype(bool)), "trace_name"].head(500)
    ]
    mix_ok = set(kinds0) == set(CROP_CYCLE)

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")

    def read_fn(name, is_noise=False):
        return wave.read(name, is_noise=is_noise)

    ds = DKPNPartialCropDataset(tr, read_fn, augment=True, seed=42)
    ds.set_virtual_epoch(0)
    ld = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    ds_eval_tr = DKPNPartialCropDataset(tr.sample(n=min(400, len(tr)), random_state=1), read_fn, augment=False, seed=0)
    ds_eval_de = DKPNPartialCropDataset(de.sample(n=min(400, len(de)), random_state=1), read_fn, augment=False, seed=0)
    model = build_dkpn_random().to(device)
    init_tr = s_metrics(model, ds_eval_tr, device)
    init_de = s_metrics(model, ds_eval_de, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = 0
    hist = []
    kinds_seen = set()
    t0 = time.time()
    nan = False
    while step < N_STEPS:
        for batch in ld:
            opt.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            kw = {k: batch[k].to(device) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
            logits = dkpn_logits(model, x)
            loss = partial_label_nll(logits, **kw)
            if not torch.isfinite(loss):
                nan = True
                break
            loss.backward()
            # grad at visible S
            opt.step()
            for k in batch["crop_kind"]:
                kinds_seen.add(k)
            step += 1
            if step % 200 == 0:
                hist.append({"step": step, "loss": float(loss.detach()), "kinds": dict.fromkeys(kinds_seen, True)})
                print(hist[-1], flush=True)
            if step >= N_STEPS:
                break
        if nan:
            break
        ds.set_virtual_epoch(step // max(len(tr) // BATCH, 1))
    model.eval()
    # non-zero S grad check
    batch = next(iter(DataLoader(ds_eval_tr, batch_size=8, collate_fn=collate)))
    model.zero_grad(set_to_none=True)
    x = batch["x"].to(device)
    kw = {k: batch[k].to(device) for k in ["p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask"]}
    logits = dkpn_logits(model, x)
    loss = partial_label_nll(logits, **kw)
    glog = torch.autograd.grad(loss, logits, retain_graph=False)[0]
    vis = batch["vis_s"] > 0.5
    g_s = []
    for i in range(len(vis)):
        if not bool(vis[i]):
            continue
        t = int(batch["s_c"][i].clamp(0, 3000))
        g_s.append(float(glog[i, 1, t].abs()))
    fin_tr = s_metrics(model, ds_eval_tr, device)
    fin_de = s_metrics(model, ds_eval_de, device)
    collapse = fin_tr["s_max_mean"] < 1e-6
    ok = (
        (not nan)
        and (not collapse)
        and fin_tr["f1@0.5_thr0.2"] > init_tr["f1@0.5_thr0.2"] + 0.02
        and fin_de["f1@0.5_thr0.2"] > init_de["f1@0.5_thr0.2"] + 0.02
        and mix_ok
        and {"p_centered", "s_centered"} <= kinds_seen
        and (np.mean(g_s) if g_s else 0) > 0
    )
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "steps": step,
        "sec": time.time() - t0,
        "device": str(device),
        "n_train": int(len(tr)),
        "n_dev": int(len(de)),
        "mixed_crop_epoch0_has_all_kinds": mix_ok,
        "kinds_seen_in_training": sorted(kinds_seen),
        "init_train": init_tr,
        "init_dev": init_de,
        "final_train": fin_tr,
        "final_dev": fin_de,
        "nan": nan,
        "collapse": collapse,
        "mean_abs_grad_S_true": float(np.mean(g_s) if g_s else 0.0),
        "ok": bool(ok),
        "v3_allowed": bool(ok),
        "confirm_read": False,
        "no_v2_overwrite": True,
        "history": hist,
    }
    save_json(report, ROOT / "artifacts/results/stage10/dkpn_v3_small_generalization_gate.json")
    print(json.dumps({k: report[k] for k in ["ok", "v3_allowed", "final_train", "final_dev", "init_train", "init_dev"]}, indent=2, default=str))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
