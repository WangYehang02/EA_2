#!/usr/bin/env python
"""Cache STEAD / ID-A top-10 S candidates for Phase B (resume-safe shards, multi-GPU)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.phaseB import CAND_EXTRACT_CFG, candidates_from_s_proba, sha256_file
from earthquake.utils import ensure_dir


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_model(source: str, device: str, ida_ckpt: Path | None):
    ref = SeisBenchPhaseNetReference(weight="stead", device=device)
    meta = {"source": source, "weight": "stead"}
    if source == "ida":
        assert ida_ckpt is not None and ida_ckpt.name == "best.pt"
        blob = torch.load(ida_ckpt, map_location="cpu", weights_only=False)
        ref.model.load_state_dict(blob["model"], strict=True)
        set_train_bn_eval(ref.model)
        ref.model.to(device)
        ref.model.eval()
        meta["ida_ckpt"] = str(ida_ckpt)
        meta["ida_epoch"] = int(blob.get("epoch", -1))
        meta["ida_sha256"] = sha256_file(ida_ckpt)
    else:
        ref.model.eval()
    return ref, meta


def run_shard(args) -> None:
    # Confirm waveform caching only when AUTHORIZED/RUNNING; otherwise refuse if method_lock exists.
    from earthquake.stage6.confirm_state import final_confirm_dir

    fc = final_confirm_dir()
    allow_confirm = (fc / "CONFIRM_EVAL.AUTHORIZED").exists() or (fc / "CONFIRM_EVAL.RUNNING").exists()
    out_abs = str((ROOT / args.out_dir).resolve())
    under_final = out_abs.startswith(str(fc.resolve()))
    try:
        assert_full_confirm_access_allowed(purpose="phaseB_or_confirm_cache")
        if not (allow_confirm and under_final):
            raise SystemExit("method_lock present — refuse confirm cache without AUTHORIZED+final_confirm out-dir")
    except RuntimeError:
        pass

    manifest = pd.read_csv(ROOT / args.manifest)
    n = len(manifest)
    idxs = list(range(args.rank, n, args.world_size))
    out_root = ensure_dir(ROOT / args.out_dir)
    shard_path = out_root / f"shard_{args.rank:02d}.parquet"
    done_path = out_root / f"shard_{args.rank:02d}.DONE"
    progress_path = out_root / f"shard_{args.rank:02d}.progress.json"

    done_names: set[str] = set()
    rows: list[dict] = []
    if shard_path.exists():
        prev = pd.read_parquet(shard_path)
        done_names = set(prev["trace_name"].astype(str).unique())
        rows = prev.to_dict("records")

    meta_json = json.loads((ROOT / args.manifest_json).read_text()) if args.manifest_json else {}
    device = args.device
    ida_ckpt = Path(args.ida_ckpt) if args.ida_ckpt else None
    if args.source == "ida" and (ida_ckpt is None or ida_ckpt.name != "best.pt"):
        raise SystemExit("ID-A cache requires --ida-ckpt .../best.pt")

    ref, model_meta = _load_model(args.source, device, ida_ckpt)
    events_h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    noise_h5 = resolve_instance_root() / "noise" / "Instance_noise.hdf5"
    is_noise = bool(args.noise)
    reader = InstanceHDF5Reader(noise_h5 if is_noise else events_h5).open()

    cfg_path = out_root / "candidate_extract_cfg.json"
    if args.rank == 0:
        save_json(
            {
                **CAND_EXTRACT_CFG,
                "model_meta": model_meta,
                "manifest_sha256": meta_json.get("csv_sha256"),
                "source": args.source,
            },
            cfg_path,
        )

    t0 = time.time()
    processed = 0
    fails = 0
    for j, i in enumerate(idxs):
        row = manifest.iloc[i]
        tn = str(row["trace_name"])
        if tn in done_names:
            continue
        try:
            wave = reader.read_waveform(tn)
            with torch.no_grad():
                pred = ref.predict_row(wave, row, remap_to_waveform=True)
            s_proba = np.asarray(pred["s_proba_on_waveform"], dtype=np.float64)
            p_proba = np.asarray(pred["p_proba_on_waveform"], dtype=np.float64)
            if (not np.isfinite(s_proba).all()) or (not np.isfinite(p_proba).all()):
                raise RuntimeError("nonfinite proba")
            sr = float(row.get("sampling_rate_hz", 100.0))
            cands = candidates_from_s_proba(
                s_proba,
                sampling_rate=sr,
                waveform_starttime=row["trace_start_time"],
                k=10,
                source=args.source,
            )
            top1_s = float(pred["s_pred_sample_on_waveform"])
            top1_p = float(pred["p_pred_sample_on_waveform"])
            top1_s_prob = float(pred["s_peak_probability"])
            top1_p_prob = float(pred["p_peak_probability"])
            for c in cands:
                rows.append(
                    {
                        "trace_name": tn,
                        "event_id": str(row.get("event_id", "NOISE")),
                        "true_s_sample": float(row["s_arrival_sample"])
                        if (not is_noise and pd.notna(row.get("s_arrival_sample")))
                        else np.nan,
                        "true_p_sample": float(row["p_arrival_sample"])
                        if (not is_noise and pd.notna(row.get("p_arrival_sample")))
                        else np.nan,
                        "sampling_rate_hz": sr,
                        "top1_s_sample": top1_s,
                        "top1_p_sample": top1_p,
                        "top1_s_prob": top1_s_prob,
                        "top1_p_prob": top1_p_prob,
                        **c,
                    }
                )
            done_names.add(tn)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            fails += 1
            if fails <= 5:
                print({"fail": tn, "err": repr(exc)}, flush=True)
            # still mark with empty cand? skip — analysis will see miss
            done_names.add(tn)
            processed += 1

        if processed % 50 == 0:
            _atomic_parquet(pd.DataFrame(rows), shard_path)
            elapsed = time.time() - t0
            remaining = len(idxs) - len([x for x in idxs if manifest.iloc[x].trace_name in done_names or True])
            # simpler ETA
            n_done = sum(1 for x in idxs if str(manifest.iloc[x].trace_name) in done_names)
            rate = n_done / max(elapsed, 1e-6)
            eta = (len(idxs) - n_done) / max(rate, 1e-9)
            save_json(
                {
                    "rank": args.rank,
                    "source": args.source,
                    "n_done": n_done,
                    "n_total_shard": len(idxs),
                    "fails": fails,
                    "rate_trace_per_s": rate,
                    "eta_s": eta,
                    "pid": os.getpid(),
                },
                progress_path,
            )
            print(
                {
                    "rank": args.rank,
                    "source": args.source,
                    "n_done": n_done,
                    "n_shard": len(idxs),
                    "eta_min": eta / 60,
                },
                flush=True,
            )

    reader.close()
    _atomic_parquet(pd.DataFrame(rows), shard_path)
    done_path.write_text("ok\n")
    print({"rank": args.rank, "source": args.source, "DONE": True, "n_rows": len(rows), "fails": fails}, flush=True)


def merge_shards(out_dir: Path, world: int, final_name: str) -> Path:
    frames = []
    for r in range(world):
        p = out_dir / f"shard_{r:02d}.parquet"
        d = out_dir / f"shard_{r:02d}.DONE"
        if not d.exists() or not p.exists():
            raise RuntimeError(f"missing shard {r}: {p} / {d}")
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    # dedupe traces if resume duplicated
    df = df.sort_values(["trace_name", "candidate_rank"]).drop_duplicates(
        ["trace_name", "candidate_rank", "candidate_source"], keep="last"
    )
    final = out_dir / final_name
    tmp = final.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    (out_dir / "MERGE.DONE").write_text(f"{len(df)}\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["stead", "ida"], required=True)
    parser.add_argument("--manifest", default="artifacts/results/stage6/phaseB_eval_manifest.csv")
    parser.add_argument("--manifest-json", default="artifacts/results/stage6/phaseB_eval_manifest.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ida-ckpt", default="artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        base = "artifacts/cache/stage6/phaseB"
        name = f"{'noise_' if args.noise else ''}{args.source}_top10"
        args.out_dir = f"{base}/{name}"

    out_dir = ensure_dir(ROOT / args.out_dir)
    if args.merge_only:
        final = merge_shards(out_dir, args.world_size, f"{args.source}_top10.parquet")
        print({"merged": str(final), "sha256": sha256_file(final)})
        return

    run_shard(args)


if __name__ == "__main__":
    main()
