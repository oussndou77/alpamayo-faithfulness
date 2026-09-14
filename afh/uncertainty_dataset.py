# SPDX-License-Identifier: Apache-2.0
"""
afh.uncertainty_dataset — on-the-fly training data for uncertainty fine-tuning.

Nothing heavy is ever stored. The "dataset" is a JSONL manifest where each line is a
few hundred bytes: (clip_id, t0_us, degradation spec, target text, target trajectory).
Frames are re-loaded and re-degraded at training time — deterministic by seed, so every
example is exactly reproducible.

Two stages:

  Stage A (pod, GPU, once per clip) — capture the model's CLEAN reasoning and the true
  future trajectory. `runners/run_inference_a2.py` already writes this into records.json
  (traces[0].raw_text + we add ego_future_xyz via --diag or the clean cache below).

  Stage B (cold) — build_manifest(): for each clip x N sampled specs, derive the target
  text (afh.degradation.target_text) and target trajectory (target_trajectory) and write
  one JSONL line. Clean examples (s=0) use the model's own clean reasoning as target, so
  the fine-tune preserves original behavior where the input is fine.

Training — UncertaintyDataset(manifest, frame_loader): __getitem__ re-loads the clip's
frames through `frame_loader(clip_id, t0_us) -> (frames uint8 (n_cam,n_t,3,H,W), meta)`
and applies the degradation. The loader is injected so the class is testable cold and
independent of alpamayo2_super.

CLI (Stage B):
    python -m afh.uncertainty_dataset build \
        --clean-cache fixtures/clean_cache_a2.json \
        --n-per-clip 16 --out outputs/uncertainty_manifest.jsonl
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Callable, Iterable

import numpy as np

from afh.degradation import (
    DegradationSpec, apply_degradation, sample_spec, target_text, target_trajectory,
    CLEAN_FRACTION, FAMILIES,
)


# --------------------------------------------------------------------------- Stage A cache

def clean_cache_from_records(records_path: str, diag_path: str | None = None,
                             t0_us: int = 5_100_000) -> dict:
    """
    Build the clean cache from a records.json (+ optional raw_diag.json with GT waypoints).

    Output: {clip_id: {"t0_us": int, "clean_reasoning": str, "future_xy": [[x,y],...]}}
    clean_reasoning = first rollout's raw text (the model's own words on clean input).
    future_xy comes from diag['raw_points_xy_first_rollout'] if no GT is available —
    prefer real ego_future_xyz when the runner exposes it (see runners/cache_clean_a2.py).
    """
    with open(records_path) as fh:
        records = json.load(fh)
    diag = {}
    if diag_path:
        with open(diag_path) as fh:
            diag = {d["clip_id"]: d for d in json.load(fh)}
    cache = {}
    for rec in records:
        cid = rec["clip_id"]
        traces = rec.get("traces", [])
        clean = traces[0]["raw_text"].split("\n")[0] if traces else ""
        xy = diag.get(cid, {}).get("raw_points_xy_first_rollout")
        cache[cid] = {"t0_us": t0_us, "clean_reasoning": clean,
                      "future_xy": xy, "source": "records"}
    return cache


# --------------------------------------------------------------------------- Stage B manifest

def build_manifest(clean_cache: dict, n_per_clip: int = 16, seed: int = 0,
                   families: Iterable[str] = FAMILIES,
                   clean_fraction: float = CLEAN_FRACTION,
                   n_cam: int = 7) -> list[dict]:
    """
    For each clip, sample n_per_clip specs and derive targets. Returns a list of dicts
    (one per example). Camera choice inside a spec is resolved at load time by
    apply_degradation (needs n_cam); we pre-resolve it here with a lightweight dry run on
    a tiny dummy tensor so the target TEXT (which names cameras) is fixed in the manifest.
    """
    rng = random.Random(seed)
    dummy = np.zeros((n_cam, 4, 3, 8, 8), dtype=np.uint8)
    out = []
    for cid, info in clean_cache.items():
        xy_true = np.asarray(info.get("future_xy") or [], dtype=float)
        for k in range(n_per_clip):
            spec = sample_spec(rng.randrange(1 << 30), families=families,
                               clean_fraction=clean_fraction)
            # resolve cameras/params deterministically (same seed => same choice on real frames)
            _, spec = apply_degradation(dummy, spec)
            text = target_text(spec, clean_reasoning=info.get("clean_reasoning"))
            traj = (target_trajectory(xy_true, spec.severity).round(3).tolist()
                    if xy_true.size else None)
            out.append({
                "clip_id": cid, "t0_us": info.get("t0_us", 5_100_000),
                "spec": asdict(spec),
                "target_text": text,
                "target_xy": traj,
                "severity": spec.severity, "family": spec.family,
            })
    return out


def write_manifest(entries: list[dict], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def read_manifest(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def manifest_summary(entries: list[dict]) -> str:
    n = len(entries)
    fam = {}
    for e in entries:
        fam[e["family"]] = fam.get(e["family"], 0) + 1
    clips = len({e["clip_id"] for e in entries})
    sev = [e["severity"] for e in entries if e["severity"] > 0]
    lines = [f"{n} examples over {clips} clips",
             "  " + ", ".join(f"{k}: {v}" for k, v in sorted(fam.items())),
             f"  clean fraction: {fam.get('clean', 0) / max(n, 1):.0%}"]
    if sev:
        lines.append(f"  severity (degraded only): mean {np.mean(sev):.2f}, "
                     f"min {min(sev):.2f}, max {max(sev):.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- training-time Dataset

class UncertaintyDataset:
    """
    Map-style dataset. frame_loader(clip_id, t0_us) -> (frames uint8 (n_cam,n_t,3,H,W), meta)
    is injected: on the pod wrap alpamayo2_super.load_physical_aiavdataset; in tests use a
    synthetic loader. A tiny LRU keeps the last few clips' clean frames in memory so the
    n_per_clip examples of one clip don't re-decode video n_per_clip times.
    Works with torch.utils.data.DataLoader (implements __len__/__getitem__).
    """

    def __init__(self, entries: list[dict], frame_loader: Callable, cache_clips: int = 4):
        self.entries = entries
        self.loader = frame_loader
        self._cache: dict[tuple, tuple] = {}
        self._cache_clips = cache_clips

    def __len__(self):
        return len(self.entries)

    def _frames(self, clip_id: str, t0_us: int):
        key = (clip_id, t0_us)
        if key not in self._cache:
            if len(self._cache) >= self._cache_clips:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self.loader(clip_id, t0_us)
        return self._cache[key]

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        frames, meta = self._frames(e["clip_id"], e["t0_us"])
        spec = DegradationSpec(**e["spec"])
        degraded, spec = apply_degradation(np.asarray(frames), spec)
        return {
            "frames": degraded,                       # uint8 (n_cam, n_t, 3, H, W)
            "target_text": e["target_text"],
            "target_xy": (np.asarray(e["target_xy"], dtype=np.float32)
                          if e["target_xy"] is not None else None),
            "severity": float(e["severity"]),
            "family": e["family"],
            "spec": asdict(spec),
            "clip_id": e["clip_id"],
            "meta": meta,
        }


# --------------------------------------------------------------------------- CLI

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the uncertainty fine-tuning manifest (cold)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clean-cache", help="clean cache JSON (Stage A output)")
    b.add_argument("--records", help="alternatively: records.json from run_inference_a2")
    b.add_argument("--diag", help="raw_diag.json (waypoints) matching --records")
    b.add_argument("--n-per-clip", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--n-cam", type=int, default=7)
    b.add_argument("--out", default="outputs/uncertainty_manifest.jsonl")
    s = sub.add_parser("summary")
    s.add_argument("manifest")
    a = ap.parse_args()

    if a.cmd == "build":
        if a.clean_cache:
            cache = json.load(open(a.clean_cache))
        elif a.records:
            cache = clean_cache_from_records(a.records, a.diag)
        else:
            ap.error("provide --clean-cache or --records")
        entries = build_manifest(cache, n_per_clip=a.n_per_clip, seed=a.seed, n_cam=a.n_cam)
        write_manifest(entries, a.out)
        print(manifest_summary(entries))
        print(f"-> {a.out}")
    else:
        print(manifest_summary(read_manifest(a.manifest)))


if __name__ == "__main__":
    _main()
