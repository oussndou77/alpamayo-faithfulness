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

Held-out evaluation — build_split(): train/test are separated by clip_id (a clip is
never on both sides), and whole families and/or family COMBINATIONS can be reserved for
test only. The train manifest then never contains a held-out family (alone or inside a
composite) nor a held-out combination; the test manifest contains every family, single
and composite, plus explicit examples of each held-out item. check_split() asserts all of
this and is run by build_split() itself.

CLI (Stage B):
    python -m afh.uncertainty_dataset build \
        --clean-cache fixtures/clean_cache_a2.json \
        --n-per-clip 16 --out outputs/uncertainty_manifest.jsonl

    python -m afh.uncertainty_dataset split \
        --records fixtures/records_a2.json --diag fixtures/raw_diag_a2.json \
        --test-fraction 0.3 --holdout-family desync --holdout-combo glare+blur \
        --composite-fraction 0.3 --out-dir outputs/uncertainty_split
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from typing import Callable, Iterable

import numpy as np

from afh.degradation import (
    DegradationSpec, apply_degradation, sample_spec, sample_composite_spec, compose,
    target_text, target_trajectory, combo_key, spec_families, normalize_combo,
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
                   n_cam: int = 7,
                   composite_fraction: float = 0.0,
                   n_components: int = 2,
                   exclude_combos: Iterable = (),
                   extra_specs: Iterable[DegradationSpec] | None = None) -> list[dict]:
    """
    For each clip, sample n_per_clip specs and derive targets. Returns a list of dicts
    (one per example). Camera choice inside a spec is resolved at load time by
    apply_degradation (needs n_cam); we pre-resolve it here with a lightweight dry run on
    a tiny dummy tensor so the target TEXT (which names cameras) is fixed in the manifest.

    composite_fraction: share of DEGRADED examples that are composites of `n_components`
    distinct families (never a combo in `exclude_combos`). Default 0 reproduces the
    single-family manifests of earlier versions exactly (same seed -> same lines).
    extra_specs: templates appended for EVERY clip (used to guarantee held-out coverage in
    a test manifest); each gets a clip-specific seed so clips differ.
    """
    rng = random.Random(seed)
    dummy = np.zeros((n_cam, 4, 3, 8, 8), dtype=np.uint8)
    families = tuple(families)
    exclude_combos = tuple(exclude_combos)
    out = []
    for cid, info in clean_cache.items():
        xy_true = np.asarray(info.get("future_xy") or [], dtype=float)
        specs = []
        for k in range(n_per_clip):
            spec_seed = rng.randrange(1 << 30)
            spec = sample_spec(spec_seed, families=families, clean_fraction=clean_fraction)
            # separate stream so composite_fraction=0 leaves the legacy sequence untouched
            if (composite_fraction > 0 and spec.family != "clean"
                    and random.Random(spec_seed ^ 0x5EED).random() < composite_fraction):
                spec = sample_composite_spec(spec_seed, families=families,
                                             n_components=n_components,
                                             exclude_combos=exclude_combos)
            specs.append(spec)
        for j, tpl in enumerate(extra_specs or ()):
            d = tpl.to_dict()
            d["seed"] = _stable_int(f"{seed}:{cid}:extra:{j}")
            if d.get("components"):
                specs.append(compose(*[{kk: vv for kk, vv in c.items() if kk != "seed"}
                                       for c in d["components"]], seed=d["seed"]))
            else:
                specs.append(DegradationSpec(**d))
        for spec in specs:
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
                "combo": combo_key(spec),
            })
    return out


# --------------------------------------------------------------------------- held-out split

def _stable_int(key: str) -> int:
    """Process-independent 30-bit integer from a string (Python's hash() is salted)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") >> 2


def split_clips(clip_ids: Iterable[str], test_fraction: float = 0.2,
                seed: int = 0) -> tuple[list[str], list[str]]:
    """
    Deterministic clip-level split. Clips are ranked by sha256(seed:clip_id) and the first
    round(n * test_fraction) go to test (at least one when test_fraction > 0 and n >= 2).
    Independent of input order; disjoint by construction.
    """
    ids = sorted(set(clip_ids))
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    n_test = int(round(len(ids) * test_fraction))
    if test_fraction > 0 and len(ids) >= 2:
        n_test = min(max(n_test, 1), len(ids) - 1)
    ranked = sorted(ids, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    test = set(ranked[:n_test])
    return [c for c in ids if c not in test], [c for c in ids if c in test]


def _holdout_tags(entry: dict, holdout_families: set, holdout_combos: set) -> dict:
    fams = spec_families(entry["spec"])
    return {"heldout_family": bool(fams & holdout_families),
            "heldout_combo": combo_key(entry["spec"]) in holdout_combos}


def build_split(clean_cache: dict, test_fraction: float = 0.2, seed: int = 0,
                holdout_families: Iterable[str] = (), holdout_combos: Iterable = (),
                n_per_clip: int = 16, n_test_per_clip: int | None = None,
                n_holdout_per_clip: int = 2,
                composite_fraction: float = 0.0, n_components: int = 2,
                clean_fraction: float = CLEAN_FRACTION, n_cam: int = 7,
                families: Iterable[str] = FAMILIES) -> dict:
    """
    Held-out train/test manifests.

      train: train clips only; families minus holdout_families; composites never form a
             holdout combo nor contain a held-out family.
      test:  test clips only; all families (single + composite, incl. held-out combos),
             plus n_holdout_per_clip explicit examples per held-out family and per held-out
             combo so every held-out item is measured on every test clip. Clean examples
             stay in test (clean_fraction) for the non-regression report.

    Every entry gets split="train"/"test" and heldout_family / heldout_combo tags, so the
    evaluator can report in-distribution and held-out slices separately.
    Returns {"train": [...], "test": [...], "meta": {...}}.
    """
    families = tuple(families)
    h_fam = set(holdout_families)
    h_combo = {normalize_combo(c) for c in holdout_combos}
    unknown = h_fam - set(families)
    for c in h_combo:
        unknown |= set(c.split("+")) - set(families)
    if unknown:
        raise ValueError(f"unknown held-out families: {sorted(unknown)}")
    train_fams = tuple(f for f in families if f not in h_fam)
    if not train_fams:
        raise ValueError("all families are held out; nothing left to train on")
    # a combo already excluded via a held-out family needs no separate exclusion, but
    # listing it is harmless; composites need n_components distinct train families
    train_comp = composite_fraction if len(train_fams) >= n_components else 0.0

    train_ids, test_ids = split_clips(clean_cache.keys(), test_fraction, seed)
    train_cache = {c: clean_cache[c] for c in train_ids}
    test_cache = {c: clean_cache[c] for c in test_ids}

    train = build_manifest(train_cache, n_per_clip=n_per_clip, seed=seed,
                           families=train_fams, clean_fraction=clean_fraction, n_cam=n_cam,
                           composite_fraction=train_comp, n_components=n_components,
                           exclude_combos=h_combo)
    extra = []
    for f in sorted(h_fam):
        extra += [DegradationSpec(family=f, severity=sev)
                  for sev in _holdout_severities(n_holdout_per_clip)]
    for c in sorted(h_combo):
        extra += [compose(*[(f, sev) for f in c.split("+")])
                  for sev in _holdout_severities(n_holdout_per_clip)]
    test = build_manifest(test_cache, n_per_clip=(n_test_per_clip if n_test_per_clip
                                                   is not None else n_per_clip),
                          seed=seed + 1, families=families, clean_fraction=clean_fraction,
                          n_cam=n_cam, composite_fraction=composite_fraction,
                          n_components=n_components, extra_specs=extra)
    for name, entries in (("train", train), ("test", test)):
        for e in entries:
            e["split"] = name
            e.update(_holdout_tags(e, h_fam, h_combo))
    meta = {"seed": seed, "test_fraction": test_fraction,
            "train_clips": train_ids, "test_clips": test_ids,
            "holdout_families": sorted(h_fam), "holdout_combos": sorted(h_combo),
            "composite_fraction": composite_fraction, "n_components": n_components}
    check_split(train, test, h_fam, h_combo)
    return {"train": train, "test": test, "meta": meta}


def _holdout_severities(n: int) -> list[float]:
    """Evenly spread severities in [0.3, 1.0] so held-out items cover mild to severe."""
    if n <= 0:
        return []
    if n == 1:
        return [0.7]
    return [round(0.3 + 0.7 * i / (n - 1), 3) for i in range(n)]


def check_split(train: list[dict], test: list[dict], holdout_families: Iterable[str] = (),
                holdout_combos: Iterable = ()) -> None:
    """Raise ValueError if a clip is on both sides or a held-out item leaked into train."""
    overlap = {e["clip_id"] for e in train} & {e["clip_id"] for e in test}
    if overlap:
        raise ValueError(f"clip leakage: {sorted(overlap)[:5]} in both train and test")
    h_fam = set(holdout_families)
    h_combo = {normalize_combo(c) for c in holdout_combos}
    for e in train:
        fams = spec_families(e["spec"])
        if fams & h_fam:
            raise ValueError(f"held-out family {sorted(fams & h_fam)} in train "
                             f"(clip {e['clip_id']})")
        if combo_key(e["spec"]) in h_combo:
            raise ValueError(f"held-out combo {combo_key(e['spec'])} in train "
                             f"(clip {e['clip_id']})")


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
    combos = {}
    for e in entries:
        k = e.get("combo") or combo_key(e["spec"])
        if "+" in k:
            combos[k] = combos.get(k, 0) + 1
    sev = [e["severity"] for e in entries if e["severity"] > 0]
    lines = [f"{n} examples over {clips} clips",
             "  " + ", ".join(f"{k}: {v}" for k, v in sorted(fam.items())),
             f"  clean fraction: {fam.get('clean', 0) / max(n, 1):.0%}"]
    if combos:
        lines.append("  composites: " + ", ".join(f"{k}: {v}" for k, v in sorted(combos.items())))
    ho = [e for e in entries if e.get("heldout_family") or e.get("heldout_combo")]
    if ho:
        lines.append(f"  held-out examples: {len(ho)}")
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
    b.add_argument("--composite-fraction", type=float, default=0.0)
    sp = sub.add_parser("split", help="held-out train/test manifests (by clip, family, combo)")
    sp.add_argument("--clean-cache")
    sp.add_argument("--records")
    sp.add_argument("--diag")
    sp.add_argument("--test-fraction", type=float, default=0.2)
    sp.add_argument("--holdout-family", action="append", default=[])
    sp.add_argument("--holdout-combo", action="append", default=[],
                    help='e.g. "glare+blur" (repeatable)')
    sp.add_argument("--composite-fraction", type=float, default=0.3)
    sp.add_argument("--n-per-clip", type=int, default=16)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-cam", type=int, default=7)
    sp.add_argument("--out-dir", default="outputs/uncertainty_split")
    s = sub.add_parser("summary")
    s.add_argument("manifest")
    a = ap.parse_args()

    def load_cache():
        if a.clean_cache:
            return json.load(open(a.clean_cache))
        if a.records:
            return clean_cache_from_records(a.records, a.diag)
        ap.error("provide --clean-cache or --records")

    if a.cmd == "build":
        entries = build_manifest(load_cache(), n_per_clip=a.n_per_clip, seed=a.seed,
                                 n_cam=a.n_cam, composite_fraction=a.composite_fraction)
        write_manifest(entries, a.out)
        print(manifest_summary(entries))
        print(f"-> {a.out}")
    elif a.cmd == "split":
        import os
        sp = build_split(load_cache(), test_fraction=a.test_fraction, seed=a.seed,
                         holdout_families=a.holdout_family, holdout_combos=a.holdout_combo,
                         n_per_clip=a.n_per_clip, composite_fraction=a.composite_fraction,
                         n_cam=a.n_cam)
        for name in ("train", "test"):
            path = os.path.join(a.out_dir, f"{name}.jsonl")
            write_manifest(sp[name], path)
            print(f"[{name}] {len(sp['meta'][name + '_clips'])} clips -> {path}")
            print(manifest_summary(sp[name]))
        with open(os.path.join(a.out_dir, "split_meta.json"), "w") as fh:
            json.dump(sp["meta"], fh, indent=2)
    else:
        print(manifest_summary(read_manifest(a.manifest)))


if __name__ == "__main__":
    _main()
