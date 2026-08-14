#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
run_inference_a2.py — Phase-A capture runner ported to NVIDIA Alpamayo 2 Super (34B).

Same contract as run_inference.py (v1, Alpamayo-R1-10B): K INDEPENDENT rollouts per
clip (different seeds), Axis-2 scene objects from obstacle.offline, and the SAME
records.json schema — so the whole cold harness (parser backends, axes, scorecard,
fixtures workflow) runs unchanged on the new model's outputs.

What changed vs v1 (API mapped from NVIDIA's official Aug-2026 blog + repo):
  * package alpamayo2_super; class Alpamayo2Super; from_pretrained(..., device_map="cuda:0")
  * helper.prepare_model_inputs(data, model.config, model.tokenizer) replaces the manual
    create_message + apply_chat_template + base_inputs assembly
  * sample_trajectories_from_data(...) (no _with_vlm_rollout suffix) returns
    (pred_xyz, pred_rot, logprob, extra) — arity handled defensively
  * diffusion_kwargs={"inference_step": N} is an explicit argument
  * up to 7 cameras; the loader's camera set is whatever load_physical_aiavdataset
    returns (probe_a2.py dumps it) — nothing here hardcodes 4 cameras

Run the probe FIRST on a fresh pod: python runners/probe_a2.py

Usage:
    python runners/run_inference_a2.py --clips 1 3 4 --k-rollouts 5 \
        --clip-index /workspace/alpamayo2/notebooks/clip_ids.parquet \
        --out outputs/records_a2.json --diag outputs/raw_diag_a2.json
"""

import argparse
import copy
import json
import os
import sys

# harness + sibling-runner imports work regardless of CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_inference import load_scene_objects, _extract_reasoning  # unchanged v1 logic


def _to_xy(arr):
    """(…, T, >=2) tensor/ndarray -> (T, 2) float ndarray, squeezing leading singleton dims."""
    import numpy as np
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    # if several samples survived (num_traj_samples>1), keep the first
    while a.ndim > 2:
        a = a[0]
    return a[:, :2].astype(float)


def run(clip_indices, k_rollouts, out_path, clip_index_path,
        dt=0.1, top_p=0.98, temperature=0.6, inference_steps=10,
        model_id="nvidia/Alpamayo2-Super", attn=None, diag_path=None, t0_us=5_100_000):
    import numpy as np
    import torch
    import pandas as pd
    from alpamayo2_super import helper
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    from afh.parser import parse_trace
    from afh.axes.consistency import summarize_trajectory
    from afh.trace import ClipRecord, CoCTrace

    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if attn:
        kwargs["attn_implementation"] = attn
    model = Alpamayo2Super.from_pretrained(model_id, **kwargs)

    df = pd.read_parquet(clip_index_path)
    col = "clip_id" if "clip_id" in df.columns else df.columns[0]
    clip_ids = df[col].tolist()

    records, diags = [], []
    for idx in clip_indices:
        clip_id = clip_ids[idx]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
        except Exception as e:
            print(f"[{idx}] {clip_id} skip (load failed): {e}")
            continue

        try:
            base_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
        except Exception as e:
            print(f"[{idx}] {clip_id} skip (prepare_model_inputs failed): {e}")
            continue

        gt = _to_xy(data["ego_future_xyz"]) if "ego_future_xyz" in data else None

        traces, trajs, ades, raw_first = [], [], [], None
        for k in range(k_rollouts):
            torch.manual_seed(k)
            torch.cuda.manual_seed_all(k)
            mi = helper.to_device(copy.deepcopy(base_inputs), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.sample_trajectories_from_data(
                    data=mi, top_p=top_p, temperature=temperature,
                    num_traj_samples=1,
                    diffusion_kwargs={"inference_step": inference_steps},
                    return_extra=True)
            pred_xyz, extra = out[0], out[-1]

            sentences = _extract_reasoning(extra["cot"]) if isinstance(extra, dict) and "cot" in extra else []
            claims = []
            for s in sentences:
                claims.extend(parse_trace(clip_id, k, s).claims)
            traces.append(CoCTrace(clip_id=clip_id, sample_index=k,
                                   raw_text="\n".join(sentences), claims=claims))

            xy = _to_xy(pred_xyz)
            trajs.append(summarize_trajectory([tuple(map(float, p)) for p in xy], dt=dt))
            if gt is not None and gt.shape == xy.shape:
                ades.append(float(np.linalg.norm(xy - gt, axis=1).mean()))
            if raw_first is None:
                raw_first = xy[:25].tolist()
            ade_str = f"ADE={ades[-1]:.2f} | " if ades and len(ades) == k + 1 else ""
            print(f"[{idx}] {clip_id} rollout {k}: {ade_str}"
                  f"{sentences[0][:80] if sentences else '(empty)'}")

        scene = load_scene_objects(clip_id, t0_us)
        if scene:
            print(f"   [labels] {len(scene)} scene objects near t0 "
                  f"(closest at {min((o.x**2+o.y**2)**0.5 for o in scene):.1f} m)")

        min_ade = min(ades) if ades else None
        records.append(ClipRecord(clip_id=clip_id, traces=traces, trajectories=trajs,
                                  scene_objects=scene, min_ade=min_ade))
        diags.append({"clip_id": clip_id, "ades": ades,
                      "raw_points_xy_first_rollout": raw_first,
                      "n_sentences_per_rollout": [len(t.raw_text.split(chr(10))) for t in traces]})
        ade_msg = f", minADE={min_ade:.2f}" if min_ade is not None else " (no GT ADE)"
        print(f"[{idx}] {clip_id}: {k_rollouts} rollouts{ade_msg}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
    print(f"Saved {len(records)} clip records -> {out_path}")
    if diag_path:
        with open(diag_path, "w") as f:
            json.dump(diags, f, indent=2)
        print(f"Saved diagnostics -> {diag_path}")


def main():
    ap = argparse.ArgumentParser(description="Run Alpamayo 2 Super and save ClipRecords")
    ap.add_argument("--clips", type=int, nargs="+", default=[1, 3, 4],
                    help="clip indices into the clip index parquet (same convention as v1)")
    ap.add_argument("--k-rollouts", type=int, default=5)
    ap.add_argument("--clip-index", default="/workspace/alpamayo2/notebooks/clip_ids.parquet",
                    help="parquet listing clip_ids (path may differ in the A2 repo; "
                         "fall back to the A1 repo's parquet if absent — same dataset)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--inference-steps", type=int, default=10, help="diffusion steps")
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    ap.add_argument("--attn", default=None,
                    help="only pass to override the repo default (e.g. sdpa)")
    ap.add_argument("--out", default="outputs/records_a2.json")
    ap.add_argument("--diag", default=None)
    args = ap.parse_args()
    run(args.clips, args.k_rollouts, args.out, clip_index_path=args.clip_index,
        dt=args.dt, inference_steps=args.inference_steps, model_id=args.model_id,
        attn=args.attn, diag_path=args.diag, t0_us=args.t0_us)


if __name__ == "__main__":
    main()
