#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
run_counterfactual.py — Axis 4 experiment (v1: occlusion counterfactual).

Runs Alpamayo twice on one clip:
  (A) baseline  — original frames
  (B) CF        — frames with the target causal agent's image region masked out

then scores sensitivity with afh.axes.counterfactual.

v1 masking = azimuthal band. From the agent's 3D rig position (labels/obstacle.offline,
the same loader Axis 2 uses) we compute its azimuth atan2(y, x) and paint a black vertical
band over the matching horizontal span of the camera(s) that see that bearing. It's coarse
(no per-camera intrinsics projection yet — that's the natural v2), but it reliably hides a
nearby object and is fully reproducible.

GPU pod only. Requires the alpamayo env + physical_ai_av + HF_TOKEN (see docs/SETUP_RUNPOD.md).

Usage (probe the frame tensor layout FIRST, then run):
    python runners/run_counterfactual.py --clip 0ea6fd88-... --probe
    python runners/run_counterfactual.py --clip 0ea6fd88-... --agent vehicle \
        --agent-x 14.9 --agent-y -9.2 --k-rollouts 5 --out outputs/cf_0ea6fd88.json
"""

import argparse
import copy
import json
import math
import os

import numpy as np
import torch


# ---- frame layout (confirmed from load_physical_aiavdataset source) ----
# The loader stacks 4 cameras, sorted by camera index, in this fixed order:
#   0: camera_cross_left_120fov   mount +90° (looks left)
#   1: camera_front_wide_120fov   mount   0° (looks forward)
#   2: camera_cross_right_120fov  mount -90° (looks right)
#   3: camera_front_tele_30fov    mount   0° (narrow forward)
# We mask the target agent in EVERY camera whose FOV covers its bearing.
CAMERA_LAYOUT = [
    {"name": "cross_left_120",  "mount_az": 90.0,  "fov": 120.0},
    {"name": "front_wide_120",  "mount_az": 0.0,   "fov": 120.0},
    {"name": "cross_right_120", "mount_az": -90.0, "fov": 120.0},
    {"name": "front_tele_30",   "mount_az": 0.0,   "fov": 30.0},
]


def probe_layout(clip_id, t0_us):
    """Dump the shape/semantics of image_frames so masking can be indexed correctly."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    f = data["image_frames"]
    print(f"image_frames tensor: shape={tuple(f.shape)} dtype={f.dtype} "
          f"min={f.min().item():.3f} max={f.max().item():.3f}")
    print("interpretation: (n_cameras, n_timesteps, C, H, W) before flatten(0,1)")
    print(f"  -> n_cameras={f.shape[0]}  n_timesteps={f.shape[1]}  H={f.shape[-2]} W={f.shape[-1]}")
    print("Loader order (sorted by camera index): 0=cross_left 1=front_wide 2=cross_right 3=front_tele")
    return f.shape


def occlude_frames(frames, agent_x, agent_y, band_frac=0.22, layout=CAMERA_LAYOUT):
    """
    Paint a black vertical band over the target agent's bearing in EVERY camera whose
    field of view covers that bearing (an object near a 120° camera edge is often seen
    by two cameras — masking only one leaves it visible to the model).

    frames: tensor (n_cameras, n_timesteps, C, H, W). Returns a masked copy.
    Uses each camera's real mount azimuth (from the loader's fixed order); the agent's
    world bearing is atan2(y, x) (+y left -> +az), and its offset within a camera is
    (world_az - mount_az), mapped across that camera's FOV to a horizontal fraction.
    """
    out = frames.clone()
    world_az = math.degrees(math.atan2(agent_y, agent_x))     # +y left -> +az left
    W = out.shape[-1]
    dark = out.min()
    masked_cams = []
    for cam_idx in range(min(out.shape[0], len(layout))):
        cam = layout[cam_idx]
        rel_az = world_az - cam["mount_az"]                   # bearing inside this camera
        rel_az = (rel_az + 180.0) % 360.0 - 180.0             # wrap to [-180,180]
        if abs(rel_az) > cam["fov"] / 2.0:
            continue                                          # agent not in this camera's FOV
        u = 0.5 - (rel_az / cam["fov"])                       # +rel_az (left) -> left of image
        center = int(np.clip(u, 0.0, 1.0) * W)
        bw = max(1, int(band_frac * W))
        lo, hi = max(0, center - bw // 2), min(W, center + bw // 2)
        out[cam_idx, :, :, :, lo:hi] = dark
        masked_cams.append(f"cam{cam_idx}({cam['name']}) cols[{lo}:{hi}]")
    if masked_cams:
        print(f"[occlude] agent world_az={world_az:.1f}° -> masked {', '.join(masked_cams)}")
    else:
        print(f"[occlude] WARNING agent world_az={world_az:.1f}° not covered by any camera FOV")
    return out


def run_side(frames, data, helper, processor, model, k_rollouts):
    """Run K independent rollouts on a given frame tensor; returns (traces, trajs)."""
    from afh.parser import parse_trace
    from afh.trace import CoCTrace
    from afh.axes.consistency import summarize_trajectory
    from run_inference import _extract_reasoning

    messages = helper.create_message(frames.flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    base_inputs = {"tokenized_data": inputs,
                   "ego_history_xyz": data["ego_history_xyz"],
                   "ego_history_rot": data["ego_history_rot"]}
    traces, trajs = [], []
    for k in range(k_rollouts):
        torch.manual_seed(k); torch.cuda.manual_seed_all(k)
        mi = helper.to_device(copy.deepcopy(base_inputs), "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(mi), top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True)
        sentences = _extract_reasoning(extra["cot"][0])
        claims = []
        for s in sentences:
            claims.extend(parse_trace("cf", k, s).claims)
        traces.append(CoCTrace(clip_id="cf", sample_index=k,
                               raw_text="\n".join(sentences), claims=claims))
        xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        trajs.append(summarize_trajectory([tuple(map(float, p)) for p in xy], dt=0.1))
        print(f"    rollout {k}: {(sentences[0][:70] if sentences else '(empty)')}")
    return traces, trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--agent", default="vehicle", help="target causal agent type")
    ap.add_argument("--agent-x", type=float, help="agent rig x (forward, m)")
    ap.add_argument("--agent-y", type=float, help="agent rig y (left, m)")
    ap.add_argument("--k-rollouts", type=int, default=5)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--probe", action="store_true", help="dump frame layout and exit")
    ap.add_argument("--out", default="outputs/cf_experiment.json")
    args = ap.parse_args()

    if args.probe:
        probe_layout(args.clip, args.t0_us)
        return

    if args.agent_x is None or args.agent_y is None:
        raise SystemExit("provide --agent-x and --agent-y (from the Axis-2 scene objects). "
                         "Run run_inference.py first to see the labeled positions.")

    import alpamayo_r1.helper as helper
    from alpamayo_r1.model import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from afh.axes.counterfactual import score_counterfactual

    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
                                       attn_implementation="eager").to("cuda")
    processor = helper.get_processor(model.tokenizer)

    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]

    print("=== BASELINE ===")
    base_tr, base_tj = run_side(frames, data, helper, processor, model, args.k_rollouts)
    print("=== COUNTERFACTUAL (agent occluded) ===")
    masked = occlude_frames(frames, args.agent_x, args.agent_y)
    cf_tr, cf_tj = run_side(masked, data, helper, processor, model, args.k_rollouts)

    result = score_counterfactual(args.clip, args.agent, base_tr, base_tj, cf_tr, cf_tj)
    print("\n" + result.format_report())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "clip_id": args.clip, "target_agent": args.agent,
            "agent_xy": [args.agent_x, args.agent_y],
            "baseline_traces": [t.raw_text for t in base_tr],
            "cf_traces": [t.raw_text for t in cf_tr],
            "baseline_behaviors": [sorted(t.behaviors()) for t in base_tj],
            "cf_behaviors": [sorted(t.behaviors()) for t in cf_tj],
            "score": result.score, "verdict": result.verdict,
            "baseline_citation": result.baseline_citation, "cf_citation": result.cf_citation,
        }, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
