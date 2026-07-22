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


# loader's fixed camera order (sorted by camera index) -> dataset camera_id
LOADER_CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
               "camera_cross_right_120fov", "camera_front_tele_30fov"]


def occlude_frames(frames, agent_x, agent_y, agent_z, size_x, size_y, size_z,
                   intrinsics, extrinsics, pad=1.6):
    """
    Mask the target agent by PROJECTING its 3D cuboid into every camera that sees it
    (devkit FTheta fisheye model + extrinsics), then painting a black box over the
    projected extent. This is exact on the 120° fisheye cameras, unlike a linear
    azimuth approximation.

    frames: tensor (n_cameras, n_timesteps, C, H, W). Returns a masked copy.
    The box is sized from the cuboid's 8 corners projected into the image, padded by
    `pad` so the whole vehicle (not just its center) is covered.
    """
    out = frames.clone()
    H, W = out.shape[-2], out.shape[-1]
    dark = out.min()
    center = np.array([agent_x, agent_y, agent_z], dtype=float)
    # 8 cuboid corners in rig frame (axis-aligned; orientation ignored -> pad covers it)
    hs = np.array([size_x, size_y, size_z]) / 2.0
    corners = center + np.array([[sx*hs[0], sy*hs[1], sz*hs[2]]
                                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    masked = []
    for cam_idx in range(min(out.shape[0], len(LOADER_CAMS))):
        cam_id = LOADER_CAMS[cam_idx]
        model = intrinsics.camera_models.get(cam_id)
        pose = extrinsics.sensor_poses.get(cam_id)
        if model is None or pose is None:
            continue
        pc = np.array([pose.inv().apply(c) for c in corners])   # rig -> camera frame
        if (pc[:, 2] <= 0).all():
            continue                                            # entirely behind camera
        pc = pc[pc[:, 2] > 0]                                   # keep points in front
        px = model.ray2pixel(pc)
        x0, y0 = px[:, 0].min(), px[:, 1].min()
        x1, y1 = px[:, 0].max(), px[:, 1].max()
        # pad around the projected box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * pad, (y1 - y0) * pad
        lo_x, hi_x = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2))
        lo_y, hi_y = int(max(0, cy - bh / 2)), int(min(H, cy + bh / 2))
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        out[cam_idx, :, :, lo_y:hi_y, lo_x:hi_x] = dark
        masked.append(f"cam{cam_idx}({cam_id.split('_')[1]}) box[{lo_x}:{hi_x},{lo_y}:{hi_y}]")
    if masked:
        print(f"[occlude] agent projected & masked in {', '.join(masked)}")
    else:
        print(f"[occlude] WARNING agent not visible in any camera")
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
    ap.add_argument("--agent-z", type=float, default=0.8, help="agent rig z (up, m)")
    ap.add_argument("--size-x", type=float, default=4.6, help="cuboid length (m)")
    ap.add_argument("--size-y", type=float, default=1.9, help="cuboid width (m)")
    ap.add_argument("--size-z", type=float, default=1.7, help="cuboid height (m)")
    ap.add_argument("--k-rollouts", type=int, default=5)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--probe", action="store_true", help="dump frame layout and exit")
    ap.add_argument("--dump-mask", action="store_true",
                    help="save masked cameras as PNG and exit (verify before spending rollouts)")
    ap.add_argument("--out", default="outputs/cf_experiment.json")
    args = ap.parse_args()

    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    if args.probe:
        data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
        f = data["image_frames"]
        print(f"image_frames: shape={tuple(f.shape)} (n_cam, n_t, C, H, W)  dtype={f.dtype}")
        print("loader order: 0=cross_left 1=front_wide 2=cross_right 3=front_tele")
        return

    if args.agent_x is None or args.agent_y is None:
        raise SystemExit("provide --agent-x/--agent-y (from the Axis-2 scene objects).")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    intrinsics = avdi.get_clip_feature(args.clip, "camera_intrinsics", maybe_stream=True)
    extrinsics = avdi.get_clip_feature(args.clip, "sensor_extrinsics", maybe_stream=True)
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]

    masked = occlude_frames(frames, args.agent_x, args.agent_y, args.agent_z,
                            args.size_x, args.size_y, args.size_z, intrinsics, extrinsics)

    if args.dump_mask:
        from PIL import Image
        for cam in range(frames.shape[0]):
            img = masked[cam, -1].permute(1, 2, 0).cpu().numpy().astype("uint8")
            Image.fromarray(img).save(f"maskcheck_cam{cam}.png")
            print(f"saved maskcheck_cam{cam}.png")
        print("Inspect the PNGs; if the agent is fully covered, re-run without --dump-mask.")
        return

    import alpamayo_r1.helper as helper
    from alpamayo_r1.model import AlpamayoR1
    from afh.axes.counterfactual import score_counterfactual

    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
                                       attn_implementation="eager").to("cuda")
    processor = helper.get_processor(model.tokenizer)

    print("=== BASELINE ===")
    base_tr, base_tj = run_side(frames, data, helper, processor, model, args.k_rollouts)
    print("=== COUNTERFACTUAL (agent occluded) ===")
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
