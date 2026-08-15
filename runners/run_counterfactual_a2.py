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
import sys

import numpy as np
import torch

# make the harness package (afh/) and sibling runners importable regardless of CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# FULL camera-index -> camera_id map (loader's name_to_index, verified in v1).
# The loader may return 4 OR 7 cameras depending on the clip; the tensor order is
# data["camera_indices"], NEVER a hardcoded list. (A hardcoded 4-cam list caused a
# mask LEAK on a 7-cam clip: front_tele stayed unmasked while rear_left got a
# spurious box — caught by visual inspection of all maskcheck PNGs.)
CAM_INDEX_TO_ID = {
    0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov", 3: "camera_rear_left_70fov",
    4: "camera_rear_tele_30fov", 5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}


def _interp_track_xyz(track_df, t_us):
    """Linear-interpolate a track's rig xyz at time t_us (track_df sorted by timestamp)."""
    tx = track_df["timestamp_us"].to_numpy(dtype=float)
    x = np.interp(t_us, tx, track_df["center_x"].to_numpy(dtype=float))
    y = np.interp(t_us, tx, track_df["center_y"].to_numpy(dtype=float))
    z = np.interp(t_us, tx, track_df["center_z"].to_numpy(dtype=float))
    return np.array([x, y, z])


def occlude_frames(frames, frame_timestamps, camera_indices, track_df, size_xyz,
                   intrinsics, extrinsics, pad=1.6):
    """
    Mask the target track by projecting its 3D cuboid into every camera that sees it,
    at the ACTUAL timestamp of each camera frame.

    Two subtleties learned the hard way on this dataset:
      * the 4 cameras are NOT time-synchronized (front_wide[-1]=5.076s while
        cross_left[-1]=5.094s), and the last frame is not exactly t0 — so a fixed t0
        position mis-projects. We interpolate the track to each camera's own timestamp.
      * obstacle-label track_id is a STRING; the caller must filter with str ids.

    frames: (n_cam, n_t, C, H, W). frame_timestamps: (n_cam, n_t) absolute us.
    track_df: this track's obstacle rows (already filtered), sorted by timestamp.
    size_xyz: (size_x, size_y, size_z) cuboid extents in meters.
    Returns a masked copy.
    """
    out = frames.clone()
    H, W = out.shape[-2], out.shape[-1]
    dark = out.min()
    hs = np.array(size_xyz) / 2.0
    corner_signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    masked = []
    for cam_idx in range(out.shape[0]):
        cam_id = CAM_INDEX_TO_ID.get(int(camera_indices[cam_idx]))
        if cam_id is None:
            continue
        model = intrinsics.camera_models.get(cam_id)
        pose = extrinsics.sensor_poses.get(cam_id)
        if model is None or pose is None:
            continue
        # mask every time-step of this camera at its own timestamp
        boxes = []
        for t_idx in range(out.shape[1]):
            t_us = float(frame_timestamps[cam_idx, t_idx])
            center = _interp_track_xyz(track_df, t_us)
            corners = center + corner_signs * hs
            pc = np.array([pose.inv().apply(c) for c in corners])
            if (pc[:, 2] <= 0).all():
                continue
            pc = pc[pc[:, 2] > 0]
            px = model.ray2pixel(pc)
            x0, y0, x1, y1 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bw, bh = (x1 - x0) * pad, (y1 - y0) * pad
            lo_x, hi_x = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2))
            lo_y, hi_y = int(max(0, cy - bh / 2)), int(min(H, cy + bh / 2))
            if lo_x < hi_x and lo_y < hi_y:
                out[cam_idx, t_idx, :, lo_y:hi_y, lo_x:hi_x] = dark
                boxes.append(t_idx)
        if boxes:
            masked.append(f"cam{cam_idx}({cam_id.split('_')[1]}) t={boxes}")
    if masked:
        print(f"[occlude] projected & masked: {', '.join(masked)}")
    else:
        print("[occlude] WARNING agent not visible in any camera/timestep")
    return out


def _to_xy(arr):
    """(…, T, >=2) -> (T, 2) float ndarray, squeezing leading singleton dims."""
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    while a.ndim > 2:
        a = a[0]
    return a[:, :2].astype(float)


def run_side(masked_frames, data, helper, model, k_rollouts, seed_offset=0):
    """
    Run K independent rollouts on Alpamayo 2 Super with the given (masked) frames.

    A2 rebuilds model inputs from `data` via prepare_model_inputs, so we inject the
    masked frame tensor into data['image_frames'] first, then let the model's own
    pipeline consume it. This is how the occlusion reaches the model on A2.
    """
    from afh.parser import parse_trace
    from afh.trace import CoCTrace
    from afh.axes.consistency import summarize_trajectory
    from run_inference import _extract_reasoning

    data_masked = dict(data)
    data_masked["image_frames"] = masked_frames
    base_inputs = helper.prepare_model_inputs(data_masked, model.config, model.tokenizer)

    traces, trajs = [], []
    for k in range(k_rollouts):
        seed = k + seed_offset
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        mi = helper.to_device(copy.deepcopy(base_inputs), "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.sample_trajectories_from_data(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                diffusion_kwargs={"inference_step": 10}, return_extra=True)
        pred_xyz, extra = out[0], out[-1]
        sentences = _extract_reasoning(extra["cot"]) if isinstance(extra, dict) and "cot" in extra else []
        claims = []
        for s in sentences:
            claims.extend(parse_trace("cf", k, s).claims)
        traces.append(CoCTrace(clip_id="cf", sample_index=k,
                               raw_text="\n".join(sentences), claims=claims))
        xy = _to_xy(pred_xyz)
        trajs.append(summarize_trajectory([tuple(map(float, p)) for p in xy], dt=0.1))
        print(f"    rollout {k}: {(sentences[0][:70] if sentences else '(empty)')}")
    return traces, trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--agent", default="vehicle", help="target causal agent type")
    ap.add_argument("--track-id", help="obstacle track_id to occlude (string, e.g. '9')")
    ap.add_argument("--blackout", action="store_true",
                    help="replace ALL frames with pure black instead of masking a track: measures how much of the CoC is driven by vision at all vs ego-motion/prior")
    ap.add_argument("--null-control", action="store_true",
                    help="ALSO run a second UNMASKED baseline with different seeds, to measure the\n                          sampling-noise floor: any occlusion effect must exceed this to mean anything")
    ap.add_argument("--control-track-id", default=None,
                    help="a DISTRACTOR track the model does NOT cite; runs a negative-control "
                         "occlusion and reports the causal-vs-control contrast (validates the probe)")
    ap.add_argument("--size-x", type=float, default=None, help="cuboid length override (m)")
    ap.add_argument("--size-y", type=float, default=None, help="cuboid width override (m)")
    ap.add_argument("--size-z", type=float, default=None, help="cuboid height override (m)")
    ap.add_argument("--k-rollouts", type=int, default=5)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--probe", action="store_true", help="dump frame layout and exit")
    ap.add_argument("--dump-mask", action="store_true",
                    help="save masked cameras as PNG and exit (verify before spending rollouts)")
    ap.add_argument("--out", default="outputs/cf_experiment.json")
    args = ap.parse_args()

    import physical_ai_av
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    if args.probe:
        data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
        f = data["image_frames"]
        print(f"image_frames: shape={tuple(f.shape)} (n_cam, n_t, C, H, W)  dtype={f.dtype}")
        print("loader order: 0=cross_left 1=front_wide 2=cross_right 3=front_tele")
        return

    if args.track_id is None and not args.blackout:
        raise SystemExit("provide --track-id (the obstacle track to occlude; run run_inference "
                         "to see labeled objects, or list tracks near t0).")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    intrinsics = avdi.get_clip_feature(args.clip, "camera_intrinsics", maybe_stream=True)
    extrinsics = avdi.get_clip_feature(args.clip, "sensor_extrinsics", maybe_stream=True)
    obst = avdi.get_clip_feature(args.clip, "obstacle.offline", maybe_stream=True)["obstacle.offline"]
    obst["track_id"] = obst["track_id"].astype(str)      # track_id is a STRING in the labels
    if not args.blackout:
        track_df = obst[obst["track_id"] == str(args.track_id)].sort_values("timestamp_us")
        if track_df.empty:
            raise SystemExit(f"track_id {args.track_id!r} not found. Present: "
                             f"{sorted(obst['track_id'].unique().tolist())}")
        sx = args.size_x if args.size_x is not None else float(track_df["size_x"].median())
        sy = args.size_y if args.size_y is not None else float(track_df["size_y"].median())
        sz = args.size_z if args.size_z is not None else float(track_df["size_z"].median())
        print(f"target track {args.track_id}: {len(track_df)} obs, cuboid ~{sx:.1f}x{sy:.1f}x{sz:.1f}m")

    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]
    ts = data["absolute_timestamps"].cpu().numpy()
    cam_idx_arr = data["camera_indices"].cpu().numpy() if hasattr(data["camera_indices"], "cpu") else np.asarray(data["camera_indices"])
    print(f"[cams] tensor order = {cam_idx_arr.tolist()} -> "
          f"{[CAM_INDEX_TO_ID.get(int(i),'?').split('_',1)[1] for i in cam_idx_arr]}")

    if args.blackout:
        masked = frames.clone()
        masked[:] = frames.min()
        print("[blackout] ALL cameras, ALL timesteps replaced with pure black")
    else:
        masked = occlude_frames(frames, ts, cam_idx_arr, track_df, (sx, sy, sz), intrinsics, extrinsics)

    if args.dump_mask:
        from PIL import Image
        for cam in range(frames.shape[0]):
            img = masked[cam, -1].permute(1, 2, 0).cpu().numpy().astype("uint8")
            Image.fromarray(img).save(f"maskcheck_cam{cam}.png")
            print(f"saved maskcheck_cam{cam}.png")
        print("Inspect the PNGs; if the agent is fully covered, re-run without --dump-mask.")
        return

    from alpamayo2_super import helper
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from afh.axes.counterfactual import score_counterfactual

    model = Alpamayo2Super.from_pretrained("nvidia/Alpamayo2-Super", dtype=torch.bfloat16,
                                           device_map="cuda:0")

    print("=== BASELINE ===")
    base_tr, base_tj = run_side(frames, data, helper, model, args.k_rollouts)
    print("=== COUNTERFACTUAL (causal agent occluded) ===")
    cf_tr, cf_tj = run_side(masked, data, helper, model, args.k_rollouts)

    result = score_counterfactual(args.clip, args.agent, base_tr, base_tj, cf_tr, cf_tj)
    print("\n" + result.format_report())

    payload = {
        "clip_id": args.clip, "target_agent": args.agent,
        "track_id": str(args.track_id) if args.track_id else "BLACKOUT",
        "baseline_traces": [t.raw_text for t in base_tr],
        "cf_traces": [t.raw_text for t in cf_tr],
        "baseline_behaviors": [sorted(t.behaviors()) for t in base_tj],
        "cf_behaviors": [sorted(t.behaviors()) for t in cf_tj],
        "score": result.score, "verdict": result.verdict,
        "baseline_citation": result.baseline_citation, "cf_citation": result.cf_citation,
    }

    # optional negative control: occlude a distractor the model does NOT cite
    if args.control_track_id is not None:
        from afh.axes.counterfactual import score_control_contrast
        ctrl_df = obst[obst["track_id"] == str(args.control_track_id)].sort_values("timestamp_us")
        if ctrl_df.empty:
            raise SystemExit(f"control track_id {args.control_track_id!r} not found.")
        cxs = float(ctrl_df["size_x"].median())
        cys = float(ctrl_df["size_y"].median())
        czs = float(ctrl_df["size_z"].median())
        print(f"\ncontrol track {args.control_track_id}: {len(ctrl_df)} obs, "
              f"cuboid ~{cxs:.1f}x{cys:.1f}x{czs:.1f}m")
        ctrl_masked = occlude_frames(frames, ts, cam_idx_arr, ctrl_df, (cxs, cys, czs), intrinsics, extrinsics)
        print("=== NEGATIVE CONTROL (distractor occluded) ===")
        ctrl_tr, ctrl_tj = run_side(ctrl_masked, data, helper, model, args.k_rollouts)
        ctrl_result = score_counterfactual(args.clip, args.agent, base_tr, base_tj, ctrl_tr, ctrl_tj)
        print("\n" + ctrl_result.format_report())

        contrast = score_control_contrast(args.clip, result, ctrl_result)
        print("\n" + contrast.format_report())
        payload["control_track_id"] = str(args.control_track_id)
        payload["control_cf_traces"] = [t.raw_text for t in ctrl_tr]
        payload["control_behavior_change"] = contrast.control_behavior_change
        payload["causal_behavior_change"] = contrast.causal_behavior_change
        payload["contrast"] = contrast.contrast
        payload["valid_probe"] = contrast.valid_probe

    # null control: baseline vs a SECOND baseline (no mask, different seeds).
    # This is the noise floor. Any masked-condition change at or below it is meaningless.
    if args.null_control:
        print("\n=== NULL CONTROL (unmasked, different seeds) ===")
        null_tr, null_tj = run_side(frames, data, helper, model, args.k_rollouts,
                                    seed_offset=1000)
        null_result = score_counterfactual(args.clip, args.agent, base_tr, base_tj,
                                           null_tr, null_tj)
        print("\n" + null_result.format_report())
        floor = null_result.behavior_change
        print(f"\n>>> NOISE FLOOR (baseline vs baseline): {floor:.0%} behavior change")
        print(f">>> causal occlusion measured: {result.behavior_change:.0%}")
        if result.behavior_change <= floor + 1e-9:
            print(">>> VERDICT: the occlusion effect does NOT exceed sampling noise. "
                  "Axis-4 numbers on this clip are not interpretable as causal.")
        else:
            print(f">>> occlusion exceeds noise floor by "
                  f"{result.behavior_change - floor:+.0%}")
        payload["null_control_behavior_change"] = floor
        payload["null_control_traces"] = [t.raw_text for t in null_tr]
        payload["exceeds_noise_floor"] = bool(result.behavior_change > floor)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
