#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
project_agent.py — DIAGNOSTIC for Axis 4 masking.

Projects a labeled 3D agent (rig frame) into every loaded camera using the devkit's
real FTheta fisheye model + extrinsics, and draws a marker where it lands. This is how
we verify masking hits the right pixels on these 120° fisheye cameras (a linear azimuth
approximation does NOT — the distortion is large).

Run on the pod, then download the annotated PNGs to confirm the projection is correct
before wiring it into run_counterfactual.occlude_frames.

Usage:
    python runners/project_agent.py --clip 0ea6fd88-... --agent-x 14.9 --agent-y -9.2 --agent-z 0.8
"""

import argparse
import numpy as np


# loader's fixed camera order (sorted by camera index) -> dataset camera_id
LOADER_CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
               "camera_cross_right_120fov", "camera_front_tele_30fov"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--agent-x", type=float, required=True, help="rig x forward (m)")
    ap.add_argument("--agent-y", type=float, required=True, help="rig y left (m)")
    ap.add_argument("--agent-z", type=float, default=0.8, help="rig z up (m), ~car center height")
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    args = ap.parse_args()

    import physical_ai_av
    from PIL import Image, ImageDraw
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    intr = avdi.get_clip_feature(args.clip, "camera_intrinsics", maybe_stream=True)   # CameraIntrinsics
    extr = avdi.get_clip_feature(args.clip, "sensor_extrinsics", maybe_stream=True)   # SensorExtrinsics

    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]      # (n_cam, n_t, 3, H, W)
    n_cam = frames.shape[0]

    # agent in rig homogeneous coords
    p_rig = np.array([args.agent_x, args.agent_y, args.agent_z], dtype=float)
    print(f"agent rig xyz = {p_rig}")

    for cam_idx in range(n_cam):
        cam_id = LOADER_CAMS[cam_idx] if cam_idx < len(LOADER_CAMS) else f"cam{cam_idx}"
        model = intr.camera_models.get(cam_id)
        pose = extr.sensor_poses.get(cam_id)     # rig<-camera? or camera<-rig? check both
        if model is None or pose is None:
            print(f"cam{cam_idx} {cam_id}: no intrinsics/extrinsics")
            continue

        # extrinsic pose maps camera-frame -> rig-frame (sensor pose in rig).
        # so rig->camera is the inverse.
        p_cam = pose.inv().apply(p_rig)
        # camera convention for ray2pixel: z = optical forward. If the extrinsic uses
        # x-forward rig-style axes, we may need to reorder — print raw to inspect.
        px = model.ray2pixel(p_cam[None, :])[0]
        H, W = frames.shape[-2], frames.shape[-1]
        in_bounds = (0 <= px[0] < W) and (0 <= px[1] < H)
        z = p_cam[2]
        print(f"cam{cam_idx} {cam_id}: p_cam={np.round(p_cam,2)} "
              f"-> pixel=({px[0]:.0f},{px[1]:.0f}) in_bounds={in_bounds} front(z>0)={z>0}")

        # draw marker on the last (closest to t0) frame
        img = frames[cam_idx, -1].permute(1, 2, 0).cpu().numpy().astype("uint8")
        im = Image.fromarray(img)
        if in_bounds and z > 0:
            d = ImageDraw.Draw(im)
            r = 40
            d.ellipse([px[0]-r, px[1]-r, px[0]+r, px[1]+r], outline=(255, 0, 0), width=8)
            d.line([px[0]-r, px[1], px[0]+r, px[1]], fill=(255, 0, 0), width=4)
            d.line([px[0], px[1]-r, px[0], px[1]+r], fill=(255, 0, 0), width=4)
        im.save(f"proj_cam{cam_idx}.png")
        print(f"   saved proj_cam{cam_idx}.png")


if __name__ == "__main__":
    main()
