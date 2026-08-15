#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_vqa_mask_a2.py — ask Alpamayo 2 Super what it SEES on clean vs masked frames.

Purpose: disambiguate the Axis-4 result on this clip. The CoC text keeps saying
"parked vehicle" under every condition. Two rival explanations:
  (a) the black mask is itself perceived as a vehicle/object  -> VQA on masked frames
      should still answer "yes, a parked vehicle";
  (b) the CoC narrative is decoupled from perception          -> VQA should answer
      "no vehicle" (or describe an artifact) while the CoC still cites one.

Runs the same question on clean frames and on track-masked frames, prints both.

Usage:
    python runners/probe_vqa_mask_a2.py --clip 0ea6fd88-... --track-id 9
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch

from run_counterfactual_a2 import occlude_frames  # projection masking, unchanged


QUESTIONS = [
    "Is there a parked vehicle in the lane ahead of the ego vehicle? Answer yes or no, then describe what you see in the lane ahead.",
    "Describe any unusual visual artifacts or occlusions in the front camera view.",
]


def ask(model, helper, text_tasks, data, question):
    ti = text_tasks.prepare_vqa_inputs(data=data, model=model, question=question)
    ti = helper.to_device(ti, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = text_tasks.generate_text(model, ti, top_p=1.0, temperature=0.1,
                                       max_new_tokens=512)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    args = ap.parse_args()

    import physical_ai_av
    from alpamayo2_super import helper
    from alpamayo2_super import text_tasks
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    intr = avdi.get_clip_feature(args.clip, "camera_intrinsics", maybe_stream=True)
    extr = avdi.get_clip_feature(args.clip, "sensor_extrinsics", maybe_stream=True)
    obst = avdi.get_clip_feature(args.clip, "obstacle.offline", maybe_stream=True)["obstacle.offline"]
    obst["track_id"] = obst["track_id"].astype(str)
    tr = obst[obst["track_id"] == str(args.track_id)].sort_values("timestamp_us")
    if tr.empty:
        raise SystemExit(f"track {args.track_id} not found")
    size = (float(tr["size_x"].median()), float(tr["size_y"].median()),
            float(tr["size_z"].median()))

    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    ts = data["absolute_timestamps"].cpu().numpy()
    masked_frames = occlude_frames(data["image_frames"], ts, tr, size, intr, extr)
    data_masked = dict(data)
    data_masked["image_frames"] = masked_frames

    model = Alpamayo2Super.from_pretrained("nvidia/Alpamayo2-Super",
                                           dtype=torch.bfloat16, device_map="cuda:0")

    for q in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"QUESTION: {q}")
        print("-" * 70)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        clean = ask(model, helper, text_tasks, data, q)
        print(">>> CLEAN frames:")
        print(clean if isinstance(clean, str) else str(clean))
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        cf = ask(model, helper, text_tasks, data_masked, q)
        print("\n>>> MASKED frames (track occluded):")
        print(cf if isinstance(cf, str) else str(cf))


if __name__ == "__main__":
    main()
