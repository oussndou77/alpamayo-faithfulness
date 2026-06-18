#!/usr/bin/env python3
"""
run_inference.py — PHASE A: run Alpamayo on clips and save ClipRecords to JSON.

Runs on a GPU pod (>=24 GB VRAM). Based on NVlabs/alpamayo notebooks/inference.ipynb.
Produces the JSON that the cold harness (afh.runner) consumes — so the GPU work happens
once, here, and all scoring happens on CPU afterwards.

API recap (from the official notebook):
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(clip_id)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(messages, ...)
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=N, max_generation_length=256, return_extra=True)
    # extra["cot"][0]  -> Chain-of-Causation trace(s)
    # pred_xyz         -> [batch, num_traj_sets, num_traj_samples, T, 3]

Usage (on pod):
    python runners/run_inference.py --clips 774 --num-samples 5 --out outputs/records.json
"""

import argparse
import json
import copy
import os

# TODO(Phase A): these imports require the alpamayo package + GPU env (see SETUP_RUNPOD.md).
# Kept inside main() so this file imports cleanly on a laptop for inspection.


def run(clip_indices, num_samples, out_path, top_p=0.98, temperature=0.6,
        max_gen=256, model_id="nvidia/Alpamayo-R1-10B"):
    import numpy as np
    import torch
    import pandas as pd
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # local imports of the harness (to summarize trajectories + scene right here)
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from afh.parser import parse_trace
    from afh.axes.consistency import summarize_trajectory
    from afh.trace import ClipRecord, SceneObject

    model = AlpamayoR1.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
    records = []

    for idx in clip_indices:
        clip_id = clip_ids[idx]
        data = load_physical_aiavdataset(clip_id)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        model_inputs = helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, "cuda")

        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(model_inputs), top_p=top_p, temperature=temperature,
                num_traj_samples=num_samples, max_generation_length=max_gen, return_extra=True)

        # extra["cot"] holds the CoC trace(s); pred_xyz is [b, sets, samples, T, 3]
        cots = extra["cot"][0]
        traces, trajs = [], []
        for i in range(pred_xyz.shape[2]):
            raw = cots[i] if isinstance(cots, (list, tuple)) else str(cots)
            traces.append(parse_trace(clip_id, i, raw))
            xy = pred_xyz.cpu().numpy()[0, 0, i, :, :2]
            trajs.append(summarize_trajectory([tuple(p) for p in xy], dt=0.5))

        # TODO(Phase A): populate scene_objects from the dataset's cuboid / 2D-box labels.
        # The loader exposes object annotations; map object_type + 3D location (ego frame).
        scene = []  # SceneObject(object_type=..., x=..., y=...)

        # accuracy (reported separately): minADE vs logged future
        gt = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]
        pxy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        min_ade = float(np.linalg.norm(pxy - gt[None].transpose(0, 2, 1), axis=1).mean(-1).min())

        records.append(ClipRecord(clip_id=clip_id, traces=traces, trajectories=trajs,
                                   scene_objects=scene, min_ade=min_ade))
        print(f"[{clip_id}] {num_samples} samples, minADE={min_ade:.2f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
    print(f"Saved {len(records)} clip records -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Run Alpamayo and save ClipRecords (Phase A)")
    ap.add_argument("--clips", type=int, nargs="+", default=[774],
                    help="clip indices into clip_ids.parquet")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="num_traj_samples (>=2 to exercise the stability axis)")
    ap.add_argument("--out", default="outputs/records.json")
    args = ap.parse_args()
    run(args.clips, args.num_samples, args.out)


if __name__ == "__main__":
    main()
