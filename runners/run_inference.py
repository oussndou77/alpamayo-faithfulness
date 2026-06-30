#!/usr/bin/env python3
"""
run_inference.py — PHASE A: run Alpamayo on clips and save ClipRecords to JSON.

Runs on a GPU pod (>=24 GB VRAM; ~48 GB recommended so the eager attention path fits).
Based on NVlabs/alpamayo notebooks/inference.ipynb. Produces the JSON that the cold
harness (afh.runner) consumes — GPU work happens once here, all scoring is CPU afterwards.

Design notes (validated on a real pod, June 2026):
  * attn_implementation="eager": this model does NOT support sdpa, and flash_attention_2
    requires building flash-attn. eager works out of the box (needs ~48 GB for headroom).
  * dt=0.1: real Alpamayo trajectories are 64 waypoints @ 10 Hz (6.4 s), so dt=0.1 s.
  * K INDEPENDENT rollouts per clip (different seeds), NOT one call with num_traj_samples=K.
    A single multi-sample call shares ONE Chain-of-Causation trace across the K
    trajectories, so it can't exercise the stability axis (all K reasonings are identical).
    Independent rollouts give genuine reasoning variation for axis 3.
  * extra["cot"][0] is a numpy array of reasoning sentences (often several per rollout);
    each is parsed into claims, all collected into that rollout's CoCTrace.

API recap (official notebook):
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
                                       attn_implementation="eager").to("cuda")
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(clip_id)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(messages, ...)
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True)

Usage (on pod, from inside the alpamayo repo with afh on PYTHONPATH):
    python run_inference.py --clips 0 1 2 --k-rollouts 5 \
        --clip-index notebooks/clip_ids.parquet --out outputs/records.json
"""

import argparse
import json
import copy
import os


def _extract_reasoning(cot_entry):
    """extra['cot'][0] -> ordered list of unique reasoning sentences (handles ndarray / nesting)."""
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
            return
        if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
        else:
            out.append(str(x))

    walk(cot_entry)
    seen, res = set(), []
    for s in out:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def run(clip_indices, k_rollouts, out_path, clip_index_path="notebooks/clip_ids.parquet",
        dt=0.1, top_p=0.98, temperature=0.6, max_gen=256,
        attn="eager", model_id="nvidia/Alpamayo-R1-10B", diag_path=None, t0_us=5_100_000):
    import numpy as np
    import torch
    import pandas as pd
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # harness imports (parse + summarize here so the JSON is fully populated)
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from afh.parser import parse_trace
    from afh.axes.consistency import summarize_trajectory
    from afh.trace import ClipRecord, CoCTrace

    model = AlpamayoR1.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn).to("cuda")
    processor = helper.get_processor(model.tokenizer)

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
        frames = data["image_frames"].flatten(0, 1)
        messages = helper.create_message(frames)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        base_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        gt = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]   # (T, 2)

        traces, trajs, ades, raw_first = [], [], [], None
        for k in range(k_rollouts):
            torch.manual_seed(k)
            torch.cuda.manual_seed_all(k)
            mi = helper.to_device(copy.deepcopy(base_inputs), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(mi), top_p=top_p, temperature=temperature,
                    num_traj_samples=1, max_generation_length=max_gen, return_extra=True)

            sentences = _extract_reasoning(extra["cot"][0])
            claims = []
            for s in sentences:
                claims.extend(parse_trace(clip_id, k, s).claims)
            traces.append(CoCTrace(clip_id=clip_id, sample_index=k,
                                   raw_text="\n".join(sentences), claims=claims))

            xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]   # (T, 2)
            trajs.append(summarize_trajectory([tuple(map(float, p)) for p in xy], dt=dt))
            ades.append(float(np.linalg.norm(xy - gt, axis=1).mean()))
            if raw_first is None:
                raw_first = xy[:25].tolist()
            print(f"[{idx}] {clip_id} rollout {k}: ADE={ades[-1]:.2f} | "
                  f"{sentences[0][:80] if sentences else '(empty)'}")

        # TODO(Phase A, axis 2): populate scene_objects from the dataset's label features
        # (avdi.get_clip_feature(clip_id, avdi.features.LABELS....)); map object_type +
        # 3D location into the ego frame at t0. Left empty until that API is wired in.
        scene = []

        min_ade = min(ades) if ades else None
        records.append(ClipRecord(clip_id=clip_id, traces=traces, trajectories=trajs,
                                   scene_objects=scene, min_ade=min_ade))
        diags.append({"clip_id": clip_id, "ades": ades,
                      "raw_points_xy_first_rollout": raw_first,
                      "n_sentences_per_rollout": [len(t.raw_text.split(chr(10))) for t in traces]})
        print(f"[{idx}] {clip_id}: {k_rollouts} rollouts, minADE={min_ade:.2f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
    print(f"Saved {len(records)} clip records -> {out_path}")
    if diag_path:
        with open(diag_path, "w") as f:
            json.dump(diags, f, indent=2)
        print(f"Saved diagnostics -> {diag_path}")


def main():
    ap = argparse.ArgumentParser(description="Run Alpamayo and save ClipRecords (Phase A)")
    ap.add_argument("--clips", type=int, nargs="+", default=[0, 1, 2],
                    help="clip indices into the clip index parquet")
    ap.add_argument("--k-rollouts", type=int, default=5,
                    help="independent rollouts per clip (different seeds) -> stability axis")
    ap.add_argument("--clip-index", default="notebooks/clip_ids.parquet",
                    help="path to the parquet listing clip_ids")
    ap.add_argument("--dt", type=float, default=0.1, help="trajectory timestep (10 Hz -> 0.1)")
    ap.add_argument("--attn", default="eager", help="attn implementation (eager works OOTB)")
    ap.add_argument("--out", default="outputs/records.json")
    ap.add_argument("--diag", default=None, help="optional path for raw diagnostics JSON")
    args = ap.parse_args()
    run(args.clips, args.k_rollouts, args.out, clip_index_path=args.clip_index,
        dt=args.dt, attn=args.attn, diag_path=args.diag)


if __name__ == "__main__":
    main()
