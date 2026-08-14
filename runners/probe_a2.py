#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_a2.py — FIRST thing to run on the Alpamayo 2 Super pod (hello-world probe).

Resolves, in one cheap session, every unknown the port depends on:
  1. loader output: keys, image_frames shape, camera_indices (how many cameras, which),
     absolute_timestamps, ego_future_xyz presence/shape;
  2. one trajectory rollout: return arity, pred_xyz shape, extra[] keys, the exact
     structure of extra["cot"] (ndarray? nesting? sentences?);
  3. one VQA call: which prepare_* function exists, result[] keys, and the raw format
     of 2D grounding output (bounding boxes);
  4. peak VRAM for the workflow.

Every stage is wrapped so a failure still prints maximum information.
Default clip is the one NVIDIA's blog validates (030c760c..., t0=2s).

Usage:
    python runners/probe_a2.py [--clip <id>] [--t0-us 2000000] [--skip-vqa]
"""

import argparse
import traceback

import torch


def stage(name):
    print(f"\n{'='*20} {name} {'='*20}")


def describe(x, name, depth=0, max_depth=3):
    """Print type/shape/dtype/sample of an arbitrary nested object."""
    pad = "  " * depth
    t = type(x).__name__
    if torch.is_tensor(x):
        print(f"{pad}{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    elif hasattr(x, "shape") and hasattr(x, "dtype"):
        print(f"{pad}{name}: {t} shape={tuple(x.shape)} dtype={x.dtype}")
        if getattr(x, "size", 0) and x.size <= 8:
            print(f"{pad}  values={x.tolist()}")
    elif isinstance(x, dict):
        print(f"{pad}{name}: dict keys={list(x.keys())}")
        if depth < max_depth:
            for k, v in x.items():
                describe(v, k, depth + 1, max_depth)
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {t} len={len(x)}")
        if depth < max_depth and len(x) > 0:
            describe(x[0], f"{name}[0]", depth + 1, max_depth)
    elif isinstance(x, str):
        s = x if len(x) <= 200 else x[:200] + "..."
        print(f"{pad}{name}: str {s!r}")
    else:
        print(f"{pad}{name}: {t} = {str(x)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="030c760c-ae38-49aa-9ad8-f5650a545d26",
                    help="default = the clip NVIDIA's blog validates")
    ap.add_argument("--t0-us", type=int, default=2_000_000)
    ap.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    ap.add_argument("--skip-vqa", action="store_true")
    args = ap.parse_args()

    stage("1/5 imports")
    from alpamayo2_super import helper
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    import alpamayo2_super.text_tasks as text_tasks
    print("imports OK")
    print("text_tasks exports:", [n for n in dir(text_tasks) if not n.startswith("_")])

    stage("2/5 data loader")
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    describe(data, "data")
    if "camera_indices" in data:
        print("\n>>> camera_indices:", data["camera_indices"].tolist()
              if hasattr(data["camera_indices"], "tolist") else data["camera_indices"])
        # index -> name mapping learned on the A1 loader; verify it still holds for A2
        idx2name = {0: "cross_left_120", 1: "front_wide_120", 2: "cross_right_120",
                    3: "rear_left_70", 4: "rear_tele_30", 5: "rear_right_70",
                    6: "front_tele_30"}
        ci = data["camera_indices"].tolist() if hasattr(data["camera_indices"], "tolist") else list(data["camera_indices"])
        for tensor_idx, cam_index in enumerate(ci):
            print(f"    tensor[{tensor_idx}] = camera_index {cam_index} -> {idx2name.get(int(cam_index), '?')} (mapping TO VERIFY)")

    stage("3/5 model load")
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0")
    print("model loaded")
    print("methods:", [m for m in dir(model) if "sample" in m or "generate" in m])

    stage("4/5 one trajectory rollout")
    try:
        model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
        model_inputs = helper.to_device(model_inputs, "cuda")
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.sample_trajectories_from_data(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=1, diffusion_kwargs={"inference_step": 10},
                return_extra=True)
        print(f"return arity: {len(out)}")
        pred_xyz, pred_rot = out[0], out[1]
        extra = out[-1]
        describe(pred_xyz, "pred_xyz")
        describe(pred_rot, "pred_rot")
        if len(out) == 4:
            describe(out[2], "logprob")
        describe(extra, "extra", max_depth=2)
        if isinstance(extra, dict) and "cot" in extra:
            cot = extra["cot"]
            print("\n>>> extra['cot'] deep dive:")
            describe(cot, "cot", max_depth=4)
            try:
                flat = cot.reshape(-1) if hasattr(cot, "reshape") else cot
                print("cot flat[0]:", str(flat[0])[:400])
            except Exception as e:
                print("cot flatten failed:", e)
        print(f"\n>>> peak VRAM (trajectory): {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    except Exception:
        print("TRAJECTORY ROLLOUT FAILED:")
        traceback.print_exc()

    if not args.skip_vqa:
        stage("5/5 one VQA call (grounding format discovery)")
        try:
            question = ("Which specific objects in the scene most influence the ego "
                        "vehicle's next maneuver? Point them out.")
            if hasattr(text_tasks, "prepare_vqa_inputs"):
                task_inputs = text_tasks.prepare_vqa_inputs(
                    data=data, model_config=model.config, tokenizer=model.tokenizer,
                    question=question)
                print("used prepare_vqa_inputs")
            else:
                task_inputs = text_tasks.prepare_text_generation_inputs(
                    data=data, model_config=model.config, tokenizer=model.tokenizer,
                    task="vqa")
                print("fallback: prepare_text_generation_inputs(task='vqa')")
            task_inputs = helper.to_device(task_inputs, "cuda")
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = text_tasks.generate_text(
                    model, task_inputs, top_p=1.0, temperature=0.1, max_new_tokens=1024)
            describe(result, "vqa result", max_depth=3)
            for key in ("answer", "grounding", "boxes", "bboxes", "cot"):
                if isinstance(result, dict) and key in result:
                    print(f"\n>>> result[{key!r}]:")
                    describe(result[key], key, max_depth=4)
            print(f"\n>>> peak VRAM (VQA): {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        except Exception:
            print("VQA FAILED:")
            traceback.print_exc()

    stage("probe done")
    print("Paste this FULL output back into the chat; it resolves every port unknown.")


if __name__ == "__main__":
    main()
