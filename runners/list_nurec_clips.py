#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
list_nurec_clips.py — dump all NuRec clip UUIDs to a text file (no download).

Run once on the pod (venv active, HF_TOKEN exported):
    python runners/list_nurec_clips.py --out outputs/nurec_clips.txt

Then use with the Phase-G selector:
    python runners/select_clips_g.py \\
        --clip-index /workspace/alpamayo2/notebooks/clip_ids.parquet \\
        --nurec-list outputs/nurec_clips.txt \\
        --out outputs/phase_g_nurec_candidates.csv

Cross-checks your audited clips (0ea6fd88, 06b483cf, 0a1ef808) automatically.
"""

import argparse
import os


AUDITED = [
    "0ea6fd88-dcdd-434e-9fa3-56ce0fb35bf2",  # parked-car scene (main audit clip)
    "06b483cf-6d9c-4b18-b54b-4429c80867e3",  # lane-change clip
    "0a1ef808-c891-4da2-919d-c44aa3be5085",  # curve-adapt clip
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/nurec_clips.txt")
    ap.add_argument("--version", default="26.04_release",
                    help="NuRec release folder name inside the dataset repo")
    ap.add_argument("--repo-id", default="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec")
    args = ap.parse_args()

    from huggingface_hub import HfApi, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    api = HfApi()
    print(f"[nurec] listing {args.repo_id} / sample_set/{args.version} ...")
    items = list(api.list_repo_tree(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=f"sample_set/{args.version}",
        recursive=False,
    ))
    uuids = [it.path.split("/")[-1] for it in items
             if "/" in it.path and len(it.path.split("/")[-1]) >= 8]

    print(f"[nurec] {len(uuids)} clips found")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(uuids) + "\n")
    print(f"[nurec] saved -> {args.out}")

    # cross-check with audited clips
    print("\n[nurec] cross-check with audited clips:")
    nurec_set = set(uuids)
    for cid in AUDITED:
        match = cid in nurec_set
        prefix_match = any(u.startswith(cid[:8]) for u in nurec_set)
        status = "✓ EXACT MATCH" if match else ("~ prefix match" if prefix_match else "✗ not found")
        print(f"  {cid[:8]}...  {status}")

    print("\nNext step:")
    print(f"  python runners/select_clips_g.py "
          f"--clip-index /workspace/alpamayo2/notebooks/clip_ids.parquet "
          f"--nurec-list {args.out} "
          f"--out outputs/phase_g_nurec_candidates.csv")


if __name__ == "__main__":
    main()
