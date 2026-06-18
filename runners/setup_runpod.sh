#!/usr/bin/env bash
# setup_runpod.sh — environment setup for running Alpamayo-R1-10B inference on a GPU pod.
# Target: a CUDA pod with >=24 GB VRAM (RTX 3090 / A100 / H100), per the Alpamayo README.
#
# This is a first-pass recipe; expect to refine it during Phase A against the real repo,
# exactly as we hardened the CARLA setup script. Steps are intentionally explicit.
set -e

echo "=== Alpamayo-Faithfulness :: GPU pod setup ==="

# 1) System / Python
python3 --version
pip install --upgrade pip

# 2) Clone the official Alpamayo repo (inference code is Apache 2.0)
if [ ! -d alpamayo ]; then
  git clone https://github.com/NVlabs/alpamayo.git
fi
cd alpamayo

# 3) Install Alpamayo + its deps (follow the repo's instructions; this is the typical path)
pip install -e . || pip install -r requirements.txt || true
# torch must match the pod's CUDA; install the matching wheel if the above didn't.
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 4) Hugging Face auth (weights + dataset are gated; accept the licenses on HF first).
#    DO NOT hardcode the token. Export HF_TOKEN in the shell, or use `huggingface-cli login`.
if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN not set. Run: export HF_TOKEN=...   (after accepting the model+dataset licenses on huggingface.co)"
fi
pip install -U "huggingface_hub" mediapy pandas

# 5) Sanity note: the model card is nvidia/Alpamayo-R1-10B; the example data comes from
#    nvidia/PhysicalAI-Autonomous-Vehicles. The inference notebook expects a clip_ids.parquet.
echo "Next: from the repo, run the official notebook once to confirm inference works,"
echo "then run ../runners/run_inference.py to dump ClipRecords for the faithfulness harness."
echo "=== setup complete (verify torch.cuda is True above) ==="
