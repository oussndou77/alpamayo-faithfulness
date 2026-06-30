#!/usr/bin/env bash
# setup_runpod.sh — set up Alpamayo-R1-10B inference on a GPU pod (Phase A).
# Validated on RunPod, June 2026 (image: runpod/pytorch ...cu1281-torch280-ubuntu2404).
#
# Target GPU: ~48 GB VRAM (L40S / A100). 24 GB is the model's stated minimum but the
# eager-attention path OOMs at 24 GB on the VLM rollout — use 48 GB for headroom.
#
# Run from /workspace so the build lands on a persistent Network Volume.
set -e

echo "=== Alpamayo-Faithfulness :: GPU pod setup ==="

# 1) Clone the official Alpamayo repo (inference code, Apache 2.0)
cd /workspace
if [ ! -d alpamayo ]; then
  git clone https://github.com/NVlabs/alpamayo.git
fi
cd alpamayo

# 2) uv + a Python 3.12 venv (Alpamayo requires exactly 3.12.x; uv is the supported manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ar1_venv --python 3.12
# shellcheck disable=SC1091
source ar1_venv/bin/activate

# 3) Install deps EXCEPT flash-attn (avoids a long/fragile build; we use eager attention).
#    torch 2.8.0 / transformers 4.57.1 are pinned by the repo's pyproject/uv.lock.
uv sync --active --no-install-package flash-attn
uv pip install hf_transfer          # the RunPod image sets HF_HUB_ENABLE_HF_TRANSFER=1

# 4) Cache weights on the persistent volume so the 22 GB download happens only once.
export HF_HOME=/workspace/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

echo ""
echo "=== setup done (verify 'cuda True' above) ==="
echo "Next, in THIS shell:"
echo "  source /workspace/alpamayo/ar1_venv/bin/activate   # if you reconnect"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  export HF_HOME=/workspace/.hf_cache"
echo "  export HF_HUB_ENABLE_HF_TRANSFER=1"
echo "  hf auth login            # or: export HF_TOKEN=hf_...   (gated licenses must be APPROVED)"
echo ""
echo "Then clone the harness and run the producer (see docs/SETUP_RUNPOD.md):"
echo "  cd /workspace && git clone https://github.com/oussndou77/alpamayo-faithfulness.git"
echo "  cd alpamayo-faithfulness && python runners/run_inference.py \\"
echo "      --clips 0 1 2 --k-rollouts 5 \\"
echo "      --clip-index /workspace/alpamayo/notebooks/clip_ids.parquet \\"
echo "      --out outputs/records.json --diag outputs/raw_diag.json"
