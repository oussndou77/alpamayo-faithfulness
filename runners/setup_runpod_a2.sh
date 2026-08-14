#!/usr/bin/env bash
# setup_runpod_a2.sh — set up NVIDIA Alpamayo 2 Super (34B) inference on a GPU pod.
#
# GPU: 1x H100 80GB (validated by NVIDIA: ~72 GiB peak) or A100 80GB (cheaper, slower).
#      A 48 GB pod is NOT enough for the 2 Super.
# Disk: network volume >= 150 GB (weights ~68 GB bf16 + HF cache + dataset chunks).
#
# Official flow (from NVlabs/alpamayo2 README): uv sync --locked --dev in-repo venv.
# Run from /workspace so everything lands on the persistent Network Volume.
set -e

echo "=== Alpamayo-Faithfulness :: Alpamayo 2 Super pod setup ==="

# 0) Caches on the persistent volume (do this BEFORE install so nothing lands in ~)
export HF_HOME=/workspace/.hf_cache
export UV_CACHE_DIR=/workspace/.uv_cache
export MPLCONFIGDIR=/tmp/alpamayo2_super_mpl
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1) Clone the official Alpamayo 2 repo (inference code, Apache 2.0)
cd /workspace
if [ ! -d alpamayo2 ]; then
  git clone https://github.com/NVlabs/alpamayo2.git
fi
cd alpamayo2

# 2) uv + the repo-managed venv (the README's exact flow)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --dev
# shellcheck disable=SC1091
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

# 3) HF auth — the model is gated; access must be APPROVED on
#    https://huggingface.co/nvidia/Alpamayo2-Super before the download will work.
#    Export HF_TOKEN before running this script, or run `hf auth login` interactively.
if [ -n "$HF_TOKEN" ]; then
  hf auth login --token "$HF_TOKEN"
else
  echo ">>> No HF_TOKEN in env; run 'hf auth login' manually before inference."
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ""
echo "=== setup done (verify 'cuda True' + an 80GB device above) ==="
echo "If you reconnect, re-run in the new shell:"
echo "  cd /workspace/alpamayo2"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  export HF_HOME=/workspace/.hf_cache UV_CACHE_DIR=/workspace/.uv_cache"
echo "  export MPLCONFIGDIR=/tmp/alpamayo2_super_mpl HF_HUB_ENABLE_HF_TRANSFER=1"
echo "  source .venv/bin/activate"
echo ""
echo "Then clone the harness and run the PROBE FIRST (never the full run cold):"
echo "  cd /workspace && git clone https://github.com/oussndou77/alpamayo-faithfulness.git"
echo "  cd alpamayo-faithfulness && python runners/probe_a2.py"
