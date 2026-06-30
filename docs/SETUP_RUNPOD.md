# Running Alpamayo inference on a GPU pod (Phase A)

The faithfulness axes run on CPU, but **producing** the CoC traces + trajectories needs
Alpamayo, which needs a GPU. This is the one GPU step; everything else is cold.

This recipe was validated on RunPod in June 2026.

## Requirements

- A CUDA pod with **~48 GB VRAM** (L40S or A100). 24 GB is Alpamayo's stated minimum, but
  this release does **not** support `sdpa` and `flash_attention_2` needs building, so we run
  the **`eager`** attention path — which OOMs at 24 GB on the VLM rollout. Use 48 GB.
- Image: a recent **PyTorch + CUDA 12.x** template (e.g. `runpod/pytorch:...cu1281-torch280-ubuntu2404`).
- A **persistent Network Volume** mounted at `/workspace` (so the 22 GB of weights and the
  venv survive a stop/restart).
- A Hugging Face account with **APPROVED access** (not merely requested) to:
  - the model: `nvidia/Alpamayo-R1-10B` (non-commercial license),
  - the dataset: `nvidia/PhysicalAI-Autonomous-Vehicles` (NVIDIA AV Dataset License).
  Approval can take **1–2 business days**; an account alone is not access.
- A HF access token (Read), exported as `HF_TOKEN`. **Never commit it.**

## Steps

1. Spin up the pod (48 GB GPU, PyTorch image, ~60 GB disk, Network Volume on `/workspace`), SSH in.

2. Run the setup script (clones Alpamayo, installs uv + a Python 3.12 venv, syncs deps
   without flash-attn, installs `hf_transfer`, checks `torch.cuda`):
   ```bash
   cd /workspace
   git clone https://github.com/oussndou77/alpamayo-faithfulness.git
   bash alpamayo-faithfulness/runners/setup_runpod.sh
   ```

3. Activate the env in your shell and authenticate (do this again after any reconnect):
   ```bash
   cd /workspace/alpamayo
   source ar1_venv/bin/activate
   export PATH="$HOME/.local/bin:$PATH"
   export HF_HOME=/workspace/.hf_cache
   export HF_HUB_ENABLE_HF_TRANSFER=1
   hf auth login            # or: export HF_TOKEN=hf_...
   hf auth whoami           # should print your username
   ```

4. (Optional) Smoke test the model once. The official `src/alpamayo_r1/test_inference.py`
   defaults to `flash_attention_2`; to run it without building flash-attn, copy it and set
   `attn_implementation="eager"` in the `from_pretrained(...)` call. You should see a
   trajectory + a Chain-of-Causation trace, and a `minADE`.

5. Produce ClipRecords with the harness's producer (run it from the harness repo, with the
   Alpamayo venv active so `alpamayo_r1` is importable):
   ```bash
   cd /workspace/alpamayo-faithfulness
   python runners/run_inference.py \
       --clips 0 1 2 --k-rollouts 5 \
       --clip-index /workspace/alpamayo/notebooks/clip_ids.parquet \
       --out outputs/records.json --diag outputs/raw_diag.json
   ```
   - `--k-rollouts >= 2` runs **independent** rollouts (different seeds) so the stability
     axis is meaningful. (A single `num_traj_samples=K` call shares one reasoning trace
     across K trajectories — it cannot test reasoning stability.)
   - `--diag` dumps raw early trajectory points + per-rollout ADE for frame sanity checks.

6. Score it (on the pod or, after copying `outputs/` off, on your laptop — pure CPU):
   ```bash
   python -c "from afh.runner import load_clip_records, evaluate_records; \
   print(evaluate_records(load_clip_records('outputs/records.json')).format_table())"
   ```

7. **Stop the pod** when done (weights persist in `/workspace/.hf_cache` for next time).

## Notes / gotchas (learned on a real pod)

- **Gated access**: accept **both** licenses (model **and** dataset) and wait for the
  approval emails. "I have an account" is not "I have access".
- **Attention**: this model rejects `sdpa`; `flash_attention_2` requires building flash-attn.
  Use `attn_implementation="eager"` (the default in `run_inference.py`). eager needs ~48 GB.
- **hf_transfer**: the RunPod image sets `HF_HUB_ENABLE_HF_TRANSFER=1`, but the package
  isn't installed by default — `uv pip install hf_transfer` (or `export HF_HUB_ENABLE_HF_TRANSFER=0`).
- **Token resolution**: if you customise `HF_HOME`, exporting `HF_TOKEN` explicitly is the
  robust way to make the dataset loader authenticate.
- **Persistence**: keep work under `/workspace` (the Network Volume). The 22 GB weights
  cached in `/workspace/.hf_cache` then survive restarts (no re-download).
- **Dropped sessions**: SSH terminals drop often — run long jobs in `tmux`
  (`tmux new -s alpa`, reattach with `tmux attach -t alpa`).
- **Don't paste large scripts** into the pod terminal (it corrupts indentation/long lines);
  that's why the producer lives in this repo — `git clone` it instead.
- **Trajectory units**: trajectories are 64 waypoints @ 10 Hz (6.4 s), so `dt=0.1`. Lateral
  maneuvers are read from an early ~2 s window, because road curvature dominates the lateral
  signal over the full horizon.

## Still TODO (axis 2 — grounding)

Scene-object annotations (cuboids / 2D boxes) are **not** returned by the inference loader;
they live in the dataset's `labels/` features (`avdi.get_clip_feature(clip_id,
avdi.features.LABELS....)`). Wiring those into `run_inference.py`'s `scene_objects`
(object_type + 3D location in the ego frame at t0) is the next Phase-A task; until then the
grounding axis reports `n/a`.
