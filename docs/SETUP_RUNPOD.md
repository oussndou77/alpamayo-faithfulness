# Running Alpamayo inference on a GPU pod (Phase A)

The faithfulness axes run on CPU, but **producing** the CoC traces + trajectories needs
Alpamayo, which needs a GPU. This is the one GPU step; everything else is cold.

## Requirements
- A CUDA pod with **>=24 GB VRAM** (RTX 3090 / A100 / H100 — per the Alpamayo README).
- A Hugging Face account that has **accepted the licenses** for:
  - the model: `nvidia/Alpamayo-R1-10B` (non-commercial),
  - the dataset: `nvidia/PhysicalAI-Autonomous-Vehicles` (NVIDIA AV Dataset License).
- A HF access token exported as `HF_TOKEN` (never commit it).

## Steps
1. Spin up the pod, SSH in.
2. `bash runners/setup_runpod.sh` — clones NVlabs/alpamayo, installs deps, checks `torch.cuda`.
3. `export HF_TOKEN=...` (after accepting the licenses on the HF website).
4. Run the official `notebooks/inference.ipynb` **once** to confirm the model loads and
   produces a trajectory + a CoC trace (smoke test).
5. `python runners/run_inference.py --clips 774 --num-samples 5 --out outputs/records.json`
   - `--num-samples >= 2` is needed to exercise the stability axis.
6. Copy `outputs/records.json` off the pod. **Then shut the pod down** — all faithfulness
   scoring runs on your laptop:
   `python -c "from afh.runner import load_clip_records, evaluate_records; print(evaluate_records(load_clip_records('outputs/records.json')).format_table())"`

## Notes / gotchas (to be expanded during Phase A)
- The dataset is huge; start with the lightweight sample (`dgural/PhysicalAI-Autonomous-Vehicles-Sample`, 100 items) or a few clip indices.
- Scene object annotations (for the grounding axis) come from the dataset's cuboid / 2D-box
  labels; wiring them into `run_inference.py`'s `scene_objects` is a Phase-A task.
- Expect to harden this recipe on first contact, like we did for the CARLA pod.
