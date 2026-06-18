# Alpamayo-Faithfulness

**A faithfulness evaluation harness for the reasoning traces of NVIDIA Alpamayo** — the open reasoning Vision-Language-Action (VLA) models for autonomous driving.

Alpamayo doesn't just predict a driving trajectory; it produces a **Chain-of-Causation (CoC)** reasoning trace that *explains* the decision in natural language (e.g. *"Nudge left to increase clearance from the cyclists on the right"*). This explainability is the model's headline feature — and the basis for trust, debugging, and safety/regulatory acceptance of reasoning-based autonomy.

But a stated reason is only useful if it is **faithful**: if the cause the model verbalizes actually drives the trajectory it produces. A model that says *"I slowed because of the pedestrian"* while its trajectory ignores the pedestrian is producing a plausible **post-hoc rationalization**, not an explanation — and that is arguably more dangerous than no explanation at all.

**This repo measures that faithfulness.** It is an evaluation harness, not a model: it runs Alpamayo, captures its CoC traces and trajectories, and scores how well the reasoning corresponds to the behavior.

> Status: early. The data contract and the cold-testable axes (1–3) come first; the model-in-the-loop and counterfactual axes follow. See `docs/FAITHFULNESS.md` for the formal definitions and the roadmap below for progress.

## Why this, and why now

NVIDIA's open AV stack (Alpamayo, AlpaSim, AlpaGym, Cosmos) and **Alpamayo Recipes** already cover how to *train, fine-tune, distill, and RL-post-train* these models. **None of them evaluate whether the reasoning is honest.** Faithfulness of chain-of-thought is a known open problem in LLMs; for a driving VLA, where the reasoning is a safety artifact, it is unaddressed and high-stakes. This harness targets exactly that gap.

It is also deliberately **light on compute**: the hard part is the evaluation logic, not training. Alpamayo-R1-10B inference runs on a single 24 GB GPU (RTX 3090 / A100 / H100), and the parser + faithfulness axes are pure Python that runs on a laptop against recorded traces.

## The four faithfulness axes

A CoC trace is an explicit causal claim: **[action] because [cause involving agent X]**. We test whether that claim holds along four axes (formal definitions in `docs/FAITHFULNESS.md`):

1. **Action–trajectory consistency** — does the verbalized action ("slow down", "nudge left") match what the numeric trajectory actually does?
2. **Causal-agent grounding** — does the agent the model blames ("the cyclist on the right") actually exist in the scene, in roughly that position? (Uses the dataset's 3D/2D object annotations as ground truth — an anti-hallucination check.)
3. **Sampling stability** — Alpamayo decodes stochastically (temperature > 0). Across repeated samples of the *same* scene, are the stated causes consistent, or does the model contradict itself?
4. **Counterfactual sensitivity** *(advanced)* — if we remove or alter the agent the model says it reacted to, do the trajectory **and** the reasoning change coherently? If removing the cyclist doesn't change the "nudge left", the reason was unfaithful.

Axes 1–3 are computable without modifying the visual input (cold-testable on recorded traces). Axis 4 requires scene manipulation and comes later (potentially via Cosmos-generated counterfactual scenes).

## Repository layout

```
afh/                  # the package (Alpamayo FaitHfulness)
  trace.py            # data contract: CoCTrace, ParsedClaim, TrajectorySummary
  parser.py           # parse raw CoC text -> structured (action, cause, agent) claims
  axes/
    consistency.py    # axis 1: action <-> trajectory
    grounding.py      # axis 2: claimed agent <-> scene annotations
    stability.py      # axis 3: agreement across samples
  scorecard.py        # aggregate per-clip + dataset-level faithfulness scorecard
  runner.py           # orchestration: clip -> (inference) -> axes -> score
runners/
  setup_runpod.sh     # one-shot environment setup for an Alpamayo GPU pod
  run_inference.py    # run Alpamayo on clips, save traces + trajectories to disk
tests/                # cold tests (no GPU) against fixture traces
fixtures/             # example CoC traces + scene annotations for offline testing
docs/
  FAITHFULNESS.md     # formal definition of the four axes
  SETUP_RUNPOD.md     # GPU pod recipe for running Alpamayo inference
```

The design separates **GPU work** (running Alpamayo, in `runners/`) from **CPU work** (parsing + scoring, in `afh/`). Roughly 80% of the harness — the parser and axes 1–3 — is built and tested cold against fixtures, with no GPU, then exercised on real model output in a focused pod session.

## Roadmap

- [ ] **Phase A** — Stand up Alpamayo-R1-10B inference on a GPU pod; capture real CoC traces + trajectories on the PhysicalAI-AV sample.
- [ ] **Phase B** — CoC trace parser: raw text → structured `(action, cause, causal_agent)` claims.
- [ ] **Phase C** — Implement faithfulness axes 1–3 (cold-testable).
- [ ] **Phase D** — Aggregate into a per-clip and dataset-level faithfulness scorecard.
- [ ] **Phase E** — Counterfactual axis (axis 4), scene manipulation.

## License & data

The **code** in this repository is released under the Apache 2.0 license (see `LICENSE`).

This project **does not redistribute** Alpamayo model weights or the NVIDIA PhysicalAI-AV dataset. Alpamayo weights are under NVIDIA's **non-commercial** license; the PhysicalAI-AV dataset is under the NVIDIA AV Dataset License Agreement. Obtain them from NVIDIA / Hugging Face under their respective terms. This harness is intended for **research, experimentation, and evaluation**.

## Acknowledgements

Built on NVIDIA's open release of [Alpamayo](https://github.com/NVlabs/alpamayo) and the PhysicalAI-Autonomous-Vehicles dataset.
