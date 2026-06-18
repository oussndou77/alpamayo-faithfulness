"""
afh.trace — Data contract for the faithfulness harness.

These are the stable structures every other module agrees on. Keeping them small and
explicit (like RCIB's trace contract) means the parser, the axes, and the scorecard can
be developed and tested independently.

Units: positions in meters, time in seconds, speeds in m/s. The ego's initial heading
defines +x (forward); +y is left.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple


# ──────────────────────────────────────────────────────────────
# Raw model output
# ──────────────────────────────────────────────────────────────

@dataclass
class TrajectorySummary:
    """
    Compact, behavior-level summary of a predicted trajectory τ.
    Computed from the raw (x, y) points by afh.axes.consistency, but stored here so the
    axes and scorecard share one representation.
    """
    # observable behaviors derived from the trajectory
    longitudinal: str            # "decelerate" | "accelerate" | "maintain" | "stop"
    lateral: str                 # "nudge_left" | "nudge_right" | "straight"
    # supporting numbers (for transparency / debugging)
    v_start: float = 0.0         # m/s at the first step
    v_end: float = 0.0           # m/s at the last step
    max_lateral_offset: float = 0.0   # signed meters (+left, -right)

    def behaviors(self) -> set:
        """The set of behaviors this trajectory exhibits (excludes neutral states)."""
        out = set()
        if self.longitudinal in ("decelerate", "accelerate", "stop"):
            out.add(self.longitudinal)
        if self.lateral in ("nudge_left", "nudge_right"):
            out.add(self.lateral)
        return out


@dataclass
class SceneObject:
    """One annotated object in the scene (from the dataset's cuboid / 2D-box labels)."""
    object_type: str             # e.g. "car", "pedestrian", "cyclist"
    x: float                     # 3D location relative to ego, +x forward (m)
    y: float                     # +y left (m)
    object_id: Optional[int] = None
    box_2d: Optional[Dict[str, Tuple[float, float, float, float]]] = None  # per-camera

    def side(self) -> str:
        """Coarse side of this object relative to the ego's path: left / ahead / right."""
        # +y is left; a small dead-zone around the centerline counts as "ahead".
        if self.y > 1.5:
            return "left"
        if self.y < -1.5:
            return "right"
        return "ahead"


# ──────────────────────────────────────────────────────────────
# Parsed reasoning
# ──────────────────────────────────────────────────────────────

@dataclass
class ParsedClaim:
    """
    One structured causal claim extracted from a CoC trace:
        [action / polarity] because [cause involving causal_agent on agent_side]
    Fields are normalized to small controlled vocabularies (see afh.parser).
    """
    action: str                  # "decelerate" | "accelerate" | "lateral_nudge" | "maintain" | "stop" | "other"
    action_polarity: Optional[str] = None   # for lateral: "left" | "right"; else None
    cause: Optional[str] = None              # free-ish reason phrase, normalized
    causal_agent: Optional[str] = None       # "cyclist" | "pedestrian" | "vehicle" | ... | None (environmental)
    agent_side: Optional[str] = None         # "left" | "ahead" | "right" | None
    raw: str = ""                            # the source span, for traceability

    def is_environmental(self) -> bool:
        """A claim with no causal agent (e.g. 'red light') — excluded from grounding."""
        return self.causal_agent is None


@dataclass
class CoCTrace:
    """A single Chain-of-Causation trace and everything parsed from it."""
    clip_id: str
    sample_index: int            # which sample (0..N-1) when num_traj_samples > 1
    raw_text: str                # the original string from extra["cot"]
    claims: List[ParsedClaim] = field(default_factory=list)

    def dominant_agent(self) -> Optional[str]:
        """The most-cited causal agent in this trace (for stability comparison)."""
        agents = [c.causal_agent for c in self.claims if c.causal_agent]
        if not agents:
            return None
        # most common
        return max(set(agents), key=agents.count)

    def dominant_action(self) -> Optional[Tuple[str, Optional[str]]]:
        acts = [(c.action, c.action_polarity) for c in self.claims
                if c.action not in ("other", "maintain")]
        if not acts:
            return None
        return max(set(acts), key=acts.count)


# ──────────────────────────────────────────────────────────────
# A full per-clip record (raw output + ground truth, ready for scoring)
# ──────────────────────────────────────────────────────────────

@dataclass
class ClipRecord:
    """
    Everything the harness needs to score one clip. Produced by runners/run_inference.py
    on a GPU pod, serialized to JSON, then consumed cold by the axes + scorecard.
    """
    clip_id: str
    traces: List[CoCTrace] = field(default_factory=list)         # N samples
    trajectories: List[TrajectorySummary] = field(default_factory=list)  # N, aligned with traces
    scene_objects: List[SceneObject] = field(default_factory=list)
    min_ade: Optional[float] = None     # accuracy vs logged future (reported, never mixed into faithfulness)

    def to_dict(self) -> dict:
        return asdict(self)
