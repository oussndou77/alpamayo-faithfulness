"""
afh.axes.grounding — Axis 2: causal-agent grounding (anti-hallucination).

Does the agent the model blames actually exist in the scene, on the claimed side?
Uses the dataset's object annotations as ground truth. See docs/FAITHFULNESS.md, Axis 2.
"""

from typing import List, Optional
from afh.trace import ParsedClaim, SceneObject
from afh.parser import AGENT_SYNONYMS

GROUNDED_FULL, GROUNDED_CLASS_ONLY, HALLUCINATED = 1.0, 0.5, 0.0


def _class_matches(agent: str, obj_type: str) -> bool:
    """Does a scene object's type match the claimed agent (with synonyms)?"""
    words = AGENT_SYNONYMS.get(agent, [agent])
    o = obj_type.lower()
    return any(w in o or o in w for w in words)


def claim_grounding(claim: ParsedClaim, scene: List[SceneObject]) -> Optional[float]:
    """
    Score one claim's causal agent against scene objects.
    Returns None for environmental claims (no agent) — excluded from this axis.
    """
    if claim.is_environmental() or not claim.causal_agent:
        return None

    if not scene:
        return None      # no scene annotations available (axis 2 not wired yet) -> not evaluable,
                         # NOT a hallucination; returning 0.0 here would falsely tank the score

    matches = [o for o in scene if _class_matches(claim.causal_agent, o.object_type)]
    if not matches:
        return HALLUCINATED                 # named an agent that isn't in the scene

    if claim.agent_side is None:
        return GROUNDED_FULL                # class matches, no side asserted to check

    if any(o.side() == claim.agent_side for o in matches):
        return GROUNDED_FULL                # class AND side match
    return GROUNDED_CLASS_ONLY              # right kind of agent, wrong side


def trace_grounding(claims: List[ParsedClaim], scene: List[SceneObject]) -> Optional[float]:
    """Mean grounding over agent-bearing claims (None if the trace names no agents)."""
    scores = [s for s in (claim_grounding(c, scene) for c in claims) if s is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)
