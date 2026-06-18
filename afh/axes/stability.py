"""
afh.axes.stability — Axis 3: sampling stability across repeated samples of one clip.

Alpamayo decodes stochastically; across N samples of the SAME scene, is the stated
cause stable, or does the model contradict itself? See docs/FAITHFULNESS.md, Axis 3.
"""

from typing import List, Dict
from afh.trace import CoCTrace

_OPPOSITE_POLARITY = {("lateral_nudge", "left"): ("lateral_nudge", "right"),
                      ("lateral_nudge", "right"): ("lateral_nudge", "left")}


def _modal_agreement(items: List) -> float:
    """Fraction of items equal to the most common non-None item."""
    vals = [i for i in items if i is not None]
    if not vals:
        return None
    modal = max(set(vals), key=vals.count)
    return vals.count(modal) / len(vals)


def stability(traces: List[CoCTrace]) -> Dict:
    """
    Compute stability over the N traces of one clip.
    Returns a dict with agent_agreement, action_agreement, a combined score, and a
    contradiction flag (opposite lateral polarities asserted for the same scene).
    """
    if len(traces) < 2:
        return {"agent_agreement": None, "action_agreement": None,
                "score": None, "contradiction": False, "n_samples": len(traces)}

    agents = [t.dominant_agent() for t in traces]
    actions = [t.dominant_action() for t in traces]

    agent_ag = _modal_agreement(agents)
    action_ag = _modal_agreement(actions)

    # contradiction: any pair asserting opposite polarities
    action_set = {a for a in actions if a is not None}
    contradiction = any(_OPPOSITE_POLARITY.get(a) in action_set for a in action_set)

    parts = [p for p in (agent_ag, action_ag) if p is not None]
    score = sum(parts) / len(parts) if parts else None

    return {"agent_agreement": agent_ag, "action_agreement": action_ag,
            "score": score, "contradiction": contradiction, "n_samples": len(traces)}
