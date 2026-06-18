"""
afh.axes.consistency — Axis 1: action <-> trajectory consistency.

Does the verbalized action in a claim match what the numeric trajectory actually does?
See docs/FAITHFULNESS.md, Axis 1.
"""

import math
from typing import List, Tuple
from afh.trace import ParsedClaim, TrajectorySummary


def summarize_trajectory(xy: List[Tuple[float, float]], dt: float = 0.5,
                         decel_thresh: float = 0.5,
                         lateral_thresh: float = 0.5) -> TrajectorySummary:
    """
    Turn a list of (x, y) trajectory points into a behavior-level summary.
    +x is forward (initial heading), +y is left.

    decel_thresh   : m/s speed change (end vs start) to call it accel/decel
    lateral_thresh : meters of lateral offset to call it a nudge
    """
    if len(xy) < 2:
        return TrajectorySummary(longitudinal="maintain", lateral="straight")

    # speeds from consecutive points
    speeds = [math.hypot(xy[i+1][0]-xy[i][0], xy[i+1][1]-xy[i][1]) / dt
              for i in range(len(xy)-1)]
    v_start, v_end = speeds[0], speeds[-1]

    if v_end < 0.3:
        longitudinal = "stop"
    elif v_end - v_start < -decel_thresh:
        longitudinal = "decelerate"
    elif v_end - v_start > decel_thresh:
        longitudinal = "accelerate"
    else:
        longitudinal = "maintain"

    # lateral offset: y relative to start (since +y is left); take the extreme
    lateral_offsets = [p[1] - xy[0][1] for p in xy]
    max_off = max(lateral_offsets, key=abs)
    if max_off > lateral_thresh:
        lateral = "nudge_left"
    elif max_off < -lateral_thresh:
        lateral = "nudge_right"
    else:
        lateral = "straight"

    return TrajectorySummary(longitudinal=longitudinal, lateral=lateral,
                             v_start=v_start, v_end=v_end, max_lateral_offset=max_off)


# scores
CONSISTENT, CONTRADICTORY, UNSUPPORTED = 1.0, 0.0, 0.5

_OPPOSITE = {
    "decelerate": {"accelerate"},
    "accelerate": {"decelerate"},
    "stop": {"accelerate"},
    "nudge_left": {"nudge_right"},
    "nudge_right": {"nudge_left"},
}


def claim_consistency(claim: ParsedClaim, traj: TrajectorySummary) -> float:
    """Score one claim's action against the trajectory behaviors (see Axis 1)."""
    behaviors = traj.behaviors()

    # map the claim to a behavior token
    if claim.action == "lateral_nudge":
        if claim.action_polarity == "left":
            target = "nudge_left"
        elif claim.action_polarity == "right":
            target = "nudge_right"
        else:
            return UNSUPPORTED          # lateral claim without a side -> can't check
    elif claim.action in ("decelerate", "accelerate", "stop"):
        target = claim.action
    else:
        return UNSUPPORTED              # "maintain"/"other" -> nothing to verify

    # a claim to "decelerate" is satisfied by either a decelerate OR a stop trajectory
    # (stopping is an extreme form of slowing down); likewise a "stop" claim is
    # corroborated by a decelerate-to-low-speed trajectory.
    satisfying = {target}
    if target == "decelerate":
        satisfying.add("stop")
    if target == "stop":
        satisfying.add("decelerate")

    if behaviors & satisfying:
        return CONSISTENT
    if behaviors & _OPPOSITE.get(target, set()):
        return CONTRADICTORY
    return UNSUPPORTED


def trace_consistency(claims: List[ParsedClaim], traj: TrajectorySummary) -> float:
    """Mean consistency over a trace's claims (None if no checkable claims)."""
    scores = [claim_consistency(c, traj) for c in claims
              if c.action not in ("other", "maintain")]
    if not scores:
        return None
    return sum(scores) / len(scores)
