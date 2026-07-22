"""
afh.parser — Parse raw Chain-of-Causation text into structured ParsedClaim objects.

Input:  the raw string from Alpamayo's extra["cot"], e.g.
        "Nudge left to increase clearance from the cyclists on the right."
Output: one or more ParsedClaim (one per causal agent; environmental causes -> agent=None).

Two intended backends (configurable):
  - "heuristic": fast keyword/rule parser (implemented below as a first pass; no deps).
  - "llm": an LLM-as-judge with strict structured (JSON-schema) output for robust parsing
    — this is where the ARIA-style structured-extraction approach plugs in. TODO.

The controlled vocabularies live here so axes can rely on normalized values.
"""

from typing import List
from afh.trace import ParsedClaim, CoCTrace

# Controlled vocabularies (extend as we see real traces in Phase A).
AGENT_SYNONYMS = {
    "pedestrian": ["pedestrian", "person", "people", "walker", "jaywalker"],
    "cyclist": ["cyclist", "bicycle", "bike", "biker"],
    "vehicle": ["vehicle", "car", "truck", "bus", "van", "lead vehicle", "motorcycle"],
    "animal": ["animal", "dog", "deer"],
}
SIDE_WORDS = {"left": ["left"], "right": ["right"], "ahead": ["ahead", "front", "in front"]}
DECEL_WORDS = ["slow", "brake", "decelerate", "yield", "stop", "slow down"]
ACCEL_WORDS = ["accelerate", "speed up", "resume speed"]
# Car-following phrases: "keep distance to the lead vehicle" is NOT a braking command,
# it's a following behavior (the ego matches the lead and may speed up, hold, or slow).
# It is therefore not a checkable accel/decel assertion -> treat as "maintain" (unsupported),
# rather than forcing "decelerate" and producing a false contradiction against a trajectory
# that actually accelerates to follow. (A genuine "slow down"/"brake" still hits DECEL above.)
FOLLOW_WORDS = ["keep distance", "keep your distance", "keep a safe distance",
                "maintain distance", "maintain speed", "keep pace",
                "follow the lead", "following the lead"]
LATERAL_WORDS = ["nudge", "merge", "shift", "move over", "steer", "swerve", "change lane"]
ENV_CAUSES = ["red light", "traffic light", "stop sign", "crosswalk", "green light", "weather"]


def _match_vocab(text: str, vocab: dict):
    t = text.lower()
    for canonical, words in vocab.items():
        if any(w in t for w in words):
            return canonical
    return None


def parse_trace(clip_id: str, sample_index: int, raw_text: str,
                backend: str = "heuristic") -> CoCTrace:
    """Parse one raw CoC string into a CoCTrace with structured claims."""
    if backend == "heuristic":
        claims = _parse_heuristic(raw_text)
    elif backend == "llm":
        claims = _parse_llm(raw_text)  # TODO: structured LLM-as-judge
    else:
        raise ValueError(f"unknown parser backend: {backend}")
    return CoCTrace(clip_id=clip_id, sample_index=sample_index,
                    raw_text=raw_text, claims=claims)


def _parse_heuristic(raw_text: str) -> List[ParsedClaim]:
    """
    Minimal first-pass parser: split into clauses and pull out action / agent / side.
    Deliberately simple and transparent; we'll refine it against real Phase-A traces,
    and/or switch to the LLM backend for robustness.
    """
    text = raw_text.strip()
    if not text:
        return []

    t = text.lower()

    # ── action + polarity ──
    # The ego's action verb takes priority over an agent's state. We detect the action,
    # and for lateral actions take the polarity from the side word NEAREST the action
    # verb (so "nudge right ... cyclist on the left" is a right nudge, not left).
    action, polarity = "other", None
    if any(w in t for w in LATERAL_WORDS):
        action = "lateral_nudge"
        polarity = _polarity_near_action(t)
    elif any(w in t for w in DECEL_WORDS):
        # "slow down", "brake", "keep distance", "yield" -> decelerate (ego action).
        # Note: deliberately checked before a bare "stop" so "slowing/stopping" agent
        # states don't override the ego's own "slow down".
        action = "decelerate"
    elif "stop" in t:
        action = "stop"
    elif any(w in t for w in ACCEL_WORDS):
        action = "accelerate"
    elif any(w in t for w in FOLLOW_WORDS):
        # car-following ("keep distance to the lead vehicle") -> maintain: a following
        # claim doesn't assert a specific accel/decel, so it's left unchecked rather than
        # scored as a contradiction. Placed after DECEL so "keep distance AND slow down"
        # still reads as a genuine deceleration.
        action = "maintain"

    # ── causal agent + side ──
    agent = _match_vocab(t, AGENT_SYNONYMS)
    side = _agent_side_tail(t, agent)            # side word near the AGENT mention
    if side is None and action != "lateral_nudge":
        side = _match_vocab(t, SIDE_WORDS)       # fall back to any side word

    cause = _match_vocab(t, {c: [c] for c in ENV_CAUSES}) or None

    return [ParsedClaim(action=action, action_polarity=polarity, cause=cause,
                        causal_agent=agent, agent_side=side, raw=text)]


def _polarity_near_action(text: str) -> str:
    """
    For a lateral action, pick the side word closest to a lateral action verb,
    so the ego's own direction wins over an agent's side ('nudge right ... on the left').
    """
    best_side, best_dist = None, 10**9
    for verb in LATERAL_WORDS:
        vi = text.find(verb)
        if vi < 0:
            continue
        for side, words in {"left": ["left"], "right": ["right"]}.items():
            for w in words:
                wi = text.find(w)
                if wi >= 0 and abs(wi - vi) < best_dist:
                    best_dist, best_side = abs(wi - vi), side
    return best_side


def _agent_side_tail(text: str, agent) -> str:
    """Look for a side word occurring near/after the agent mention."""
    if not agent:
        return None
    for word in AGENT_SYNONYMS.get(agent, []):
        idx = text.find(word)
        if idx >= 0:
            tail = text[idx:]
            for side, words in SIDE_WORDS.items():
                if any(w in tail for w in words):
                    return side
    return None


def _parse_llm(raw_text: str) -> List[ParsedClaim]:
    """
    TODO (Phase B): LLM-as-judge with strict JSON-schema output, mirroring the
    structured-extraction approach used in ARIA. Should return the same ParsedClaim
    shape so axes are backend-agnostic. Kept out of the default path so the harness
    has zero network/credential dependency for cold testing.
    """
    raise NotImplementedError("LLM parser backend not implemented yet (Phase B).")
