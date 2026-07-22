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

import json
import os
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
LATERAL_WORDS = ["nudge", "merge", "shift", "move over", "steer", "swerve", "change lane",
                 "change to the left", "change to the right"]
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
    if agent is not None:
        side = _agent_side_tail(t, agent)            # side word near the AGENT mention
        if side is None and action != "lateral_nudge":
            side = _match_vocab(t, SIDE_WORDS)       # fall back to any side word
    else:
        side = None   # no agent -> no agent_side (a curve's side is not an agent side;
                      # bug surfaced by the LLM-parser comparison on real traces)

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
    Phase B backend: LLM-as-parser with structured JSON output.

    Sends the raw CoC text to an LLM (Anthropic API) with the controlled vocabularies
    and few-shot examples drawn from real Alpamayo traces, and expects a JSON array of
    claims. Same output contract as the heuristic parser, so all axes stay backend-agnostic.

    Config (env):
      ANTHROPIC_API_KEY  — required to use this backend
      AFH_LLM_MODEL      — model id (default: claude-haiku-4-5-20251001, cheap + fast)

    Robustness: on any API/JSON/vocab failure the call falls back to the heuristic
    parser with a printed warning — the harness never crashes because of the LLM.
    """
    try:
        return _parse_llm_strict(raw_text)
    except Exception as e:
        print(f"[parser.llm] fallback to heuristic ({type(e).__name__}: {str(e)[:120]})")
        return _parse_heuristic(raw_text)


_LLM_SYSTEM = """You extract structured driving claims from an autonomous-driving model's \
Chain-of-Causation text. Answer ONLY with a JSON array, no prose, no markdown fences.

Each claim object has exactly these keys:
  action: one of "decelerate" | "accelerate" | "lateral_nudge" | "maintain" | "stop" | "other"
  action_polarity: for lateral_nudge only, "left" or "right"; else null
  causal_agent: one of "pedestrian" | "cyclist" | "vehicle" | "animal" | null (null if the \
cause is environmental like a curve, a red light, or lane positioning)
  agent_side: where the agent is relative to the ego: "left" | "ahead" | "right" | null
  cause: a short reason phrase copied or condensed from the text, or null

Rules:
- action is the EGO's maneuver, not the agent's motion.
- "keep distance" / "follow the lead vehicle" is car-following -> action "maintain".
- "change lanes to the left" / "nudge left" -> lateral_nudge with polarity "left".
- agent_side is the side where the AGENT is (e.g. "parked car on the right" -> "right"), \
not the direction of the ego's maneuver.
- One claim per distinct assertion; merge exact repetitions.

Examples:
Input: "Nudge left to increase clearance from the parked car on the right shoulder."
Output: [{"action":"lateral_nudge","action_polarity":"left","causal_agent":"vehicle",\
"agent_side":"right","cause":"increase clearance from the parked car"}]
Input: "Keep distance to the lead vehicle because it is directly ahead in the same lane."
Output: [{"action":"maintain","action_polarity":null,"causal_agent":"vehicle",\
"agent_side":"ahead","cause":"lead vehicle directly ahead"}]
Input: "Adapt speed for the left curve ahead."
Output: [{"action":"other","action_polarity":null,"causal_agent":null,\
"agent_side":null,"cause":"left curve ahead"}]"""

_ALLOWED = {
    "action": {"decelerate", "accelerate", "lateral_nudge", "maintain", "stop", "other"},
    "action_polarity": {"left", "right", None},
    "causal_agent": {"pedestrian", "cyclist", "vehicle", "animal", None},
    "agent_side": {"left", "ahead", "right", None},
}


def _call_anthropic(system: str, user: str, model: str, api_key: str) -> str:
    """Minimal Anthropic Messages API call (stdlib only; no new dependency)."""
    import urllib.request
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        data=json.dumps({
            "model": model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return "".join(b.get("text", "") for b in payload.get("content", [])
                   if b.get("type") == "text")


def _parse_llm_strict(raw_text: str) -> List[ParsedClaim]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("AFH_LLM_MODEL", "claude-haiku-4-5-20251001")

    text = raw_text.strip()
    if not text:
        return []
    out = _call_anthropic(_LLM_SYSTEM, text, model, api_key).strip()
    if out.startswith("```"):                      # strip accidental fences
        out = out.strip("`")
        out = out[out.find("["):]
    data = json.loads(out)
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON array")

    claims = []
    for c in data:
        action = c.get("action")
        if action not in _ALLOWED["action"]:
            raise ValueError(f"invalid action {action!r}")
        pol = c.get("action_polarity")
        if pol not in _ALLOWED["action_polarity"]:
            pol = None
        agent = c.get("causal_agent")
        if agent not in _ALLOWED["causal_agent"]:
            agent = None
        side = c.get("agent_side")
        if side not in _ALLOWED["agent_side"]:
            side = None
        claims.append(ParsedClaim(action=action, action_polarity=pol,
                                  cause=c.get("cause") or None,
                                  causal_agent=agent, agent_side=side, raw=text))
    return claims
