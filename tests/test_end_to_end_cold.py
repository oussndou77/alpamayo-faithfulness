#!/usr/bin/env python3
"""
Cold end-to-end test of the faithfulness harness against fixtures — NO GPU, NO model.

Builds ClipRecords from fixtures/sample_traces.json (raw CoC text + trajectory + scene),
runs the parser + axes 1-3 + scorecard, and asserts the expected faithfulness signals:
  - the "faithful_brake" clip scores high,
  - the "hallucinated_cyclist" clip is caught by grounding,
  - the "contradictory_action" clip is caught by consistency,
  - the "unstable" clip raises the contradiction flag.

Run from repo root:  python tests/test_end_to_end_cold.py
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from afh.parser import parse_trace
from afh.axes.consistency import summarize_trajectory
from afh.trace import ClipRecord, SceneObject
from afh.scorecard import score_clip, DatasetScorecard

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_traces.json")


def build_records():
    with open(FIX) as f:
        data = json.load(f)
    records = []
    for clip in data["clips"]:
        cid = clip["clip_id"]
        traces = [parse_trace(cid, i, raw) for i, raw in enumerate(clip["raw_traces"])]
        # one trajectory summary, reused for each sample (fixtures share a trajectory)
        traj = summarize_trajectory([tuple(p) for p in clip["trajectory_xy"]], dt=0.5)
        trajs = [traj for _ in traces]
        scene = [SceneObject(**o) for o in clip["scene_objects"]]
        records.append(ClipRecord(clip_id=cid, traces=traces, trajectories=trajs,
                                   scene_objects=scene, min_ade=clip.get("min_ade")))
    return records


def test_parser_extracts_claims():
    recs = {r.clip_id: r for r in build_records()}
    # the cyclist clip should parse a lateral_nudge-left with a cyclist agent on the right
    claim = recs["fixture_hallucinated_cyclist"].traces[0].claims[0]
    assert claim.action == "lateral_nudge"
    assert claim.action_polarity == "left"
    assert claim.causal_agent == "cyclist"
    assert claim.agent_side == "right"


def test_faithful_clip_scores_high():
    recs = {r.clip_id: r for r in build_records()}
    s = score_clip(recs["fixture_faithful_brake"])
    # decelerates as claimed (consistency high) and the vehicle is really ahead (grounded)
    assert s.consistency == 1.0, f"expected consistent deceleration, got {s.consistency}"
    f = s.faithfulness()
    assert f is not None and f >= 0.8, f"faithful clip should score high, got {f}"


def test_grounding_catches_hallucination():
    recs = {r.clip_id: r for r in build_records()}
    s = score_clip(recs["fixture_hallucinated_cyclist"])
    # no cyclist in the scene -> grounding should be 0
    assert s.grounding == 0.0, f"hallucinated agent should score 0 grounding, got {s.grounding}"


def test_consistency_catches_contradiction():
    recs = {r.clip_id: r for r in build_records()}
    s = score_clip(recs["fixture_contradictory_action"])
    # says slow down but accelerates -> consistency 0
    assert s.consistency == 0.0, f"contradictory action should score 0 consistency, got {s.consistency}"


def test_stability_flags_unstable_clip():
    recs = {r.clip_id: r for r in build_records()}
    s = score_clip(recs["fixture_unstable"])
    # opposite lateral polarities across samples -> contradiction flag
    assert s.contradiction is True, "opposite-polarity samples should raise contradiction"


def test_scorecard_builds_and_orders():
    sc = DatasetScorecard(clip_scores=[score_clip(r) for r in build_records()])
    summ = sc.summary()
    assert summ["n_clips"] == 4
    # the faithful clip should out-score the hallucinated one
    by_id = {s.clip_id: s for s in sc.clip_scores}
    assert (by_id["fixture_faithful_brake"].faithfulness()
            > by_id["fixture_hallucinated_cyclist"].faithfulness())


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  ok  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  XX  {t.__name__} - FAILED: {e}")
        except Exception as e:
            print(f"  XX  {t.__name__} - ERROR: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=== Cold end-to-end faithfulness harness test (no GPU) ===")
    ok = _run_all()
    print()
    if ok:
        print("Sample scorecard:")
        sc = DatasetScorecard(clip_scores=[score_clip(r) for r in build_records()])
        print(sc.format_table())
    sys.exit(0 if ok else 1)
