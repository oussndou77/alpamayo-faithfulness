"""
afh.runner — Orchestrate: ClipRecord(s) -> per-axis scores -> scorecard.

Pure-CPU: consumes ClipRecords (produced on a GPU pod by runners/run_inference.py and
serialized to JSON) and runs the faithfulness axes + scorecard. No model, no GPU.
"""

import json
from typing import List
from afh.trace import ClipRecord, CoCTrace, ParsedClaim, TrajectorySummary, SceneObject
from afh.scorecard import score_clip, DatasetScorecard


def load_clip_records(path: str) -> List[ClipRecord]:
    """Load ClipRecords from a JSON file produced by run_inference.py."""
    with open(path) as f:
        data = json.load(f)
    records = []
    for d in data:
        traces = [CoCTrace(clip_id=t["clip_id"], sample_index=t["sample_index"],
                           raw_text=t["raw_text"],
                           claims=[ParsedClaim(**c) for c in t.get("claims", [])])
                  for t in d.get("traces", [])]
        trajs = [TrajectorySummary(**ts) for ts in d.get("trajectories", [])]
        scene = [SceneObject(**o) for o in d.get("scene_objects", [])]
        records.append(ClipRecord(clip_id=d["clip_id"], traces=traces,
                                   trajectories=trajs, scene_objects=scene,
                                   min_ade=d.get("min_ade")))
    return records


def evaluate_records(records: List[ClipRecord]) -> DatasetScorecard:
    return DatasetScorecard(clip_scores=[score_clip(r) for r in records])


if __name__ == "__main__":
    import sys as _sys
    _path = _sys.argv[1] if len(_sys.argv) > 1 else "fixtures/real_alpamayo_sample.json"
    print(evaluate_records(load_clip_records(_path)).format_table())
