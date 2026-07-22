#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
compare_parsers.py — Phase B evaluation: heuristic vs LLM parser on real Alpamayo traces.

Re-parses every trace of a real fixture with both backends, reports field-level agreement,
prints all disagreements, and shows the scorecard under each backend.

Needs ANTHROPIC_API_KEY (the LLM side). Cost is tiny: one Haiku call per unique trace.

Usage:
    python runners/compare_parsers.py [fixtures/real_alpamayo_labeled_sample.json]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from afh.parser import parse_trace                       # noqa: E402
from afh.runner import load_clip_records, evaluate_records  # noqa: E402

FIELDS = ("action", "action_polarity", "causal_agent", "agent_side")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/real_alpamayo_labeled_sample.json"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the LLM side of the comparison.")

    records_h = load_clip_records(path)
    records_l = load_clip_records(path)

    # cache: identical raw texts -> one LLM call
    llm_cache = {}
    n_traces = agree_all = 0
    field_agree = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    disagreements = []

    for rh, rl in zip(records_h, records_l):
        for th, tl in zip(rh.traces, rl.traces):
            th.claims = parse_trace(th.clip_id, th.sample_index, th.raw_text,
                                    backend="heuristic").claims
            key = th.raw_text.strip()
            if key not in llm_cache:
                llm_cache[key] = parse_trace(tl.clip_id, tl.sample_index, tl.raw_text,
                                             backend="llm").claims
            tl.claims = llm_cache[key]

            n_traces += 1
            ch = th.claims[0] if th.claims else None
            cl = tl.claims[0] if tl.claims else None
            if ch is None or cl is None:
                continue
            same = True
            diffs = []
            for f in FIELDS:
                field_total[f] += 1
                vh, vl = getattr(ch, f), getattr(cl, f)
                if vh == vl:
                    field_agree[f] += 1
                else:
                    same = False
                    diffs.append(f"{f}: heur={vh!r} llm={vl!r}")
            if same:
                agree_all += 1
            else:
                disagreements.append((th.clip_id[:13], th.sample_index,
                                      th.raw_text.split(chr(10))[0][:70], diffs))

    print(f"\n===== parser agreement on {n_traces} real traces "
          f"({len(llm_cache)} unique -> LLM calls) =====")
    for f in FIELDS:
        t = field_total[f] or 1
        print(f"  {f:<16} {100.0 * field_agree[f] / t:5.1f} %")
    print(f"  {'ALL FIELDS':<16} {100.0 * agree_all / max(n_traces, 1):5.1f} %")

    if disagreements:
        print(f"\n===== {len(disagreements)} disagreements =====")
        for clip, k, txt, diffs in disagreements:
            print(f"  [{clip} r{k}] \"{txt}\"")
            for d in diffs:
                print(f"      {d}")

    print("\n===== scorecard: HEURISTIC claims =====")
    print(evaluate_records(records_h).format_table())
    print("\n===== scorecard: LLM claims =====")
    print(evaluate_records(records_l).format_table())


if __name__ == "__main__":
    main()
