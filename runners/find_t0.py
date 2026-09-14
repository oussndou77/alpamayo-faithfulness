#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
find_t0.py — for each Phase-G clip, find the t0 (microseconds) where the cited object is
MOST blocking, so the GPU audit spends no rollout searching for the right moment.

CPU-only: streams obstacle labels, interpolates each track, and scores "blocking-ness"
over a grid of candidate t0 values. Blocking-ness = close (small x) AND centered (small
|y|) AND in the actionable range. Writes a small JSON the audit runner consumes.

Usage:
    python runners/find_t0.py \
        --clips 00097de1-... 0026c1ce-... 0e3771c0-... \
        --out outputs/phase_g_t0.json

    # or from the candidates CSV (top-N by score):
    python runners/find_t0.py \
        --from-csv outputs/phase_g_nurec_candidates.csv --top 5 \
        --out outputs/phase_g_t0.json
"""

import argparse
import json
import os

X_MIN, X_MAX = 5.0, 35.0
LANE_HALF_WIDTH_M = 1.8


def blocking_score(x, y, sx, sy):
    if not (X_MIN < x < X_MAX and abs(y) < LANE_HALF_WIDTH_M):
        return 0.0
    closeness = max(0.0, 1.0 - (x - X_MIN) / (X_MAX - X_MIN))
    centered = max(0.0, 1.0 - abs(y) / LANE_HALF_WIDTH_M)
    size = min(1.0, (sx * sy) / 8.0)
    return 0.45 * centered + 0.35 * closeness + 0.20 * size


def _clips_from_csv(path, top):
    import pandas as pd
    rows = []
    # the CSV has a column-shift bug on some rows (nurec token sometimes first);
    # recover clip_id as the field that looks like a UUID, score as the float field.
    import csv as _csv
    with open(path) as fh:
        for r in _csv.reader(fh):
            if not r or r[0] == "clip_id":
                continue
            uuid = next((c for c in r if len(c) == 36 and c.count("-") == 4), None)
            if not uuid:
                continue
            # score is the float in (0, 1]; sizes/coords are larger or can be negative
            score = 0.0
            for c in r:
                try:
                    v = float(c)
                except ValueError:
                    continue
                if 0.0 < v <= 1.0 and "." in c:
                    score = v
                    break
            rows.append((uuid, score))
    rows.sort(key=lambda t: t[1], reverse=True)
    seen, out = set(), []
    for uuid, sc in rows:
        if uuid not in seen and sc > 0:
            seen.add(uuid); out.append(uuid)
        if len(out) >= top:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--t0-min", type=int, default=2_000_000)
    ap.add_argument("--t0-max", type=int, default=8_000_000)
    ap.add_argument("--step", type=int, default=200_000, help="t0 grid step (us)")
    ap.add_argument("--out", default="outputs/phase_g_t0.json")
    args = ap.parse_args()

    import numpy as np
    import physical_ai_av

    if args.from_csv:
        clips = _clips_from_csv(args.from_csv, args.top)
        print(f"[csv] top {len(clips)} clips: {[c[:8] for c in clips]}")
    elif args.clips:
        clips = args.clips
    else:
        ap.error("provide --clips or --from-csv")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    grid = list(range(args.t0_min, args.t0_max + 1, args.step))
    results = {}

    for cid in clips:
        try:
            obst = avdi.get_clip_feature(cid, "obstacle.offline",
                                         maybe_stream=True)["obstacle.offline"]
            obst["track_id"] = obst["track_id"].astype(str)
            xcol = "center_x" if "center_x" in obst.columns else "x"
            ycol = "center_y" if "center_y" in obst.columns else "y"

            best = {"score": 0.0, "t0_us": None, "track_id": None, "x": None, "y": None}
            for tid, tdf in obst.groupby("track_id"):
                tdf = tdf.sort_values("timestamp_us")
                ts = tdf["timestamp_us"].to_numpy(float)
                sx = float(tdf["size_x"].median()); sy = float(tdf["size_y"].median())
                for t0 in grid:
                    if t0 < ts.min() or t0 > ts.max():
                        continue
                    x = float(np.interp(t0, ts, tdf[xcol].to_numpy(float)))
                    y = float(np.interp(t0, ts, tdf[ycol].to_numpy(float)))
                    sc = blocking_score(x, y, sx, sy)
                    if sc > best["score"]:
                        best = {"score": round(sc, 3), "t0_us": int(t0),
                                "track_id": tid, "x": round(x, 1), "y": round(y, 2),
                                "size": [round(sx, 1), round(sy, 1)]}
            results[cid] = best
            b = best
            print(f"{cid[:8]}  best t0={b['t0_us']}  track={b['track_id']}  "
                  f"x={b['x']} y={b['y']} score={b['score']}")
        except Exception as e:
            print(f"{cid[:8]}  skip: {e}")
            results[cid] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n-> {args.out}")
    print("Next (GPU): python runners/audit_phase_g.py --t0-file", args.out, "--dump-mask")


if __name__ == "__main__":
    main()
