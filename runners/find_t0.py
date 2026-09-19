#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
find_t0.py — for each Phase-G clip, find the t0 (microseconds) where the cited object is
MOST blocking, so the GPU audit spends no rollout searching for the right moment.

CPU-only: streams obstacle labels, interpolates each track, and scores "blocking-ness"
over a grid of candidate t0 values.

CAUSAL UNIQUENESS (learned from the first Phase-G run). Geometry alone is not enough:
occluding a single object only tests causal attribution when that object is the ONLY
sufficient cause of the maneuver. Two failure modes found empirically:
  * overdetermination — the ego stops for a lead vehicle AND a red light; removing the
    vehicle leaves the light, so the trajectory cannot change (clip 15eaddae: the text
    correctly re-attributed to "red traffic light ahead", citation 95% -> 0%, but the car
    was already stopped, so there was no action-side margin at all);
  * redundancy — in a queue, removing one vehicle promotes the next one to lead vehicle,
    so the explanation stays literally true (clip 0fa7060f: citation 100% -> 100%).
--unique therefore requires a single in-path agent, an ego actually moving at t0, and no
other in-path agent close behind the target. Blocking-ness = close (small x) AND centered (small
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

# Pixel occlusion is only valid while the object stays a modest fraction of every camera.
# Below ~12 m an in-path vehicle fills most of the 30-deg tele frame: masking it either
# swallows the scene or (if capped) leaves its roof visible. Those clips belong to the
# photometric NuRec track. Above ~25 m the object stops forcing a real maneuver.
MASKABLE_X_MIN, MASKABLE_X_MAX = 12.0, 25.0

# causal-uniqueness thresholds (see docstring)
QUEUE_GAP_M = 15.0        # another in-path agent within this distance behind the target
                          # would simply become the new lead vehicle -> redundant cause
MIN_EGO_SPEED_MS = 2.0    # a stopped ego has no action-side margin to measure


def blocking_score(x, y, sx, sy, x_min=X_MIN, x_max=X_MAX):
    if not (x_min < x < x_max and abs(y) < LANE_HALF_WIDTH_M):
        return 0.0
    closeness = max(0.0, 1.0 - (x - x_min) / (x_max - x_min))
    centered = max(0.0, 1.0 - abs(y) / LANE_HALF_WIDTH_M)
    size = min(1.0, (sx * sy) / 8.0)
    return 0.45 * centered + 0.35 * closeness + 0.20 * size


def _ego_speed(avdi, cid, t0_us):
    """Ego speed (m/s) at t0 from the ego-motion feature; None if unavailable."""
    import numpy as np
    try:
        ego = avdi.get_clip_feature(cid, "egomotion", maybe_stream=True)
        df = list(ego.values())[0] if isinstance(ego, dict) else ego
        ts = df["timestamp_us"].to_numpy(float)
        xc = "x" if "x" in df.columns else "center_x"
        yc = "y" if "y" in df.columns else "center_y"
        x = np.interp([t0_us - 250_000, t0_us + 250_000], ts, df[xc].to_numpy(float))
        y = np.interp([t0_us - 250_000, t0_us + 250_000], ts, df[yc].to_numpy(float))
        return float(np.hypot(x[1] - x[0], y[1] - y[0]) / 0.5)
    except Exception:
        return None


def _inpath_agents(obst, t0_us, xcol, ycol, x_far=60.0):
    """All agents in the ego corridor at t0, as (track_id, x, y), sorted by distance."""
    import numpy as np
    out = []
    for tid, tdf in obst.groupby("track_id"):
        tdf = tdf.sort_values("timestamp_us")
        ts = tdf["timestamp_us"].to_numpy(float)
        if t0_us < ts.min() or t0_us > ts.max():
            continue
        x = float(np.interp(t0_us, ts, tdf[xcol].to_numpy(float)))
        y = float(np.interp(t0_us, ts, tdf[ycol].to_numpy(float)))
        if 0 < x < x_far and abs(y) < LANE_HALF_WIDTH_M:
            out.append((tid, x, y))
    return sorted(out, key=lambda r: r[1])


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
    ap.add_argument("--maskable", action="store_true",
                    help=f"restrict to objects in [{MASKABLE_X_MIN}, {MASKABLE_X_MAX}] m — the range "
                         "where pixel occlusion covers the object cleanly on every camera")
    ap.add_argument("--unique", action="store_true",
                    help="require a UNIQUE sufficient cause: exactly one in-path agent, "
                         "no follower within QUEUE_GAP_M behind it, and a moving ego — "
                         "without this, occlusion cannot change the action (see docstring)")
    ap.add_argument("--x-min", type=float, default=None)
    ap.add_argument("--x-max", type=float, default=None)
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

    x_min = args.x_min if args.x_min is not None else (MASKABLE_X_MIN if args.maskable else X_MIN)
    x_max = args.x_max if args.x_max is not None else (MASKABLE_X_MAX if args.maskable else X_MAX)
    print(f"[range] target object distance window: {x_min}-{x_max} m")
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
                    sc = blocking_score(x, y, sx, sy, x_min, x_max)
                    if sc > best["score"]:
                        best = {"score": round(sc, 3), "t0_us": int(t0),
                                "track_id": tid, "x": round(x, 1), "y": round(y, 2),
                                "size": [round(sx, 1), round(sy, 1)]}
            if args.unique and best.get("t0_us"):
                t0 = best["t0_us"]
                agents = _inpath_agents(obst, t0, xcol, ycol)
                n_inpath = len(agents)
                tgt_x = best["x"]
                followers = [a for a in agents
                             if a[0] != best["track_id"] and tgt_x < a[1] < tgt_x + QUEUE_GAP_M]
                speed = _ego_speed(avdi, cid, t0)
                reasons = []
                if n_inpath > 1 and followers:
                    reasons.append(f"queue: {len(followers)} agent(s) right behind target")
                if speed is not None and speed < MIN_EGO_SPEED_MS:
                    reasons.append(f"ego nearly stopped ({speed:.1f} m/s) — no action margin")
                best["n_inpath"] = n_inpath
                best["ego_speed_ms"] = round(speed, 2) if speed is not None else None
                if reasons:
                    best["rejected"] = "; ".join(reasons)
                    print(f"{cid[:8]}  SKIP ({best['rejected']})")
                    results[cid] = best
                    continue
            results[cid] = best
            b = best
            extra = ""
            if args.unique:
                extra = f"  n_inpath={b.get('n_inpath')} ego={b.get('ego_speed_ms')} m/s"
            print(f"{cid[:8]}  best t0={b['t0_us']}  track={b['track_id']}  "
                  f"x={b['x']} y={b['y']} score={b['score']}{extra}")
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
