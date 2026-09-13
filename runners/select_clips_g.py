#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
select_clips_g.py — Phase G: rank PhysicalAI-AV clips by expected maneuver impact.

Goal: find clips where an obstacle sits squarely in the ego's near path, so that
occluding it should move the TRAJECTORY well above the sampling-noise floor —
where action-side faithfulness finally becomes measurable (the parked-car scene
barely intruded; we need hard blockers / crossing agents).

CPU-only (streams obstacle labels, no model). Run on a cheap pod or any machine
with dataset access:

    # Mode 1: run on all clips from the Alpamayo parquet index
    python runners/select_clips_g.py \
        --clip-index /workspace/alpamayo2/notebooks/clip_ids.parquet \
        --max-clips 200 --out outputs/phase_g_candidates.csv

    # Mode 2: restrict to NuRec clips (enables photometric counterfactuals in Phase H)
    # First dump NuRec UUIDs: python runners/list_nurec_clips.py --out nurec_clips.txt
    python runners/select_clips_g.py \
        --clip-index /workspace/alpamayo2/notebooks/clip_ids.parquet \
        --nurec-list outputs/nurec_clips.txt \
        --out outputs/phase_g_nurec_candidates.csv

    # Mode 3: pass clip UUIDs directly (no parquet)
    python runners/select_clips_g.py \
        --clip-ids 0ea6fd88-... 06b483cf-... \
        --out outputs/custom_candidates.csv

Scoring per clip: for each labeled track near t0, interpolate its position at t0;
keep objects AHEAD (5 < x < 35 m) and IN-PATH (|y| < lane half-width). Score
favors close, centered, large objects. Resumable: appends to --out, skips clips
already scored.

NuRec advantage: clips in the NuRec subset can be audited with BOTH occlusion (black
box, fast) AND photometric counterfactuals (real 3D removal via Cosmos/Harmonizer) —
the comparison that settles whether pixel occlusion is a valid intervention.
"""

import argparse
import csv
import os

LANE_HALF_WIDTH_M = 1.8   # |y| below this at t0 = in the ego's corridor
X_MIN, X_MAX = 5.0, 35.0  # actionable range ahead


def score_track(x, y, sx, sy):
    closeness = max(0.0, 1.0 - (x - X_MIN) / (X_MAX - X_MIN))      # 1 near, 0 far
    centered = max(0.0, 1.0 - abs(y) / LANE_HALF_WIDTH_M)          # 1 dead-center
    size = min(1.0, (sx * sy) / 8.0)                               # ~car footprint -> 1
    return 0.45 * centered + 0.35 * closeness + 0.20 * size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-index", default=None,
                    help="parquet with a clip_id column (Alpamayo notebook format)")
    ap.add_argument("--clip-ids", nargs="+", default=None,
                    help="explicit list of clip UUIDs (no parquet needed)")
    ap.add_argument("--nurec-list", default=None,
                    help="text file with one NuRec UUID per line; filters clip-index to "
                         "this subset and adds a nurec=True column to the output")
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--out", default="outputs/phase_g_candidates.csv")
    args = ap.parse_args()
    if not args.clip_index and not args.clip_ids:
        ap.error("provide --clip-index or --clip-ids")

    import numpy as np
    import pandas as pd
    import physical_ai_av

    # construire la liste de clips à scorer
    if args.clip_ids:
        clip_ids = list(args.clip_ids)
    else:
        df_idx = pd.read_parquet(args.clip_index)
        col = "clip_id" if "clip_id" in df_idx.columns else df_idx.columns[0]
        clip_ids = df_idx[col].tolist()

    # filtrer sur le sous-ensemble NuRec si fourni
    nurec_set = set()
    if args.nurec_list:
        with open(args.nurec_list) as fh:
            nurec_set = {line.strip() for line in fh if line.strip()}
        clip_ids = [c for c in clip_ids if c in nurec_set]
        print(f"[nurec] {len(nurec_set)} scènes dans le fichier, "
              f"{len(clip_ids)} recoupements avec le parquet index")

    clip_ids = clip_ids[: args.max_clips]
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["clip_id"].tolist())
        print(f"[resume] {len(done)} clips already scored")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["clip_id", "nurec", "best_track_id", "score", "x_m", "y_m",
                        "size_x", "size_y", "n_inpath"])
        for i, cid in enumerate(clip_ids):
            if cid in done:
                continue
            try:
                obst = avdi.get_clip_feature(cid, "obstacle.offline",
                                             maybe_stream=True)["obstacle.offline"]
                obst["track_id"] = obst["track_id"].astype(str)
                best, best_row, n_inpath = 0.0, None, 0
                for tid, tdf in obst.groupby("track_id"):
                    tdf = tdf.sort_values("timestamp_us")
                    ts = tdf["timestamp_us"].to_numpy(dtype=float)
                    if args.t0_us < ts.min() or args.t0_us > ts.max():
                        continue
                    x = float(np.interp(args.t0_us, ts, tdf["x"].to_numpy(dtype=float)))
                    y = float(np.interp(args.t0_us, ts, tdf["y"].to_numpy(dtype=float)))
                    if not (X_MIN < x < X_MAX and abs(y) < LANE_HALF_WIDTH_M):
                        continue
                    n_inpath += 1
                    sx = float(tdf["size_x"].median()); sy = float(tdf["size_y"].median())
                    sc = score_track(x, y, sx, sy)
                    if sc > best:
                        best, best_row = sc, (tid, sc, x, y, sx, sy)
                if best_row:
                    tid, sc, x, y, sx, sy = best_row
                    in_nurec = "yes" if cid in nurec_set else ""
                    w.writerow([cid, in_nurec, tid, f"{sc:.3f}", f"{x:.1f}", f"{y:.2f}",
                                f"{sx:.1f}", f"{sy:.1f}", n_inpath])
                    fh.flush()
                    print(f"[{i}] {cid[:8]} score={sc:.2f} track={tid} x={x:.1f} y={y:+.2f}")
                else:
                    in_nurec = "yes" if cid in nurec_set else ""
                w.writerow([cid, in_nurec, "", "0", "", "", "", "", 0]); fh.flush()
            except Exception as e:
                print(f"[{i}] {cid[:8]} skip: {e}")

    df = pd.read_csv(args.out)
    top = df[df["score"] > 0].sort_values("score", ascending=False).head(30)
    print(f"\nTop candidates ({len(top)}):")
    print(top.to_string(index=False))
    print(f"\nNext: verify visually (--dump-mask) then run K>=20 rollouts per condition.")


if __name__ == "__main__":
    main()
