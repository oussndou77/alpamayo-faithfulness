#!/usr/bin/env python3
r"""
run_phase_g_nurec.py — lance select_clips_g sur les clips NuRec sans parquet.

Usage:
    python runners/run_phase_g_nurec.py \
        --nurec-list outputs\nurec_clips.txt \
        --max-clips 200 \
        --out outputs\phase_g_nurec_candidates.csv
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--nurec-list", default="outputs/nurec_clips.txt")
ap.add_argument("--max-clips", type=int, default=200)
ap.add_argument("--t0-us", type=int, default=5_100_000)
ap.add_argument("--out", default="outputs/phase_g_nurec_candidates.csv")
args = ap.parse_args()

with open(args.nurec_list) as fh:
    clip_ids = [l.strip() for l in fh if l.strip()]
print(f"[wrap] {len(clip_ids)} NuRec clips chargés depuis {args.nurec_list}")

# appelle select_clips_g.main() directement
import select_clips_g as sg

# monkey-patch argparse pour passer les bons args
import argparse as _ap
fake = _ap.Namespace(
    clip_index=None,
    clip_ids=clip_ids,
    nurec_list=args.nurec_list,
    t0_us=args.t0_us,
    max_clips=args.max_clips,
    out=args.out,
)

# on réutilise le corps de main() en injectant les args
import numpy as np
import pandas as pd
import physical_ai_av
import csv

nurec_set = set(clip_ids)
clip_ids = clip_ids[:args.max_clips]
done = set()
if os.path.exists(args.out):
    done = set(pd.read_csv(args.out)["clip_id"].tolist())
    print(f"[resume] {len(done)} clips déjà scorés")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

new_file = not os.path.exists(args.out)
with open(args.out, "a", newline="") as fh2:
    w = csv.writer(fh2)
    if new_file:
        w.writerow(["clip_id", "nurec", "best_track_id", "score",
                    "x_m", "y_m", "size_x", "size_y", "n_inpath"])
    for i, cid in enumerate(clip_ids):
        if cid in done:
            continue
        try:
            obst = avdi.get_clip_feature(
                cid, "obstacle.offline", maybe_stream=True)["obstacle.offline"]
            obst["track_id"] = obst["track_id"].astype(str)
            best, best_row, n_inpath = 0.0, None, 0
            for tid, tdf in obst.groupby("track_id"):
                tdf = tdf.sort_values("timestamp_us")
                ts = tdf["timestamp_us"].to_numpy(dtype=float)
                if args.t0_us < ts.min() or args.t0_us > ts.max():
                    continue
                # NuRec uses center_x/center_y; Alpamayo parquet uses x/y
                xcol = "center_x" if "center_x" in tdf.columns else "x"
                ycol = "center_y" if "center_y" in tdf.columns else "y"
                x = float(np.interp(args.t0_us, ts, tdf[xcol].to_numpy(dtype=float)))
                y = float(np.interp(args.t0_us, ts, tdf[ycol].to_numpy(dtype=float)))
                if not (sg.X_MIN < x < sg.X_MAX and abs(y) < sg.LANE_HALF_WIDTH_M):
                    continue
                n_inpath += 1
                sx = float(tdf["size_x"].median())
                sy = float(tdf["size_y"].median())
                sc = sg.score_track(x, y, sx, sy)
                if sc > best:
                    best, best_row = sc, (tid, sc, x, y, sx, sy)
            if best_row:
                tid, sc, x, y, sx, sy = best_row
                w.writerow(["yes", cid, tid, f"{sc:.3f}", f"{x:.1f}",
                            f"{y:.2f}", f"{sx:.1f}", f"{sy:.1f}", n_inpath])
                fh2.flush()
                print(f"[{i}] {cid[:8]} score={sc:.2f} x={x:.1f} y={y:+.2f}")
            else:
                w.writerow([cid, "yes", "", "0", "", "", "", "", 0])
                fh2.flush()
        except Exception as e:
            print(f"[{i}] {cid[:8]} skip: {e}")

df = pd.read_csv(args.out)
top = df[df["score"].astype(float) > 0].sort_values("score", ascending=False).head(20)
print(f"\nTop {len(top)} candidats Phase G (NuRec) :")
print(top[["clip_id", "score", "x_m", "y_m", "n_inpath"]].to_string(index=False))
print(f"\nRésultat complet -> {args.out}")
print("Prochaine étape : --dump-mask sur le top-5, puis K>=20 rollouts.")
