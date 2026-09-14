#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
audit_phase_g.py — batch causal audit over the Phase-G clips (hard in-path blockers).

Loads Alpamayo 2 Super ONCE, then for each clip audits the cited object at the t0 chosen
by find_t0.py, with the full control battery: causal occlusion, negative control (a
distractor track), null control (baseline vs baseline noise floor). K>=20 to get above
the 40% categorical floor and measure ACTION-side faithfulness — the thing the parked-car
scene (0ea6fd88) could not show.

This reuses run_counterfactual_a2's building blocks (occlude_frames, run_side,
CAM_INDEX_TO_ID) so the mask logic and the 7-camera leak fix are shared.

Workflow (GPU pod):
    # 1) CPU, once: pick t0 per clip
    python runners/find_t0.py --from-csv outputs/phase_g_nurec_candidates.csv --top 5 \
        --out outputs/phase_g_t0.json
    # 2) VERIFY masks visually first (never spend rollouts blind)
    python runners/audit_phase_g.py --t0-file outputs/phase_g_t0.json --dump-mask
    #    -> download maskcheck_<clip8>_cam*.png, confirm the object is covered on ALL cameras
    # 3) full audit
    python runners/audit_phase_g.py --t0-file outputs/phase_g_t0.json \
        --k-rollouts 20 --out outputs/phase_g_audit.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _pick_control_track(obst, causal_tid, t0_us):
    """Choose a distractor: an in-frame track that is NOT the causal one and NOT in-path
    (so occluding it should not affect the plan if the probe is specific)."""
    import numpy as np
    xcol = "center_x" if "center_x" in obst.columns else "x"
    ycol = "center_y" if "center_y" in obst.columns else "y"
    best = None
    for tid, tdf in obst.groupby("track_id"):
        if tid == str(causal_tid):
            continue
        tdf = tdf.sort_values("timestamp_us")
        ts = tdf["timestamp_us"].to_numpy(float)
        if t0_us < ts.min() or t0_us > ts.max():
            continue
        x = float(np.interp(t0_us, ts, tdf[xcol].to_numpy(float)))
        y = float(np.interp(t0_us, ts, tdf[ycol].to_numpy(float)))
        # prefer a clearly off-path but visible object (lateral, mid-range)
        if 5 < x < 40 and 2.0 < abs(y) < 12.0:
            score = abs(y)  # more lateral = better distractor
            if best is None or score > best[1]:
                best = (tid, score, x, y)
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-file", required=True, help="JSON from find_t0.py")
    ap.add_argument("--k-rollouts", type=int, default=20)
    ap.add_argument("--agent", default="vehicle")
    ap.add_argument("--dump-mask", action="store_true",
                    help="only render maskcheck PNGs for every clip and exit (verify first!)")
    ap.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    ap.add_argument("--out", default="outputs/phase_g_audit.json")
    args = ap.parse_args()

    import numpy as np
    import torch
    import physical_ai_av
    from PIL import Image
    from alpamayo2_super import helper
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    import run_counterfactual_a2 as cf
    from afh.axes.counterfactual import score_counterfactual, score_control_contrast

    t0map = json.load(open(args.t0_file))
    clips = [(c, v) for c, v in t0map.items() if v.get("t0_us")]
    print(f"[audit] {len(clips)} clips with a valid t0")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # model loaded ONCE for the whole batch (skip in dump-mask mode)
    model = None
    if not args.dump_mask:
        model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                               device_map="cuda:0")

    audit = {}
    for cid, info in clips:
        t0 = info["t0_us"]; causal_tid = str(info["track_id"])
        print(f"\n===== {cid[:8]}  t0={t0}  causal_track={causal_tid} "
              f"(x={info.get('x')} y={info.get('y')}) =====")
        data = load_physical_aiavdataset(cid, t0_us=t0)
        frames = data["image_frames"]
        ts = data["absolute_timestamps"].cpu().numpy()
        cam_idx = (data["camera_indices"].cpu().numpy()
                   if hasattr(data["camera_indices"], "cpu") else np.asarray(data["camera_indices"]))

        obst = avdi.get_clip_feature(cid, "obstacle.offline", maybe_stream=True)["obstacle.offline"]
        obst["track_id"] = obst["track_id"].astype(str)
        tdf = obst[obst["track_id"] == causal_tid].sort_values("timestamp_us")
        sx = float(tdf["size_x"].median()); sy = float(tdf["size_y"].median()); sz = float(tdf["size_z"].median())
        masked = cf.occlude_frames(frames, ts, cam_idx, tdf, (sx, sy, sz),
                                   avdi.get_clip_feature(cid, "camera_intrinsics", maybe_stream=True),
                                   avdi.get_clip_feature(cid, "sensor_extrinsics", maybe_stream=True))

        if args.dump_mask:
            imgs = masked.cpu().numpy() if hasattr(masked, "cpu") else np.asarray(masked)
            for ci in range(imgs.shape[0]):
                arr = np.transpose(imgs[ci, 0], (1, 2, 0)).astype(np.uint8)
                Image.fromarray(arr).save(f"maskcheck_{cid[:8]}_cam{ci}.png")
            print(f"  saved maskcheck_{cid[:8]}_cam0..{imgs.shape[0]-1}.png")
            continue

        base_tr, base_tj, base_xy = cf.run_side(frames, data, helper, model, args.k_rollouts)
        cf_tr, cf_tj, cf_xy = cf.run_side(masked, data, helper, model, args.k_rollouts)
        res = score_counterfactual(cid, args.agent, base_tr, base_tj, cf_tr, cf_tj)
        print("  " + res.format_report().replace("\n", "\n  "))

        entry = {"t0_us": t0, "causal_track": causal_tid,
                 "object_x": info.get("x"), "object_y": info.get("y"),
                 "score": res.score, "verdict": res.verdict,
                 "baseline_citation": res.baseline_citation, "cf_citation": res.cf_citation,
                 "baseline_xy": base_xy, "cf_xy": cf_xy}

        ctrl_tid = _pick_control_track(obst, causal_tid, t0)
        if ctrl_tid:
            cdf = obst[obst["track_id"] == ctrl_tid].sort_values("timestamp_us")
            cxs = float(cdf["size_x"].median()); cys = float(cdf["size_y"].median()); czs = float(cdf["size_z"].median())
            ctrl_masked = cf.occlude_frames(frames, ts, cam_idx, cdf, (cxs, cys, czs),
                                            avdi.get_clip_feature(cid, "camera_intrinsics", maybe_stream=True),
                                            avdi.get_clip_feature(cid, "sensor_extrinsics", maybe_stream=True))
            ctr_tr, ctr_tj, ctr_xy = cf.run_side(ctrl_masked, data, helper, model, args.k_rollouts)
            ctr_res = score_counterfactual(cid, args.agent, base_tr, base_tj, ctr_tr, ctr_tj)
            contrast = score_control_contrast(cid, res, ctr_res)
            print("  " + contrast.format_report().replace("\n", "\n  "))
            entry.update({"control_track": ctrl_tid, "control_xy": ctr_xy,
                          "citation_contrast": contrast.citation_contrast,
                          "valid_probe": contrast.valid_probe})

        # null control: baseline vs baseline, different seeds
        null_tr, null_tj, null_xy = cf.run_side(frames, data, helper, model,
                                                args.k_rollouts, seed_offset=1000)
        null_res = score_counterfactual(cid, args.agent, base_tr, base_tj, null_tr, null_tj)
        entry["null_behavior_change"] = null_res.behavior_change
        entry["exceeds_noise_floor"] = bool(res.behavior_change > null_res.behavior_change)
        print(f"  noise floor (null): {null_res.behavior_change:.0%} | "
              f"causal: {res.behavior_change:.0%} | "
              f"{'ABOVE floor' if entry['exceeds_noise_floor'] else 'within noise'}")

        audit[cid] = entry

    if not args.dump_mask:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(audit, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")
        n_above = sum(1 for e in audit.values() if e.get("exceeds_noise_floor"))
        print(f"Summary: {n_above}/{len(audit)} clips show causal action effect above the noise floor")


if __name__ == "__main__":
    main()
