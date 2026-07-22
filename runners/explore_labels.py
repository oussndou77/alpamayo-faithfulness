#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
explore_labels.py — discover the dataset's label features and their schemas (Axis 2 prep).

The PhysicalAI-AV wiki's "Machine Labels" page is still "Coming Soon", so the only reliable
way to learn what's in `labels/` (cuboids? object types? which columns?) is to look.

NO GPU needed: this only touches dataset metadata + one clip's label features.
Requires: `pip install physical_ai_av` and a HF token with dataset access
(export HF_TOKEN=... / $env:HF_TOKEN="..." on Windows).

Usage:
    python runners/explore_labels.py [clip_id]
"""

import os
import sys

import pandas as pd

CASE_STUDY_CLIP = "0347d9f9-1493-4954-865d-1d8464e28501"   # the clip from the README case study


def describe(obj, name, max_rows=3):
    print(f"\n  --- {name} -> {type(obj).__name__}")
    if isinstance(obj, pd.DataFrame):
        print(f"      shape={obj.shape}")
        print(f"      columns={list(obj.columns)}")
        try:
            print(obj.head(max_rows).to_string(max_colwidth=40))
        except Exception:
            pass
    elif isinstance(obj, pd.Series):
        print(f"      index={list(obj.index)[:20]}")
        print(obj.head(max_rows))
    elif isinstance(obj, dict):
        print(f"      keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, f"{name}[{k}]", max_rows=2)
    else:
        print(f"      repr: {repr(obj)[:300]}")


def main():
    import physical_ai_av

    clip_id = sys.argv[1] if len(sys.argv) > 1 else CASE_STUDY_CLIP
    print(f"clip_id = {clip_id}")
    if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        print("WARNING: no HF_TOKEN in env and no cached login found — gated access may fail.")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    print("\n========== 1. All feature directories ==========")
    fdf = avdi.features.features_df
    for directory, grp in fdf.groupby("directory"):
        print(f"\n[{directory}]  ({len(grp)} features)")
        for feat in grp.index:
            print(f"   - {feat}")

    print("\n========== 2. Feature presence for this clip ==========")
    try:
        pres = avdi.feature_presence.loc[clip_id]
        present = [k for k, v in pres.items() if bool(v)]
        print(f"features present on this clip ({len(present)}):")
        for p in present:
            print(f"   - {p}")
    except Exception as e:
        print(f"could not read feature presence: {e}")

    print("\n========== 3. Loading each labels/* feature for this clip ==========")
    label_feats = sorted(avdi.features.LABELS.ALL) if hasattr(avdi.features, "LABELS") else []
    if not label_feats:
        print("no LABELS directory found!")
    for feat in label_feats:
        if "egomotion" in feat:
            print(f"\n  --- {feat} (skipped: egomotion, already used by the trajectory loader)")
            continue
        try:
            data = avdi.get_clip_feature(clip_id, feat, maybe_stream=True)
            describe(data, feat)
        except Exception as e:
            print(f"\n  --- {feat} -> FAILED: {type(e).__name__}: {str(e)[:200]}")

    print("\n========== 4. Bonus: reasoning/* features (human CoC references?) ==========")
    for dirname in ("REASONING",):
        ns = getattr(avdi.features, dirname, None)
        if ns is None:
            print(f"(no {dirname} directory)")
            continue
        for feat in sorted(ns.ALL):
            try:
                data = avdi.get_clip_feature(clip_id, feat, maybe_stream=True)
                describe(data, feat, max_rows=2)
            except Exception as e:
                print(f"\n  --- {feat} -> FAILED: {type(e).__name__}: {str(e)[:200]}")

    print("\nDone. Paste this full output back for the Axis-2 wiring.")


if __name__ == "__main__":
    main()
