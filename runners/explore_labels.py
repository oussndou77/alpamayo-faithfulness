#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
explore_labels.py — discover the obstacle-label schema and find label-covered clips (Axis 2 prep).

v2, after a first run on a pod taught us:
  * the labels directory holds exactly: egomotion, egomotion.offline, obstacle.offline
  * offline labels don't cover every clip (our case-study clip has none)
  * loading a whole obstacle chunk with pd.read_parquet OOM-kills a small container
    -> read the parquet SCHEMA without loading, then filter to a single clip_id (pyarrow)

NO GPU needed. Requires: pip install physical_ai_av pandas pyarrow, and HF_TOKEN.
Usage: python runners/explore_labels.py
"""

import io
import os
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

OBSTACLE_FEATURE = "obstacle.offline"
ALPAMAYO_CLIPS_URL = ("https://raw.githubusercontent.com/NVlabs/alpamayo/main/"
                      "notebooks/clip_ids.parquet")


def main():
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    print("========== 1. Which clips have obstacle labels? ==========")
    pres = avdi.feature_presence
    if OBSTACLE_FEATURE not in pres.columns:
        print(f"columns available: {list(pres.columns)}")
        raise SystemExit(f"{OBSTACLE_FEATURE} not in feature_presence columns!")
    labeled = pres.index[pres[OBSTACLE_FEATURE].astype(bool)]
    print(f"clips with {OBSTACLE_FEATURE}: {len(labeled)} / {len(pres)}")

    print("\n========== 2. Intersection with Alpamayo demo clips ==========")
    try:
        with urllib.request.urlopen(ALPAMAYO_CLIPS_URL, timeout=30) as r:
            alpa = pd.read_parquet(io.BytesIO(r.read()))
        col = "clip_id" if "clip_id" in alpa.columns else alpa.columns[0]
        alpa_ids = alpa[col].tolist()
        inter = [c for c in alpa_ids if c in set(labeled)]
        print(f"alpamayo demo clips: {len(alpa_ids)} | with obstacle labels: {len(inter)}")
        print("first 10 candidates for the next labeled capture:")
        for c in inter[:10]:
            print(f"   {c}  (index {alpa_ids.index(c)})")
        target = inter[0] if inter else labeled[0]
    except Exception as e:
        print(f"could not fetch alpamayo clip list ({e}); using any labeled clip")
        inter = []
        target = labeled[0]

    print(f"\n========== 3. Obstacle schema for clip {target} ==========")
    chunk = avdi.get_clip_chunk(target)
    fname = avdi.features.get_chunk_feature_filename(chunk, OBSTACLE_FEATURE)
    print(f"chunk file: {fname}")
    local = avdi.download_file(fname)
    print(f"downloaded to: {local}")

    pf = pq.ParquetFile(local)
    print(f"\nparquet schema ({pf.metadata.num_rows} rows total in chunk):")
    print(pf.schema_arrow)

    print("\n--- rows for this clip only (memory-safe filtered read) ---")
    tbl = pq.read_table(local, filters=[("clip_id", "==", target)])
    df = tbl.to_pandas()
    if df.empty:
        # clip_id may be the index rather than a column; fall back to first row-group sample
        print("(no 'clip_id' column match; sampling first rows of the chunk instead)")
        df = next(pf.iter_batches(batch_size=500)).to_pandas()
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(5).to_string(max_colwidth=30))
    for cand in ("object_type", "category", "label", "class", "type"):
        if cand in df.columns:
            print(f"\nunique {cand}: {sorted(df[cand].astype(str).unique())[:25]}")
    print("\nDone. Paste this output back for the Axis-2 wiring.")


if __name__ == "__main__":
    main()
