#!/usr/bin/env python3
"""
Reproducible dataset build. Raw API output is the source of truth and is never
edited; cleaning is a pure function applied on top, so you can always re-derive
processed data without re-downloading.

    python pipeline.py en:yogurts en:pasta-sauces
    python pipeline.py en:yogurts --clean-only      # re-clean from saved raw
"""
import argparse, json, os, sys
import pandas as pd

from off_category import download_category, flatten
from off_clean import clean_off, quality_report

RAW, PROC = "data/raw", "data/processed"


def build(category, clean_only=False, pause=0.5):
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)
    slug = category.replace(":", "_")
    raw_path = os.path.join(RAW, slug + ".jsonl")

    if clean_only or os.path.exists(raw_path):
        if not os.path.exists(raw_path):
            sys.exit("no raw file at %s - run without --clean-only first" % raw_path)
        print("using cached raw: %s" % raw_path)
        rows = [flatten(json.loads(l)) for l in open(raw_path, encoding="utf-8")]
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c.endswith("_100g") or c == "nova_group":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df = download_category(category, out=os.path.join(RAW, slug),
                               pause=pause, as_dataframe=True)

    clean = clean_off(df)
    out = os.path.join(PROC, slug + ".csv")
    clean.to_csv(out, index=False)
    print("\n" + quality_report(df, clean))
    print("-> %s" % out)
    return clean


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="+")
    p.add_argument("--clean-only", action="store_true",
                   help="rebuild processed data from cached raw, no network")
    p.add_argument("--pause", type=float, default=0.5)
    a = p.parse_args()
    for c in a.categories:
        print("\n" + "=" * 70)
        build(c, a.clean_only, a.pause)
