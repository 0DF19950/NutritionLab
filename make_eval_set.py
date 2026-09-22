#!/usr/bin/env python3
"""
Build a BLIND labelling worksheet for the competitor model.

Without ground truth you cannot tell whether score=0.675 is a good match, and
every weight you tune is guesswork. This samples target products, pulls their
top candidates, strips the scores, and shuffles - so your labels describe the
products rather than agreeing with the model. Scores are kept in a key file and
rejoined by score_eval.py.

    python make_eval_set.py en:yogurts --targets 25 --per-target 8 --country france
"""
import argparse, os
import numpy as np
import pandas as pd

from off_clean import clean_off
from competitors import Universe

OUT = "eval"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("category")
    p.add_argument("--targets", type=int, default=25)
    p.add_argument("--per-target", type=int, default=8)
    p.add_argument("--country", default="france")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    slug = a.category.replace(":", "_")
    df = pd.read_csv("data/processed/%s.csv" % slug, dtype={"code": str})
    if a.country:
        df = df[df["country_main"].eq(a.country)]
    df = df.reset_index(drop=True)
    u = Universe(df)
    rng = np.random.default_rng(a.seed)

    # stratify targets over flavour so the eval is not all strawberry
    pool = u.df[u.df["product_name"].notna() & u.df["proteins_100g"].notna()]
    groups = [g for _, g in pool.groupby(pool["flavour"].fillna("_none"))]
    rng.shuffle(groups)
    # round-robin across flavour strata until we have enough targets, so the
    # eval set is not dominated by whichever flavour happens to be commonest
    picks, used, gi = [], set(), 0
    while len(picks) < a.targets and gi < a.targets * 50:
        g = groups[gi % len(groups)]
        gi += 1
        avail = g[~g["code"].isin(used)]
        if avail.empty:
            continue
        row = avail.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        used.add(row["code"])
        picks.append(row)

    pairs, key = [], []
    for t in picks:
        try:
            i, ranked = u.competitors(t["code"])
        except Exception:
            continue
        top = ranked.head(a.per_target)
        for _, r in top.iterrows():
            j = r["idx"]
            pid = "%s_%s" % (t["code"], u.df.at[j, "code"])
            pairs.append({
                "pair_id": pid,
                "target_name": t["product_name"], "target_brand": t.get("brand"),
                "target_size_g": t.get("quantity_g"), "target_cats": t.get("categories"),
                "cand_name": u.df.at[j, "product_name"], "cand_brand": u.df.at[j, "brand"],
                "cand_size_g": u.df.at[j, "quantity_g"], "cand_cats": u.df.at[j, "categories"],
                "is_competitor": "",          # <- you fill this in: 1 or 0
                "note": "",
            })
            key.append({"pair_id": pid, "score": r["score"],
                        "coverage": r["coverage"],
                        **{("f_" + k): v for k, v in r["features"].items()}})

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(pairs).sample(frac=1, random_state=a.seed).to_csv(
        "%s/pairs_to_label.csv" % OUT, index=False)          # shuffled, no scores
    pd.DataFrame(key).to_csv("%s/_key.csv" % OUT, index=False)
    print("%d pairs from %d targets -> %s/pairs_to_label.csv" % (len(pairs), len(picks), OUT))
    print("fill in is_competitor (1/0), then run: python score_eval.py")


if __name__ == "__main__":
    main()
