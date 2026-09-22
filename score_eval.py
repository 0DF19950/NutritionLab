#!/usr/bin/env python3
"""Join your labels back to the hidden scores and report how the model does."""
import pandas as pd, numpy as np

lab = pd.read_csv("eval/pairs_to_label.csv")
key = pd.read_csv("eval/_key.csv")
d = lab.merge(key, on="pair_id")
d = d[d["is_competitor"].notna()]
if d.empty:
    raise SystemExit("no labels yet - fill in the is_competitor column first")
d["is_competitor"] = d["is_competitor"].astype(int)

print("labelled pairs: %d  (%.0f%% positive)" % (len(d), d.is_competitor.mean() * 100))
print("\nmean score by label:")
print(d.groupby("is_competitor")["score"].agg(["mean", "median", "count"]).round(3).to_string())

# does the score separate good from bad at all?
pos, neg = d[d.is_competitor == 1]["score"], d[d.is_competitor == 0]["score"]
if len(pos) and len(neg):
    auc = (np.greater.outer(pos, neg).mean() + 0.5 * np.equal.outer(pos, neg).mean())
    print("\nAUC (P[score(good) > score(bad)]): %.3f   %s"
          % (auc, "no better than chance" if auc < 0.6 else
             "weak" if auc < 0.7 else "usable" if auc < 0.85 else "strong"))

print("\nprecision at each score threshold:")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    s = d[d.score >= t]
    if len(s):
        print("  >=%.1f  n=%-4d precision=%.2f" % (t, len(s), s.is_competitor.mean()))

print("\nwhich features actually predict a true competitor:")
for f in [c for c in d.columns if c.startswith("f_")]:
    s = d[[f, "is_competitor"]].dropna()
    if len(s) > 10 and s[f].nunique() > 1:
        print("  %-14s corr=%+.2f (n=%d)" % (f[2:], s[f].corr(s.is_competitor), len(s)))
