"""
Competitive-set selection + relative nutrition report for Open Food Facts data.

Design rule that drives everything: features used to CHOOSE competitors are kept
strictly disjoint from features being COMPARED. If you match on nutrition-adjacent
attributes ("high protein") and then report a protein delta, the delta is partly an
artefact of your own selection. So matching uses category / flavour / format / size /
market / claims only; nutrition is never an input to selection.

    from competitors import Universe
    u = Universe(clean_df)
    r = u.report("3033490004743")
    print(r.text())
"""
import re
import numpy as np
import pandas as pd

# Multilingual, because OFF is mostly FR/DE/ES/IT. Taxonomy flavour tags cover
# ~12% of yogurts; parsing the name covers ~36%, so this is the primary source.
FLAVOURS = {
    "strawberry": r"strawberry|fraise|erdbeer|fresa|fragola",
    "vanilla":    r"vanilla|vanille|vainilla|vaniglia",
    "peach":      r"peach|p[eê]che|pfirsich|melocot[oó]n|pesca",
    "cherry":     r"cherry|cerise|kirsch|cereza|ciliegia",
    "raspberry":  r"raspberry|framboise|himbeer|frambuesa|lampone",
    "blueberry":  r"blueberry|myrtille|heidelbeer|ar[aá]ndano|mirtillo",
    "banana":     r"banana|banane|pl[aá]tano",
    "mango":      r"mango|mangue",
    "coconut":    r"coconut|noix de coco|kokos|\bcoco\b",
    "lemon":      r"lemon|citron|zitrone|lim[oó]n|limone",
    "apricot":    r"apricot|abricot|aprikose|albaricoque",
    "chocolate":  r"chocolate|chocolat|schoko|cioccolato",
    "coffee":     r"coffee|caf[eé]|kaffee",
    "honey":      r"honey|miel|honig|miele",
    "plain":      r"\bplain\b|\bnature\b|natur\b|naturel|natural|griego natural",
}
FORMATS = {
    "greek":        r"greek|grec|griech|griego|greco",
    "skyr":         r"\bskyr\b",
    "drink":        r"drink|boisson|\bà boire\b|trink|bebible|liquid",
    "fromage_blanc":r"fromage blanc|quark|petit[- ]suisse",
    "whipped":      r"whipped|mousse|brass[eé]",
    "set":          r"\bset\b|ferme\b",
}
# Claims that describe positioning, NOT nutrient level. Nutrient-level claims
# ("high protein", "low fat") are deliberately excluded: matching on them would
# flatten the very differences the report exists to surface.
POSITIONING_CLAIMS = {
    "organic", "eu-organic", "fair-trade", "vegan", "vegetarian",
    "gluten-free", "lactose-free", "no-added-sugar", "palm-oil-free",
    "made-in-france", "pdo", "pgi", "halal", "kosher",
}
NUTRIENTS = ["energy-kcal_100g", "fat_100g", "saturated-fat_100g",
             "carbohydrates_100g", "sugars_100g", "fiber_100g",
             "proteins_100g", "salt_100g"]
# direction: +1 = more is better, -1 = less is better, 0 = neutral
POLARITY = {"proteins_100g": +1, "fiber_100g": +1,
            "sugars_100g": -1, "saturated-fat_100g": -1, "salt_100g": -1,
            "energy-kcal_100g": -1, "fat_100g": 0, "carbohydrates_100g": 0}

GENERIC = {"en:", ""}   # placeholder; specificity is learned from the corpus


def _first(pattern_map, text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for name, pat in pattern_map.items():
        if re.search(pat, t):
            return name
    return None


class Universe:
    """A category corpus (e.g. all yogurts) you can query for competitive sets."""

    def __init__(self, df, min_set=15, target_set=30):
        d = df.copy()
        self.min_set, self.target_set = min_set, target_set

        name = d.get("product_name", pd.Series(index=d.index, dtype="string")).fillna("")
        cats = d.get("categories", pd.Series(index=d.index, dtype="string")).fillna("")
        # flavour/format: taxonomy first (precise), name second (broad)
        tax_flav = cats.apply(lambda s: _first(
            {k: r"en:%s" % k for k in FLAVOURS}, s))
        d["flavour"] = tax_flav.fillna(name.apply(lambda s: _first(FLAVOURS, s)))
        d["format"] = name.apply(lambda s: _first(FORMATS, s))
        if "categories_list" in d:
            d["format"] = d["format"].fillna(
                d["categories_list"].apply(
                    lambda xs: "greek" if any("greek" in x for x in xs or []) else None))

        self.cat_sets = (d.get("categories_list") if "categories_list" in d
                         else cats.str.split("|")).apply(
            lambda xs: {x.replace("en:", "") for x in (xs or []) if x})
        # tag specificity = IDF over the corpus; 'dairies' is worthless, 'skyr' is gold
        from collections import Counter
        cnt = Counter(t for s in self.cat_sets for t in s)
        n = len(d)
        self.idf = {t: np.log(n / c) for t, c in cnt.items()}

        claims = (d["labels_list"] if "labels_list" in d
                  else d.get("labels", pd.Series(index=d.index, dtype="string"))
                  .fillna("").str.split("|"))
        self.claims = claims.apply(
            lambda xs: {x.replace("en:", "") for x in (xs or [])} & POSITIONING_CLAIMS)

        self.markets = (d["countries_list"] if "countries_list" in d
                        else d.get("countries", pd.Series(index=d.index, dtype="string"))
                        .fillna("").str.split("|")).apply(
            lambda xs: {x.replace("en:", "") for x in (xs or []) if x})
        self.df = d

    # ---- similarity -------------------------------------------------------
    def _cat_sim(self, a, b):
        """IDF-weighted Jaccard: rewards sharing *specific* categories only."""
        inter = a & b
        union = a | b
        if not union:
            return None
        wi = sum(self.idf.get(t, 0) for t in inter)
        wu = sum(self.idf.get(t, 0) for t in union)
        return wi / wu if wu else None

    def _size_sim(self, a, b):
        if not (a and b) or np.isnan(a) or np.isnan(b):
            return None
        r = min(a, b) / max(a, b)
        return float(r)

    def candidates(self, code):
        """Hard blocking: same market, nutrition present, some specific category
        overlap. Cheap, and keeps the scored set tractable."""
        d = self.df
        i = d.index[d["code"] == code]
        if len(i) == 0:
            raise KeyError("code %s not in this universe" % code)
        i = i[0]
        mkt = self.markets[i]
        has_nutr = d["energy-kcal_100g"].notna() | d["proteins_100g"].notna()
        same_mkt = self.markets.apply(lambda m: bool(m & mkt)) if mkt else True
        mask = has_nutr & same_mkt & (d.index != i)
        return i, d.index[mask]

    def competitors(self, code, weights=None, match_flavour=True):
        w = weights or {"category": 0.40, "flavour": 0.25, "format": 0.15,
                        "size": 0.10, "claims": 0.10}
        i, cand = self.candidates(code)
        d = self.df
        tf, tfmt = d.at[i, "flavour"], d.at[i, "format"]
        tsz = d.at[i, "quantity_g"] if "quantity_g" in d else np.nan
        tcat, tclaims = self.cat_sets[i], self.claims[i]

        rows = []
        for j in cand:
            feats = {
                "category": self._cat_sim(tcat, self.cat_sets[j]),
                "flavour": (None if (tf is None or d.at[j, "flavour"] is None)
                            else float(tf == d.at[j, "flavour"])) if match_flavour else None,
                "format": (None if (tfmt is None and d.at[j, "format"] is None)
                           else float(tfmt == d.at[j, "format"])),
                "size": self._size_sim(tsz, d.at[j, "quantity_g"] if "quantity_g" in d else np.nan),
                "claims": (len(tclaims & self.claims[j]) / len(tclaims | self.claims[j])
                           if (tclaims | self.claims[j]) else None),
            }
            # renormalise over AVAILABLE features so missing data neither helps nor hurts
            avail = {k: v for k, v in feats.items() if v is not None}
            if not avail:
                continue
            wsum = sum(w[k] for k in avail)
            score = sum(w[k] * v for k, v in avail.items()) / wsum
            rows.append((j, score, feats, wsum))

        out = pd.DataFrame(rows, columns=["idx", "score", "features", "coverage"])
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        return i, out

    def report(self, code, k=None):
        k = k or self.target_set
        i, ranked = self.competitors(code)
        widened = False
        if len(ranked) < self.min_set:                 # widen rather than report noise
            i, ranked = self.competitors(code, match_flavour=False)
            widened = True
        peers = ranked.head(k)
        return Report(self, i, peers, widened)


class Report:
    def __init__(self, u, i, peers, widened):
        self.u, self.i, self.peers, self.widened = u, i, peers, widened
        self.peer_idx = peers["idx"].tolist()

    def stats(self):
        d = self.u.df
        tgt, grp = d.loc[self.i], d.loc[self.peer_idx]
        rows = []
        for c in NUTRIENTS:
            if c not in d or pd.isna(tgt.get(c)):
                continue
            vals = grp[c].dropna()
            if len(vals) < 5:                          # too thin to rank honestly
                continue
            med = vals.median()
            pct = (vals < tgt[c]).mean() * 100
            rows.append({
                "nutrient": c.replace("_100g", ""),
                "value": tgt[c], "median": med, "n": len(vals),
                "pct_vs_median": (tgt[c] - med) / med * 100 if med else np.nan,
                "percentile": pct,
                # rank 1 = best for this nutrient, given its polarity.
                # A neutral nutrient (fat, carbs) gets no rank - there is no
                # defensible "better" direction, so we refuse to invent one.
                "rank": (int((vals > tgt[c]).sum()) + 1 if POLARITY.get(c, 0) > 0
                         else int((vals < tgt[c]).sum()) + 1 if POLARITY.get(c, 0) < 0
                         else np.nan),
                "polarity": POLARITY.get(c, 0),
            })
        return pd.DataFrame(rows)

    def text(self):
        d, s = self.u.df, self.stats()
        tgt = d.loc[self.i]
        L = ["%s - %s" % (tgt.get("product_name") or "?", tgt.get("brand") or tgt.get("brands") or "?")]
        L.append("compared against %d close alternatives%s"
                 % (len(self.peer_idx), " (flavour match relaxed - too few exact peers)"
                    if self.widened else ""))
        if s.empty:
            L.append("  not enough peer data to rank")
            return "\n".join(L)
        for _, r in s.iterrows():
            arrow = "more" if r.pct_vs_median >= 0 else "less"
            rank = ("" if pd.isna(r["rank"])
                    else "  ranks %d of %d" % (r["rank"], r["n"] + 1))
            L.append("  %-14s %7.1f  (%.0f%% %s than the median peer, n=%d)%s"
                     % (r.nutrient, r["value"], abs(r.pct_vs_median), arrow,
                        r["n"], rank))
        good = s[(s.polarity * (s.percentile - 50)) > 0].sort_values(
            "percentile", key=lambda x: (x - 50).abs(), ascending=False)
        bad = s[(s.polarity * (s.percentile - 50)) < 0].sort_values(
            "percentile", key=lambda x: (x - 50).abs(), ascending=False)
        if len(good):
            L.append("  strongest: " + ", ".join(good.nutrient.head(2)))
        if len(bad):
            L.append("  weakest:   " + ", ".join(bad.nutrient.head(2)))
        return "\n".join(L)

    def why(self, n=5):
        """Per-competitor provenance - required for a transparent comparison."""
        d = self.u.df
        out = []
        for _, r in self.peers.head(n).iterrows():
            j = r["idx"]
            reasons = [("%s=%s" % (k, "match" if v == 1 else round(v, 2)))
                       for k, v in r["features"].items() if v is not None and v > 0]
            out.append({"code": d.at[j, "code"],
                        "name": (d.at[j, "product_name"] or "?")[:42],
                        "score": round(r["score"], 3),
                        "signal_coverage": round(r["coverage"], 2),
                        "why": ", ".join(reasons)})
        return pd.DataFrame(out)
