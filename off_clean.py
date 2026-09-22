"""
Clean an Open Food Facts category export (the DataFrame from off_category.py).

Nothing is dropped silently: implausible *values* become NaN and every affected
row is recorded in `quality_flags`, so you can audit or filter yourself.

    from off_clean import clean_off, quality_report
    clean = clean_off(df)
    print(quality_report(df, clean))
"""
import re
import numpy as np
import pandas as pd

NUTRIENTS = ["energy-kcal_100g", "fat_100g", "saturated-fat_100g",
             "carbohydrates_100g", "sugars_100g", "fiber_100g",
             "proteins_100g", "salt_100g", "sodium_100g"]
GRAMS = [c for c in NUTRIENTS if c != "energy-kcal_100g"]
TAG_COLS = ["countries", "labels", "categories"]

# quantity units -> multiplier into grams (or ml, treated as 1:1 for sauces)
UNITS = {"g": 1.0, "gr": 1.0, "gram": 1.0, "grams": 1.0, "grammes": 1.0,
         "г": 1.0, "kg": 1000.0, "ml": 1.0, "cl": 10.0, "l": 1000.0,
         "oz": 28.3495, "lb": 453.592}
QTY_RE = re.compile(
    r"(?:(?P<mult>\d+)\s*[x×*]\s*)?"          # 3 x 95 g
    r"(?P<num>\d+(?:[.,]\d+)?)"                # 95
    r"(?:\s*[-–]\s*(?P<num2>\d+(?:[.,]\d+)?))?"  # 470 - 490 g  -> midpoint
    r"\s*(?P<unit>kg|g|gr|grammes|grams|gram|г|ml|cl|l|oz|lb)?\b",
    re.IGNORECASE)


def _txt(s):
    """Trim, collapse inner whitespace, de-shout ALL-CAPS names."""
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.replace({"": pd.NA})
    shouty = s.notna() & s.str.len().gt(3) & s.str.match(r"^[^a-z]*$", na=False)
    return s.mask(shouty, s.str.title())


def parse_quantity(v):
    """'3 x 95 g' -> 285.0 ; '1 kg' -> 1000.0 ; '470 - 490 g' -> 480.0.
    A bare number is assumed to be grams. Returns NaN if unparseable."""
    if not isinstance(v, str):
        return np.nan
    m = QTY_RE.search(v.strip().lower())
    if not m:
        return np.nan
    num = float(m.group("num").replace(",", "."))
    if m.group("num2"):                       # a range -> midpoint
        num = (num + float(m.group("num2").replace(",", "."))) / 2
    num *= UNITS.get(m.group("unit") or "g", np.nan)
    if m.group("mult"):
        num *= int(m.group("mult"))
    return num if 0 < num < 100_000 else np.nan


def clean_off(df, drop_empty=True):
    d = df.copy()

    # Blank strings are not data. Depending on how the frame was built (straight
    # from the API vs round-tripped through CSV) missing text arrives as "" or as
    # NaN; normalise first so .isna() means the same thing either way.
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype) == "string":
            d[c] = d[c].replace(r"^\s*$", np.nan, regex=True)

    flags = pd.Series([[] for _ in range(len(d))], index=d.index)

    def flag(mask, label):
        for i in d.index[mask.fillna(False)]:
            flags[i].append(label)

    # --- identifiers -------------------------------------------------------
    d["code"] = d["code"].astype("string").str.strip()
    bad_code = ~d["code"].str.fullmatch(r"\d+", na=False) | ~d["code"].str.len().isin([8, 12, 13])
    flag(bad_code, "odd_barcode")

    # --- text --------------------------------------------------------------
    for c in ("product_name", "ingredients_text"):
        if c in d:
            d[c] = _txt(d[c])
    if "brands" in d:
        b = d["brands"].astype("string").str.strip().replace({"": pd.NA})
        d["brands_list"] = b.str.split("|").apply(
            lambda xs: [x.strip() for x in xs if x.strip()] if isinstance(xs, list) else [])
        # primary brand, title-cased so "DIA"/"Dia"/"dia" collapse
        d["brand"] = (d["brands_list"].str[0].astype("string")
                      .str.replace(r"\s+", " ", regex=True).str.strip().str.title())

    # --- tag lists: 'en:france|en:organic' -> ['france','organic'] ----------
    for c in TAG_COLS:
        if c in d:
            d[c + "_list"] = (d[c].astype("string").fillna("")
                              .str.replace(r"\b[a-z]{2}:", "", regex=True)
                              .apply(lambda s: [x for x in s.split("|") if x]))
    if "countries_list" in d:
        d["country_main"] = d["countries_list"].str[0].astype("string")

    # --- categoricals ------------------------------------------------------
    if "nutriscore_grade" in d:
        g = d["nutriscore_grade"].astype("string").str.lower().str.strip()
        g = g.where(g.isin(list("abcde")))          # unknown/not-applicable -> NA
        d["nutriscore_grade"] = pd.Categorical(g, categories=list("abcde"), ordered=True)
    if "nova_group" in d:
        n = pd.to_numeric(d["nova_group"], errors="coerce")
        d["nova_group"] = n.where(n.isin([1, 2, 3, 4])).astype("Int8")

    # --- nutriments --------------------------------------------------------
    for c in NUTRIENTS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            flag(d[c] < 0, "negative_nutriment")
            d.loc[d[c] < 0, c] = np.nan
    if "energy-kcal_100g" in d:
        bad = d["energy-kcal_100g"] > 900          # physical max (pure fat ~900)
        flag(bad, "impossible_energy")
        d.loc[bad, "energy-kcal_100g"] = np.nan
    for c in GRAMS:
        if c in d:
            bad = d[c] > 100                        # >100 g per 100 g
            flag(bad, "over_100g")
            d.loc[bad, c] = np.nan

    # internal consistency -> flag only, values may still be usable
    if {"saturated-fat_100g", "fat_100g"} <= set(d):
        flag(d["saturated-fat_100g"] > d["fat_100g"] + 0.01, "satfat_gt_fat")
    if {"sugars_100g", "carbohydrates_100g"} <= set(d):
        flag(d["sugars_100g"] > d["carbohydrates_100g"] + 0.01, "sugars_gt_carbs")
    macro = [c for c in ("fat_100g", "carbohydrates_100g", "proteins_100g", "fiber_100g") if c in d]
    if macro:
        flag(d[macro].sum(axis=1, min_count=1) > 100, "macros_gt_100g")

    # sodium is derived from salt (salt = sodium x 2.5) -> redundant column
    if {"salt_100g", "sodium_100g"} <= set(d):
        d = d.drop(columns=["sodium_100g"])

    # --- quantity ----------------------------------------------------------
    if "quantity" in d:
        d["quantity_g"] = d["quantity"].apply(parse_quantity)
        flag(d["quantity"].notna() & d["quantity_g"].isna(), "unparsed_quantity")

    # --- duplicates: flag, never drop (they are usually real variants) ------
    key = (d.get("product_name", pd.Series(index=d.index, dtype="string")).str.lower()
           + "|" + d.get("brand", pd.Series(index=d.index, dtype="string")).str.lower()
           + "|" + d.get("quantity", pd.Series(index=d.index, dtype="string")).astype("string"))
    flag(key.notna() & key.duplicated(keep=False), "near_duplicate")

    # --- structurally empty columns ---------------------------------------
    d["quality_flags"] = flags.apply(lambda x: "|".join(x) if x else "")
    d["is_clean"] = d["quality_flags"] == ""
    if drop_empty:
        empty = [c for c in d.columns if d[c].isna().all()]
        d = d.drop(columns=empty)
        d.attrs["dropped_empty"] = empty
    return d


def quality_report(raw, clean):
    lines = ["rows: %d -> %d, cols: %d -> %d"
             % (len(raw), len(clean), raw.shape[1], clean.shape[1])]
    if clean.attrs.get("dropped_empty"):
        lines.append("dropped all-empty columns: " + ", ".join(clean.attrs["dropped_empty"]))
    f = clean["quality_flags"]
    counts = (f[f != ""].str.split("|").explode().value_counts())
    lines.append("clean rows: %d (%.1f%%)" % (clean["is_clean"].sum(),
                                              100 * clean["is_clean"].mean()))
    for k, v in counts.items():
        lines.append("  %-20s %d" % (k, v))
    return "\n".join(lines)
