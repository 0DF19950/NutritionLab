#!/usr/bin/env python3
"""
Download every Open Food Facts product in a category via the Search-a-licious API.

The API caps any single query at 10 000 results (page * page_size <= 10000), so
for larger categories this shards the query on nutriscore_grade, and sub-shards
on countries_tags when a shard is still too big. Results are de-duplicated by
barcode, so overlapping shards are harmless.

Usage:
    python off_category.py en:pasta-sauces
    python off_category.py en:yogurts --out yogurts
    python off_category.py en:sodas --country en:france
    python off_category.py --list-categories yogurt        # find the right tag
"""
import argparse, csv, json, sys, time, urllib.parse, urllib.request, urllib.error

API   = "https://search.openfoodfacts.org/search"
LIMIT = 10_000          # hard server-side cap: page * page_size
PAGE  = 1_000           # max page_size the API accepts
UA    = "off-category-export/1.0 (https://github.com/openfoodfacts; contact@example.com)"

FIELDS = [
    "code", "product_name", "brands", "quantity", "categories_tags",
    "countries_tags", "labels_tags", "nutriscore_grade", "nova_group",
    "ingredients_text", "serving_size", "nutriments",
]
# nutriments come back as a list of {name, value, 100g, serving, unit}; flatten these:
NUTRIENTS = ["energy-kcal", "fat", "saturated-fat", "carbohydrates", "sugars",
             "fiber", "proteins", "salt", "sodium"]


TAXONOMY = "https://static.openfoodfacts.org/data/taxonomies/categories.json"
CACHE    = ".off_categories.json"


def load_taxonomy():
    """The full category taxonomy (~4.6 MB), cached next to the script."""
    import os
    if not os.path.exists(CACHE):
        print("fetching category taxonomy ...", file=sys.stderr)
        req = urllib.request.Request(TAXONOMY, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(CACHE, "wb") as f:
            f.write(r.read())
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


def find_categories(term, lang="en", top=15):
    """Match `term` against category names and synonyms, then get live counts."""
    tax = load_taxonomy()
    t = term.lower().strip()
    hits = []
    for tag, node in tax.items():
        names = []
        for key in ("name", "synonyms"):
            v = (node.get(key) or {}).get(lang)
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, list):
                names.extend(v)
        for n in names:
            if t in n.lower():
                # exact name match sorts first, then shortest tag
                hits.append((0 if n.lower() == t else 1, len(tag), tag, names[0]))
                break
    hits.sort()
    out = []
    for _, _, tag, name in hits[:top]:
        try:
            d = get({"q": f'categories_tags:"{tag}"', "page_size": 1, "fields": "code",
                     "facets": "nutriscore_grade"})
            items = (d.get("facets") or {}).get("nutriscore_grade", {}).get("items", [])
            n = sum(i["count"] for i in items)
        except Exception:
            n = -1
        out.append((n, tag, name))
        time.sleep(0.3)
    out.sort(reverse=True)
    return out


def get(params, tries=5):
    """GET with retry on 429/5xx and polite backoff."""
    url = f"{API}?{urllib.parse.urlencode(params, doseq=True)}"
    delay = 2.0
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                print(f"    HTTP {e.code}, retrying in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay); delay *= 2; continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < tries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
    raise RuntimeError("unreachable")


def count_and_facet(q, facet):
    """Exact total for a query plus the value distribution of `facet`."""
    d = get({"q": q, "page_size": 1, "fields": "code", "facets": facet})
    items = (d.get("facets") or {}).get(facet, {}).get("items", [])
    return sum(i["count"] for i in items), [(i["key"], i["count"]) for i in items]


def fetch_query(q, sink, seen, pause):
    """Page through one query that is known to be under the 10k cap."""
    page, got = 1, 0
    while page * PAGE <= LIMIT + PAGE - 1:
        d = get({"q": q, "page": page, "page_size": PAGE, "fields": ",".join(FIELDS)})
        hits = d.get("hits") or []
        for h in hits:
            if h.get("code") and h["code"] not in seen:
                seen.add(h["code"]); sink.append(h)
        got += len(hits)
        print(f"    page {page}: +{len(hits)} (kept {len(seen)} unique)")
        if len(hits) < PAGE:
            break
        page += 1
        time.sleep(pause)
    return got


def plan(category, country, pause):
    """Return the list of queries needed to cover the whole category."""
    base = f'categories_tags:"{category}"'
    if country:
        base += f' AND countries_tags:"{country}"'
    total, grades = count_and_facet(base, "nutriscore_grade")
    print(f"  total products: {total}")
    if total <= LIMIT:
        return [base], total

    print(f"  over the {LIMIT} cap -> sharding on nutriscore_grade")
    queries = []
    for grade, n in grades:
        shard = f'{base} AND nutriscore_grade:"{grade}"'
        if n <= LIMIT:
            print(f"    grade {grade}: {n}")
            queries.append(shard)
            continue
        print(f"    grade {grade}: {n} -> sub-sharding on countries_tags")
        time.sleep(pause)
        _, countries = count_and_facet(shard, "countries_tags")
        covered = 0
        for c, cn in countries:
            print(f"      {c}: {cn}")
            queries.append(f'{shard} AND countries_tags:"{c}"')
            covered += cn
        if covered < n:
            print(f"      !! {n - covered} products fall outside the top countries "
                  f"facet and will be missed; use the Parquet dump for full coverage",
                  file=sys.stderr)
        time.sleep(pause)
    return queries, total


def flatten(h):
    """One product -> one flat CSV row. The API returns strings or lists here
    depending on the field, and nutriments as a dict already keyed `<name>_100g`."""
    def text(v):
        # product_name / ingredients_text: plain string, or a list of {lang,text}
        # Returns None (not "") when absent, so pandas reads it as a real NaN.
        if isinstance(v, str):
            return v.strip() or None
        if isinstance(v, list):
            for n in v:
                if isinstance(n, dict) and n.get("lang") == "en" and n.get("text"):
                    return n["text"].strip() or None
            for n in v:
                if isinstance(n, dict) and n.get("text"):
                    return n["text"].strip() or None
                if isinstance(n, str):
                    return n.strip() or None
        return None

    def joined(v):
        if isinstance(v, list):
            return "|".join(str(x) for x in v) or None
        if isinstance(v, str):
            return v.strip() or None
        return v if v is not None else None

    row = {
        "code":             h.get("code") or None,
        "product_name":     text(h.get("product_name")),
        "brands":           joined(h.get("brands")),
        "quantity":         joined(h.get("quantity")),
        "serving_size":     joined(h.get("serving_size")),
        "countries":        joined(h.get("countries_tags")),
        "labels":           joined(h.get("labels_tags")),
        "categories":       joined(h.get("categories_tags")),
        "nutriscore_grade": joined(h.get("nutriscore_grade")),
        "nova_group":       joined(h.get("nova_group")),
        "ingredients_text": text(h.get("ingredients_text")),
    }
    nuts = h.get("nutriments") or {}
    if isinstance(nuts, list):   # Parquet-style list of structs, just in case
        nuts = {f'{n.get("name")}_100g': n.get("100g")
                for n in nuts if isinstance(n, dict)}
    for k in NUTRIENTS:
        row[f"{k}_100g"] = nuts.get(f"{k}_100g")
    return row


def download_category(category, out=None, country=None, pause=1.0,
                      write_csv=True, write_jsonl=True, as_dataframe=False):
    """Download a whole category. Importable from a notebook.

    Returns the list of flat dict rows, or a pandas DataFrame when
    as_dataframe=True. Pass out=None to skip writing files entirely.

        rows = download_category("en:pasta-sauces")
        df   = download_category("en:yogurts", out="yogurts", as_dataframe=True)
    """
    print("category %s" % category)
    queries, total = plan(category, country, pause)

    sink, seen = [], set()
    for i, q in enumerate(queries, 1):
        print("  [%d/%d] %s" % (i, len(queries), q))
        fetch_query(q, sink, seen, pause)
        time.sleep(pause)

    rows = [flatten(h) for h in sink]

    if out:
        if write_jsonl:
            with open("%s.jsonl" % out, "w", encoding="utf-8") as f:
                for h in sink:
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        if write_csv and rows:
            with open("%s.csv" % out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print("\nwrote %s.csv / %s.jsonl" % (out, out))

    print("%d unique products (API reported %d)" % (len(rows), total))

    if as_dataframe:
        import pandas as pd
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c.endswith("_100g") or c == "nova_group":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="off_category.py")
    p.add_argument("category", nargs="?", help='canonical tag, e.g. en:pasta-sauces')
    p.add_argument("--out", help="output basename (default: the category tag)")
    p.add_argument("--country", help='restrict to one country, e.g. en:france')
    p.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--list-categories", metavar="TERM",
                   help="search category tags matching TERM and exit")
    a = p.parse_args(argv)

    if a.list_categories:
        for n, tag, name in find_categories(a.list_categories):
            print("%8d  %-48s %s" % (n, tag, name))
        return
    if not a.category:
        p.error("category is required")

    download_category(a.category, out=a.out or a.category.replace(":", "_"),
                      country=a.country, pause=a.pause)


if __name__ == "__main__":
    # Under Jupyter, sys.argv carries the kernel's own -f/--f connection-file
    # argument, which argparse would reject. Importing the module is the
    # supported path there, so only parse argv for a real command line.
    if "ipykernel" in sys.modules or any(
            x.startswith(("-f", "--f=")) for x in sys.argv[1:]):
        print(__doc__)
        print("Notebook detected - use the function API instead, e.g.:\n"
              "    from off_category import download_category, find_categories\n"
              "    df = download_category('en:pasta-sauces', as_dataframe=True)")
    else:
        main()
