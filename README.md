# NutritionLab

Identify a product's real competitive set, then report its nutrition *relative to
that set* — rather than against a universal health score.

> "This yogurt has 21% less protein than comparable strawberry yogurts in France,
> but 29% less saturated fat. It ranks 30th of 31 on protein."

## Design rule

Features used to **choose** competitors are disjoint from features being
**compared**. Matching uses category / flavour / format / size / market /
positioning claims. Nutrition is never an input to selection — otherwise the
reported gap is partly an artefact of how the set was picked.

Note the split inside "dietary claims": `organic`, `vegan`, `gluten-free` are
positioning and are matched on. `high-protein`, `low-fat` are nutrient levels and
are deliberately excluded.

## Measured data reality (all 30,711 yogurts, not a sample)

| Signal | Coverage | Use |
|---|---|---|
| category | 100% | primary axis |
| product name | 97.7% | flavour + format are parsed from here |
| nutrition facts | 94.1% | the payload |
| brands | 84.4% | brand tier |
| ingredients | 65.9% | secondary |
| package size | 55.1% | weak filter |
| packaging detail | 27.8% | unused |
| retailer / stores | ~6% | unusable |
| **price** | **~0%** | **not available — see below** |

Flavour exists as a *category tag* for only 11.7% of yogurts but appears in the
product *name* for 36.3%, so flavour is derived, not looked up.

### Two things cannot be built from Open Food Facts

- **Price.** Open Prices holds ~311k prices against 4.73M products. "Price range"
  as a competitor signal and "price vs nutrition trade-offs" as an output need a
  separate source. Treat price as a pluggable optional signal.
- **Retailer availability.** ~6% coverage. Country works; retailer does not.

### Fields missing from the search API

`ingredients_text` and `serving_size` are **not in the search index** — they come
back empty even when requested. They exist in the bulk CSV export (211 columns).

## Layout

```
off_category.py    download a category via the search API (handles the 10k cap)
off_clean.py       cleaning: types, blanks->NaN, implausible values, quality flags
competitors.py     competitive-set selection + relative report
pipeline.py        raw -> processed, reproducible
make_eval_set.py   blind labelling worksheet
score_eval.py      join labels back to scores, report AUC / precision
data/raw/          API output, source of truth, never edited
data/processed/    cleaned CSVs, always re-derivable
eval/              hand-labelled ground truth (the only non-reproducible data)
```

## Run

```bash
python pipeline.py en:yogurts en:pasta-sauces   # download + clean
python pipeline.py en:yogurts --clean-only      # re-clean, no network
python make_eval_set.py en:yogurts --targets 25 --per-target 8
# ... fill in the is_competitor column in eval/pairs_to_label.csv ...
python score_eval.py
```

## Roadmap

1. **Label `eval/pairs_to_label.csv`** (200 pairs, ~1h). Nothing downstream is
   measurable until this exists.
2. Run `score_eval.py`. Baseline AUC and per-feature correlations tell you which
   of the five matching signals is carrying weight and which is noise.
3. Tune weights against that metric. Expand the flavour lexicon — it is the
   highest-leverage component and currently resolves ~63% of French yogurts.
4. Add ingredients + serving_size from the bulk CSV export (1.3 GB gzipped).
5. Only then: percentile confidence intervals, per-serving view, healthier
   alternatives.

## Known statistical limits

- At n≈30 a percentile carries roughly ±18 points of uncertainty. "Ranks 13 of 29"
  and "ranks 16 of 29" are not distinguishable. Report terciles or intervals.
- `serving_size` runs 38–55%, so per-serving comparison needs category-typical
  fallbacks, labelled as assumptions.
- The corpus skews French. Report set size and market so a thin comparison is
  visible as thin.
