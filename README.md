# Phase 1 — Data Audit & Preprocessing

## Setup (local machine)
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas geopandas shapely rapidfuzz matplotlib openpyxl
```
GeoPandas can be finicky to install (GDAL/GEOS dependencies). If `pip install geopandas`
fails, use conda instead: `conda install -c conda-forge geopandas`.

## Folder structure
```
geoai_conflict/
├── data/                        # put your FULL datasets here (not the samples)
│   └── subcounty_boundaries.shp # your boundary shapefile (+ .shx/.dbf/.prj siblings)
├── src/                         # shared modules used by the scripts below
├── outputs/                     # cleaned CSVs + figures land here
├── 01_phase1_data_audit.py
├── 02_conflict_cleaning.py
├── 03_deduplication_check.py
├── 04_nlp_pipeline.py
├── 05_topic_modelling.py
├── 06_spatial_feature_engineering.py
├── 07_topic_refit_temporal_safe.py
├── 08_nlp_panel_features.py
├── 09_exploratory_spatial_analysis.py
└── 10_ml_modeling.py
```
Run them in this numeric order — each one generally depends on outputs from
the ones before it (details in each section below).

## Running it
1. Replace the sample files in `data/` with your full datasets (same filenames, or
   update the `PATHS` dict at the top of `01_phase1_data_audit.py`).
2. Drop your sub-county boundary shapefile into `data/` and update
   `BOUNDARY_COUNTY_COL` / `BOUNDARY_SUBCOUNTY_COL` to match its actual attribute
   column names (open the .dbf or check in QGIS first if unsure).
3. In VS Code: open this folder, open `01_phase1_data_audit.py`, and run cells
   with Shift+Enter (the `# %%` markers create the cell boundaries). Or run the
   whole thing with `python 01_phase1_data_audit.py`.
4. Check `outputs/` for cleaned CSVs and the two PNG figures. Read the
   "Manual review items" printed at the end and in the summary cell — any
   `unmatched` sub-county needs a decision (add to `MANUAL_ALIASES` in
   `name_cleaning.py`, or confirm it's a genuine data issue).

## What's validated vs. what needs your real data
- Sub-county name matching: tested and working against your real WRA sample
  (correctly collapsed `ATHI RIVER`/`ATHI-RIVER` → `Athi River`, correctly left
  genuine mismatches like `KIAMBAA`/`Kiambu West` unmatched for review).
- WRUA spatial join: logic tested against synthetic boundary polygons — **you
  need to point it at your real shapefile** to get real results.
- NDVI/CHIRPS county filter: mechanically correct, but the samples don't include
  our 4 counties, so 0 rows out is expected here — verify on the full files.

## Phase 1b — Conflict dataset cleaning (`02_conflict_cleaning.py`)

Cleans the conflict dataset itself and implements the onset/persistence/
severity temporal methodology:

- `src/conflict_cleaning.py` — date parsing, casualty/displacement count
  parsing (handles `"69+"` and `"Unknown"` correctly), composite/chronic
  detection, severity scoring, and `decayed_persistence()` — the core
  function Phase 5 will reuse to build the sub-county+month+year ML panel.
- `src/pipeline_utils.py` — shared logging/plotting helpers (used by both
  phase scripts so behavior stays consistent going forward).
- Run it the same way as Phase 1: open in VS Code, run cells with
  Shift+Enter, or `python 02_conflict_cleaning.py`.
- Update `CONFLICT_PATH` at the top if your full conflict file has a
  different name than `Final_Main_kenya_land_water_conflicts.csv`.
- `LONG_DURATION_THRESHOLD_DAYS` (180) and `HALF_LIVES_DAYS` ([90, 180, 365])
  in `src/conflict_cleaning.py` / `02_conflict_cleaning.py` are the two
  hyperparameters this stage introduces — both are explicit sensitivity
  checks, not fixed facts, and are meant to be reported as such.

**Outputs:** `outputs/conflict_cleaned.csv`, `outputs/conflict_decay_sensitivity.png`

## Phase 1c — Duplicate detection (`03_deduplication_check.py`)

Run this AFTER both scripts above. Two checks:

1. **Exact duplicates** across all six datasets (repeated full rows or repeated
   ID values from re-exports/re-merges) — `src/dedup_check.py:check_exact_duplicates`.
2. **Near-duplicate conflict records** — the same real-world incident documented
   independently by two different sources (plausible given the conflict dataset
   was built from GDELT/ACLED/ReliefWeb/Kenya Law/NGO/media). A pair is only
   flagged when county, date proximity, AND text similarity all agree —
   validated against a deliberately injected duplicate before being shipped
   here. **Nothing is auto-dropped** — candidates are written to
   `outputs/conflict_near_duplicates_REVIEW.csv` for you to check against
   `Source_URL`/`Full_Text_Description` and decide manually.

**Outputs:** `outputs/conflict_near_duplicates_REVIEW.csv` (only if candidates are found).

## Phase 2 — NLP pipeline (`04_nlp_pipeline.py`)

Covers text cleaning, sentiment, and NER (topic modelling is a separate
script since it needs internet access — see below).

- `src/text_cleaning.py` — lemmatized tokens for topic modelling, while
  keeping raw text intact for NER/sentiment (both need real capitalization
  to work well).
- `src/sentiment.py` — VADER (offline, default) or a transformer backend
  (needs `pip install transformers torch` + internet on first run — see
  the module docstring for when you'd want to switch).
- `src/ner_extraction.py` — spaCy NER augmented with a Kenya-specific
  gazetteer (`EntityRuler`). Validated against real sample text: plain
  spaCy mislabels "Turkana" and "Dassanech" as PERSON; the gazetteer
  fixes both. **Extend the RIVERS / WATER_INFRASTRUCTURE / COMMUNITIES /
  INSTITUTIONS lists in `src/ner_extraction.py`** as you review NER
  output on the full dataset and spot terms it's missing.

Setup: `pip install spacy vaderSentiment && python -m spacy download en_core_web_sm`

**Output:** `outputs/conflict_nlp_enriched.csv` — adds clean_tokens, clean_text,
sentiment_score, sentiment_label, entities_Place/River/Water_Infrastructure/
Community/Institution, and other_entities (anything outside the 5 target
categories, kept for visibility).

## Phase 2 step 4 — Topic modelling (`05_topic_modelling.py`)

**Needs internet access** — downloads a sentence-transformer embedding model
(~90MB, `all-MiniLM-L6-v2`) from HuggingFace on first run. Everything else in
this project runs fully offline; this is the one exception.

- `src/topic_modeling.py` — BERTopic wrapper. Runs on the RAW
  `Full_Text_Description` (not the lemmatized tokens) since the embedding
  model needs natural sentence structure — see the module docstring for why.
- `MIN_TOPIC_SIZE` at the top of the script (default 15) is tunable — too
  low gives noisy micro-topics, too high merges distinct themes or pushes
  too many records into the `-1` outlier group. Watch the printed outlier
  rate; if it's above ~40-50%, lower this and re-run.
- Setup: `pip install bertopic sentence-transformers`

**Outputs:** `outputs/conflict_topics.csv`, `outputs/topic_info.csv`,
`outputs/topic_distribution.png`. After running, read `topic_info.csv`'s
representative words per topic and give each a short human-readable name
for your write-up — BERTopic's auto-generated underscored names are
functional, not thesis-quality prose.

**Sandbox note:** this script's plumbing (clustering, topic extraction,
visualization) was fully tested using an offline TF-IDF substitute, since
this sandbox cannot reach HuggingFace (confirmed by direct test). The real
sentence-transformer embeddings — and therefore actual topic *quality* —
can only be confirmed on your machine.

## Post-run fixes (mojibake, outlier reduction, topic consolidation)

If you already ran the pipeline and saw garbled characters like `Ã¢â‚¬â€`
in your text, or got 15-20+ raw BERTopic topics that felt too granular:

1. **Mojibake repair** (`src/conflict_cleaning.py:repair_mojibake`) — now runs
   automatically as the first step of `02_conflict_cleaning.py`. Re-run
   `02` → `04` → `05` in sequence after updating these files, since the text
   fix needs to propagate through NLP and topic modelling.
2. **Outlier reduction** — `05_topic_modelling.py` now reassigns `-1` outlier
   records to their nearest real topic (`c-tf-idf` strategy) before doing
   anything else with the topics.
3. **Hierarchical topic reduction** — `TARGET_N_TOPICS` (default 10) merges
   the raw fine-grained topics down to a thesis-appropriate count, using
   BERTopic's own topic-similarity hierarchy (saved to
   `outputs/topic_hierarchy.csv`) rather than manual keyword guessing.
   `topic_id_coarse` / `topic_label_coarse` columns are added alongside the
   original fine-grained `topic_id` / `topic_label` — use the coarse ones
   for your thesis narrative, keep the fine-grained ones if you want more
   granular features later (e.g. distinguishing which specific court case
   or enforcement wave, not just "riparian encroachment" broadly).

**Outputs:** adds `outputs/topic_info_coarse.csv`, `outputs/topic_hierarchy.csv`,
and a persisted `outputs/bertopic_model/` (so you don't need to re-download/
re-embed if you just want to re-run the reduction steps with different
parameters).

## Phase 3 — Spatial data engineering (`06_spatial_feature_engineering.py`)

Assembles the final ML panel: one row per (Sub-county, Year, Month).

- Conflict records are geocoded to sub-county via `Latitude`/`Longitude`
  point-in-polygon (NOT the free-text `Sub_Location` field, which is
  ward/landmark-level — e.g. "Karura Forest - Kiambu Road" — and doesn't
  reliably match sub-county boundaries; confirmed on real sample data).
- Static features (population density, total abstraction, WRUA count) are
  repeated across all months for a sub-county — these source datasets have
  no time dimension of their own. This is a real, worth-stating limitation
  in your methodology.
- Time-varying features (NDVI, rainfall, conflict onset/persistence/severity)
  genuinely differ month to month.
- `src/panel_builder.py` does the assembly; run this after 01, 02, and 03.

**Outputs:** `outputs/ml_panel.csv` (the Phase 5/6 modelling table),
`outputs/panel_summary.png`.

**Not yet included:** NLP-derived features (sentiment, topics) from
04/05 — those are at the conflict-record level and need the same
decay/aggregation treatment as conflict_persistence to become proper
sub-county-month panel features. Worth a dedicated follow-up step.

## Phase 6 (early step) — Temporal-safe topic refit (`07_topic_refit_temporal_safe.py`)

Fixes the leakage flagged earlier: `05_topic_modelling.py` fit BERTopic on
the full corpus including test-period text. This script fits ONLY on
training-period text (`CUTOFF_YEAR` — must match `10_ml_modeling.py`), then
`.transform()`s test-period text against the already-fitted model, so
test-period text can never influence topic definitions.

Reads from `outputs/conflict_nlp_enriched.csv` (from `04_nlp_pipeline.py`),
NOT `conflict_cleaned.csv` — needs `sentiment_score` and the other NLP
columns to carry through, since `08_nlp_panel_features.py` needs them.

Validated on a large synthetic set: train fit found 4 clean topics with 0%
outlier rate, test transform correctly assigned held-out documents to those
same topics with a higher (expected) 14.3% outlier rate — proof the
train/transform split works correctly, not just that it runs.

**Run this, then re-run `08_nlp_panel_features.py`** (it now automatically
prefers `conflict_topics_temporal_safe.csv` over the original
`conflict_topics.csv` if present) **and `10_ml_modeling.py`** with the same
`CUTOFF_YEAR`.

Keep the original `outputs/conflict_topics.csv` for Phase 4's exploratory
analysis / thesis theme discussion — full-corpus fitting is fine there since
it's descriptive, not a predictive train/test evaluation. Use
`conflict_topics_temporal_safe.csv` specifically for anything feeding the
Phase 6 model.

**Outputs:** `outputs/conflict_topics_temporal_safe.csv`.

## Phase 3 (continued) — NLP panel features (`08_nlp_panel_features.py`)

Adds decay-weighted sentiment and topic features to `outputs/ml_panel.csv`,
using the SAME half-life/leakage rule as `conflict_persistence` (only past
events contribute), so all decay-based features are directly comparable.

- `decayed_sentiment` — decay-weighted average tone of recent/ongoing events
- `dominant_topic_id` — the BERTopic topic carrying the most decayed weight
  (join against `outputs/topic_info.csv` for the human-readable name)
- `topic_diversity` — Shannon entropy of the decayed topic distribution
  (0 = one theme dominates, higher = several themes co-occurring)

Run this AFTER `06_spatial_feature_engineering.py`, and ideally after
`07_topic_refit_temporal_safe.py` too (it auto-prefers the leakage-safe
topics file if present, falling back to `05_topic_modelling.py`'s
full-corpus fit otherwise). `HALF_LIFE_DAYS` at the top of this script
**must match** what `06` used.

**Outputs:** overwrites `outputs/ml_panel.csv` with the new columns added;
backs up the pre-NLP version to `outputs/ml_panel_pre_nlp.csv` first.

**Reminder filed for Phase 6:** BERTopic was fit on the full corpus including
test-period documents — refit using training-period data only before using
`dominant_topic_id`/`topic_diversity` in temporal train/test validation.

## Phase 4 — Exploratory spatial analysis (`09_exploratory_spatial_analysis.py`)

Hotspot analysis, KDE, DBSCAN, temporal trends, seasonal patterns.

- `src/spatial_analysis.py` — Getis-Ord Gi* (using KNN spatial weights, not
  Queen/Rook contiguity — Turkana is geographically isolated from the other
  3 counties, so contiguity weights would leave it with zero neighbors; see
  the module docstring), KDE, and DBSCAN (projects to UTM 37N/EPSG:32737
  before clustering so `eps` is a real distance in km, not meaningless degrees).
- `src/temporal_analysis.py` — yearly trend (with linear regression
  significance test) and seasonal (calendar-month) aggregation.
- Setup: `pip install libpysal esda pyproj`

**Tunable parameters** at the top of the script: `HOTSPOT_VALUE_COL` (try
`conflict_persistence`/`conflict_severity_weighted` too, not just onset
counts), `K_NEIGHBORS` (Getis-Ord), `DBSCAN_EPS_KM`/`DBSCAN_MIN_SAMPLES`.

**Outputs:** `outputs/hotspot_analysis.csv`, `outputs/hotspot_map.png`,
`outputs/conflict_with_clusters.csv`, `outputs/kde_dbscan.png`,
`outputs/yearly_trend.csv`, `outputs/seasonal_pattern.csv`,
`outputs/temporal_seasonal.png`.

**Note:** if a `subcounty_summary` value column has too few non-missing
sub-counties for the chosen `K_NEIGHBORS`, Getis-Ord is skipped with a
clear message rather than crashing — this can happen on small test data
but shouldn't on your real ~20-30 sub-county dataset.

## Phase 6 — ML model development (`10_ml_modeling.py`)

Logistic Regression (baseline), Random Forest (comparison), XGBoost (final),
predicting conflict onset `LAG_MONTHS` ahead.

**Critical fix implemented here:** `conflict_persistence` at the exact month
an event starts is always inflated by that same event (confirmed by direct
test) — using it un-lagged to predict onset would leak the answer into the
features. `src/ml_prep.py` lags every dynamic feature before it's used as a
predictor; static features (population, abstraction, WRUA count) don't need
lagging since they don't change month to month.

- `src/ml_prep.py` — feature lagging + temporal train/test split (hard year
  cutoff, not random — matches the spec's train-earlier/test-later design)
- `src/ml_models.py` — all three models, `TimeSeriesSplit` cross-validation
  (not plain k-fold, for the same leakage reason as the lag), full metric
  suite (accuracy/precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix)
- `CUTOFF_YEAR` and `LAG_MONTHS` at the top of the script are the two most
  important tunables — check the printed train/test positive-rate before
  trusting results; too few positives on either side make metrics meaningless
- Setup: `pip install scikit-learn xgboost joblib`

**Outputs:** `outputs/model_comparison.csv`, `outputs/xgboost_model.joblib`
(loadable for Phase 7), `outputs/model_evaluation.png`.

**Outstanding reminder:** `dominant_topic_id`/`topic_diversity` (if present in
your panel) were built from a BERTopic model fit on the full corpus including
test-period text — for full temporal rigor, exclude them or refit BERTopic on
training-period text only before relying on results that include them.


## Phase 7 — Model interpretability (`11_shap_interpretability.py`)

SHAP (TreeExplainer, exact for XGBoost) — answers "which factors contribute
most to conflict risk" with per-record, directional attribution, beyond
Phase 6's plain average feature importance.

- `src/shap_analysis.py` — SHAP value computation, importance summary (mean
  |SHAP| for magnitude, mean signed SHAP for direction, kept separate since
  a feature can have large impact while being direction-ambiguous), and a
  plain-language policy interpretation generator for your thesis discussion.
- `LAG_MONTHS` and `CUTOFF_YEAR` at the top **must match** `10_ml_modeling.py`
  — this script rebuilds the same train/test split rather than saving one,
  and warns if the model's saved features don't match what the panel
  currently produces (e.g. if the panel changed since the model was trained).
- Setup: `pip install shap`

**Outputs:** `outputs/shap_feature_importance.csv`,
`outputs/shap_policy_interpretation.txt`, `outputs/shap_summary_beeswarm.png`
(the richest single figure — magnitude + direction + per-record spread),
`outputs/shap_importance_bar.png`, `outputs/shap_dependence_plots.png`.

**Review before writing up:** cross-check the direction claims in
`shap_policy_interpretation.txt` against domain knowledge — a surprising
direction could be a genuine finding or a sign of a feature-construction
issue, worth investigating either way before reporting it.

## Phase 8 — Conflict risk mapping (`12_conflict_risk_mapping.py`)

Turns the trained model into an actual sub-county risk map, exported for the
Phase 9 dashboard.

- Risk = predicted P(onset next month) using each sub-county's MOST RECENT
  complete feature row — a genuine "as of now" forecast, not a re-score of
  historical months. State this explicitly in your thesis.
- Low/Medium/High uses RELATIVE tertiles across sub-counties, not fixed
  probability thresholds — conflict onset is rare overall (Phase 6: well
  under 1% of sub-county-months), so fixed thresholds would call almost
  everywhere "Low" and hide real relative differences. The raw
  `risk_probability` is always kept alongside the category.
- `src/risk_mapping.py` — prediction + categorization + boundary merge.
- `LAG_MONTHS` at the top **must match** `10_ml_modeling.py`.

**Outputs:** `outputs/conflict_risk_layer.geojson` (dashboard-ready, with
geometry), `outputs/conflict_risk_layer.csv` (same data, no geometry),
`outputs/risk_map.png` (choropleth with legend).

**Review before presenting:** "No data" sub-counties lack complete recent
feature history (e.g. an NDVI/rainfall gap) — worth checking whether that's
genuine or fixable. The tertile split is relative to this run's sub-county
set and will shift as data/probabilities change over time.
