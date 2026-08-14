# %% [markdown]
# # Phase 2 — NLP Pipeline (Text Cleaning, Sentiment, NER)
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Covers Phase 2 steps 1-3 from the project spec. Topic modelling
# (step 4, BERTopic) is deliberately a SEPARATE script
# (`05_topic_modelling.py`) since it needs internet access to download
# a sentence-transformer embedding model on first run -- this script
# runs fully offline once spaCy's model is downloaded.
#
# Sentiment uses the VADER backend by default (fully offline, fast).
# See `src/sentiment.py` for the transformer-based alternative and why
# you might want to compare the two on your machine.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pipeline_utils import log, Timer
from text_cleaning import load_nlp_model, clean_text_series, split_delimited_field
from sentiment import compute_sentiment
from ner_extraction import load_ner_model, extract_entities
from name_cleaning import build_canonical_lookup
from output_paths import step_dir

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = step_dir("04_nlp_pipeline")

# conflict_cleaned.csv comes from step 03 (deduplication), NOT step 02 --
# 03 produces the ID-collision-resolved final version.
CONFLICT_CLEANED = step_dir("03_deduplication_check") / "conflict_cleaned.csv"
CENSUS_PATH = DATA_DIR / "kenya_census_2019_subcounty_stats.csv"
SENTIMENT_BACKEND = "vader"  # or "transformer" -- see src/sentiment.py

timer = Timer()
timer.__enter__()
log("=== PHASE 2: NLP PIPELINE STARTED ===")

# %% [markdown]
# ## 1. Load data

# %%
log("STAGE 1/4: Loading cleaned conflict data...")
if CONFLICT_CLEANED.exists():
    conflict = pd.read_csv(CONFLICT_CLEANED)
else:
    print(f"  {CONFLICT_CLEANED} not found -- run 01/02/03 first.")
    raise SystemExit(1)
print(f"  Loaded {len(conflict):,} conflict records")
log("STAGE 1/4: DONE.\n")

# %% [markdown]
# ## 2. Text cleaning
#
# Produces lemmatized tokens for topic modelling from
# Full_Text_Description, while keeping the original raw text intact
# for NER and sentiment (both need real capitalization/grammar to
# work well -- see the note in src/text_cleaning.py).

# %%
log("STAGE 2/4: Cleaning text fields...")
nlp_clean = load_nlp_model()
cleaned = clean_text_series(conflict["Full_Text_Description"], nlp_clean)
conflict["clean_tokens"] = cleaned["clean_tokens"]
conflict["clean_text"] = cleaned["clean_text"]
conflict["n_tokens_removed"] = cleaned["n_tokens_removed"]

conflict["NLP_Keywords_list"] = split_delimited_field(conflict["NLP_Keywords"])
conflict["Parties_Involved_list"] = split_delimited_field(conflict["Parties_Involved"], delimiter=";")

print(f"  Average tokens after cleaning: {conflict['clean_text'].str.split().str.len().mean():.1f}")
log("STAGE 2/4: DONE.\n")

# %% [markdown]
# ## 3. Sentiment analysis

# %%
log(f"STAGE 3/4: Computing sentiment (backend={SENTIMENT_BACKEND})...")
sentiment_result = compute_sentiment(conflict["Full_Text_Description"], backend=SENTIMENT_BACKEND)
conflict["sentiment_score"] = sentiment_result["sentiment_score"]
conflict["sentiment_label"] = sentiment_result["sentiment_label"]

print("\n  Sentiment label distribution:")
print(conflict["sentiment_label"].value_counts().to_string())
log("STAGE 3/4: DONE.\n")

# %% [markdown]
# ## 4. Named Entity Recognition
#
# Uses the gazetteer-augmented NER model -- validated against real
# sample text to correctly catch "Turkana" and "Dassanech", which
# plain spaCy mislabeled as PERSON. Extend the gazetteer lists in
# `src/ner_extraction.py` as you review output on the full dataset.

# %%
log("STAGE 4/4: Extracting named entities...")
if CENSUS_PATH.exists():
    census = pd.read_csv(CENSUS_PATH)
    lookup = build_canonical_lookup(census)
    subcounty_list = lookup.table["SubCounty"].tolist()
else:
    print(f"  WARNING: {CENSUS_PATH} not found -- NER will run without the "
          f"sub-county gazetteer (county-level places still covered).")
    subcounty_list = None

nlp_ner = load_ner_model(canonical_subcounties=subcounty_list)
entities = extract_entities(conflict["Full_Text_Description"], nlp_ner)
conflict = pd.concat([conflict.reset_index(drop=True), entities.reset_index(drop=True)], axis=1)

for col in ["entities_Place", "entities_River", "entities_Water_Infrastructure",
            "entities_Community", "entities_Institution"]:
    n_nonempty = conflict[col].apply(len).gt(0).sum()
    print(f"  {col}: entities found in {n_nonempty}/{len(conflict)} records")

conflict.to_csv(OUT_DIR / "conflict_nlp_enriched.csv", index=False)
log(f"STAGE 4/4: DONE. Saved -> {OUT_DIR / 'conflict_nlp_enriched.csv'}\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 2 (steps 1-3) COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
File written: {OUT_DIR / 'conflict_nlp_enriched.csv'}
  - clean_tokens / clean_text: lemmatized text for topic modelling
  - sentiment_score / sentiment_label
  - entities_Place / entities_River / entities_Water_Infrastructure /
    entities_Community / entities_Institution
  - other_entities: anything spaCy found outside the 5 target
    categories (mostly PERSON names of quoted officials, dates,
    numbers) -- kept for visibility, not silently dropped

NEXT: 05_topic_modelling.py (BERTopic) -- needs internet access to
download a sentence-transformer model on first run, so make sure
you're online when you run it.
""")
