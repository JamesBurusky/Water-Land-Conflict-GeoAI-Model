# %% [markdown]
# # Phase 6 (fix) — Temporal-Safe Topic Refit
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Fixes the leakage flagged during Phase 2/6: `05_topic_modelling.py`
# fit BERTopic on the FULL corpus (2004-2026), meaning topic boundaries
# were shaped by test-period documents before any train/test split
# happened. This script refits BERTopic using ONLY documents before
# CUTOFF_YEAR, then calls `.transform()` (not `.fit_transform()`) on
# documents at/after CUTOFF_YEAR -- so test-period text can influence
# which existing topic it's assigned to, but never what a topic means
# or where its boundaries sit.
#
# CUTOFF_YEAR here MUST match CUTOFF_YEAR in 10_ml_modeling.py -- using
# a different cutoff would mean the topic features and the model's
# train/test split disagree about what counts as "the future",
# reintroducing a subtler version of the same leakage problem.
#
# Needs internet access (downloads the embedding model), same as
# 05_topic_modelling.py.
#
# Requires outputs/conflict_nlp_enriched.csv from 04_nlp_pipeline.py --
# NOT conflict_cleaned.csv from 02 (which lacks sentiment_score and the
# other NLP columns that 08_nlp_panel_features.py needs downstream).
# Run order: 01 -> 02 -> 03 -> 04 -> (05 optional) -> 06 -> 07 (this) -> 08 -> 09 -> 10.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pipeline_utils import log, Timer
from topic_modeling import fit_topics_temporal_safe, build_place_name_stopwords
from name_cleaning import build_canonical_lookup
from output_paths import step_dir

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = step_dir("07_topic_refit_temporal_safe")
CONFLICT_CLEANED = step_dir("03_deduplication_check") / "conflict_cleaned.csv"  # fallback only -- lacks sentiment_score
NLP_ENRICHED_PATH = step_dir("04_nlp_pipeline") / "conflict_nlp_enriched.csv"    # has sentiment_score, needed by 08
CENSUS_PATH = DATA_DIR / "kenya_census_2019_subcounty_stats.csv"

CUTOFF_YEAR = 2022  # MUST MATCH 10_ml_modeling.py's CUTOFF_YEAR
MIN_TOPIC_SIZE = 15  # MUST MATCH 05_topic_modelling.py's MIN_TOPIC_SIZE, for a fair comparison

timer = Timer()
timer.__enter__()
log("=== TEMPORAL-SAFE TOPIC REFIT STARTED ===")

# %% [markdown]
# ## 1. Load and refit

# %%
log("STAGE 1/2: Loading conflict data and refitting topics (train-only)...")
if not NLP_ENRICHED_PATH.exists():
    if CONFLICT_CLEANED.exists():
        print(f"  {NLP_ENRICHED_PATH} not found -- run 04_nlp_pipeline.py first. "
              f"(Falling back to {CONFLICT_CLEANED} would produce a topics file "
              f"MISSING sentiment_score, which 08_nlp_panel_features.py needs -- "
              f"not doing that fallback silently.)")
    else:
        print(f"  Neither {NLP_ENRICHED_PATH} nor {CONFLICT_CLEANED} found -- "
              f"run 02_conflict_cleaning.py and 04_nlp_pipeline.py first.")
    raise SystemExit(1)

conflict = pd.read_csv(NLP_ENRICHED_PATH)
conflict["Date_Start_parsed"] = pd.to_datetime(conflict["Date_Start_parsed"])

# Same place-name exclusion as 05_topic_modelling.py -- keeps topic
# word lists free of county/sub-county/river names. Built the same way
# in both scripts so the two topic fits are comparable, not just each
# internally consistent.
if CENSUS_PATH.exists():
    canonical_subcounties = build_canonical_lookup(pd.read_csv(CENSUS_PATH)).table["SubCounty"].tolist()
else:
    print(f"  WARNING: {CENSUS_PATH} not found -- excluding county/river names "
          f"from topic words, but not sub-county names.")
    canonical_subcounties = []
place_stopwords = build_place_name_stopwords(canonical_subcounties=canonical_subcounties, nlp_enriched_df=conflict)
print(f"  Excluding {len(place_stopwords)} place-name terms from topic word lists "
      f"(static gazetteer + every specific place your own NER found in the real text)")

cutoff_date = pd.Timestamp(f"{CUTOFF_YEAR}-01-01")
result, topic_model, topic_info = fit_topics_temporal_safe(
    conflict, cutoff_date=cutoff_date,
    embedding_backend="sentence_transformer",
    min_topic_size=MIN_TOPIC_SIZE,
    extra_stopwords=place_stopwords,
)

print("\n  Topic overview (fit on training-period text only):")
print(topic_info[["Topic", "Count", "Name"]].to_string())

result.to_csv(OUT_DIR / "conflict_topics_temporal_safe.csv", index=False)
# Saving topic_info here matters, not just for reference: this topic
# model has its OWN topic numbering (fit only on training-period text,
# then test-period text transformed against it) -- completely
# different from 05_topic_modelling.py's topic model. If a dashboard
# (or anything else) reads 05's topic_info.csv while displaying
# per-record topic_id values that came from THIS script, topic "3"
# means two different things in the two files -- guaranteed mismatch.
# Saving it here lets downstream consumers use the file that actually
# matches the topic_id values in conflict_topics_temporal_safe.csv.
topic_info.to_csv(OUT_DIR / "topic_info_temporal_safe.csv", index=False)
log(f"STAGE 1/2: DONE. Saved -> {OUT_DIR / 'conflict_topics_temporal_safe.csv'}, "
    f"{OUT_DIR / 'topic_info_temporal_safe.csv'}\n")

# %% [markdown]
# ## 2. Compare against the original full-corpus topics (sanity check)
#
# Not a validity check on its own -- some drift is expected, since this
# model only saw training-period text. Large disagreement on TRAINING-
# period records specifically (which both models saw) would be worth
# investigating; disagreement on test-period records is normal and
# exactly what the fix is supposed to produce.

# %%
log("STAGE 2/2: Comparing against original (leakage-affected) topics...")
original_path = step_dir("05_topic_modelling") / "conflict_topics.csv"
if original_path.exists():
    original = pd.read_csv(original_path)[["Record_ID", "topic_id"]].rename(
        columns={"topic_id": "topic_id_original"}
    )
    compare = result[["Record_ID", "Date_Start_parsed", "topic_id"]].merge(
        original, on="Record_ID", how="left"
    )
    compare["is_train"] = compare["Date_Start_parsed"] < cutoff_date
    agreement = (compare["topic_id"] == compare["topic_id_original"])
    print(f"\n  Train-period agreement with original full-corpus fit: "
          f"{agreement[compare['is_train']].mean():.1%}")
    print(f"  Test-period agreement with original full-corpus fit: "
          f"{agreement[~compare['is_train']].mean():.1%} (lower here is EXPECTED "
          f"-- this is exactly the leakage being corrected)")
else:
    print(f"  {original_path} not found -- skipping comparison (not required, "
          f"just informative).")

log("STAGE 2/2: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("TEMPORAL-SAFE TOPIC REFIT COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
File written: {OUT_DIR / 'conflict_topics_temporal_safe.csv'}

NEXT STEPS to actually use this leakage-safe version:
  1. Re-run 08_nlp_panel_features.py -- it now prefers
     conflict_topics_temporal_safe.csv over conflict_topics.csv
     automatically if present.
  2. Re-run 10_ml_modeling.py with the SAME CUTOFF_YEAR ({CUTOFF_YEAR})
     to get leakage-safe topic-based features in the final model.

Keep outputs/conflict_topics.csv (the original full-corpus fit) for
your Phase 4 exploratory analysis and thesis discussion of themes --
that use case is fine with the full corpus since it's descriptive, not
a predictive train/test evaluation. Use conflict_topics_temporal_safe.csv
specifically for anything feeding the Phase 6 model.
""")
