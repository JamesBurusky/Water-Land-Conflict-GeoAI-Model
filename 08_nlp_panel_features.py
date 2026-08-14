# %% [markdown]
# # Phase 3 (continued) — Joining NLP Features into the ML Panel
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Adds decayed sentiment and topic features to the sub-county x month x
# year panel, using the SAME decay weighting as conflict_persistence
# (same half-life, same leakage-safe rule: only past events
# contribute) so all the decay-based features are directly comparable.
#
# Run this AFTER 05_topic_modelling.py (needs topic_id) and
# 06_spatial_feature_engineering.py (needs ml_panel.csv and
# conflict_geocoded.csv).

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pipeline_utils import log, Timer
from panel_builder import add_nlp_features
from output_paths import step_dir

pd.set_option("display.max_columns", None)

OUT_DIR = step_dir("08_nlp_panel_features")
STEP06_DIR = step_dir("06_spatial_feature_engineering")

INPUT_PANEL_PATH = STEP06_DIR / "ml_panel.csv"       # base panel from step 06 (pre-NLP)
GEOCODED_PATH = STEP06_DIR / "conflict_geocoded.csv"
TOPICS_SAFE_PATH = step_dir("07_topic_refit_temporal_safe") / "conflict_topics_temporal_safe.csv"  # preferred if present
TOPICS_PATH = step_dir("05_topic_modelling") / "conflict_topics.csv"          # full-corpus fit -- has leakage, see 07
NLP_ENRICHED_PATH = step_dir("04_nlp_pipeline") / "conflict_nlp_enriched.csv"  # fallback if neither topics file exists yet

# This step's OWN output: the full, NLP-augmented panel. Every script
# after this one (09, 10, 11, 12) reads ml_panel.csv from THIS folder,
# not from 06's -- 06's version is the base (pre-NLP) panel, kept there
# for reference; this is the complete version.
OUTPUT_PANEL_PATH = OUT_DIR / "ml_panel.csv"

HALF_LIFE_DAYS = 180  # MUST match what 06_spatial_feature_engineering.py used,
                       # so conflict_persistence and decayed_sentiment stay comparable

timer = Timer()
timer.__enter__()
log("=== PHASE 3 (CONTINUED): NLP PANEL FEATURES STARTED ===")

# %% [markdown]
# ## 1. Load panel, geocoded sub-county assignments, and NLP output

# %%
log("STAGE 1/3: Loading panel and NLP data...")
required = {"ml_panel": INPUT_PANEL_PATH, "conflict_geocoded": GEOCODED_PATH}
missing = {k: v for k, v in required.items() if not v.exists()}
if missing:
    print("  Missing required file(s) -- run 06_spatial_feature_engineering.py first:")
    for k, v in missing.items():
        print(f"    {k}: {v}")
    raise SystemExit(1)

panel = pd.read_csv(INPUT_PANEL_PATH, parse_dates=["panel_date"])
geocoded = pd.read_csv(GEOCODED_PATH)

if TOPICS_SAFE_PATH.exists():
    nlp = pd.read_csv(TOPICS_SAFE_PATH)
    print(f"  Using {TOPICS_SAFE_PATH} (leakage-safe: BERTopic fit on training-period "
          f"text only, test-period text transformed against the fitted model)")
elif TOPICS_PATH.exists():
    nlp = pd.read_csv(TOPICS_PATH)
    print(f"  WARNING: using {TOPICS_PATH} -- this was fit on the FULL corpus "
          f"including test-period text (see the note filed for Phase 6). Run "
          f"07_topic_refit_temporal_safe.py for a leakage-safe version before "
          f"finalizing any results that go into your thesis.")
elif NLP_ENRICHED_PATH.exists():
    nlp = pd.read_csv(NLP_ENRICHED_PATH)
    nlp["topic_id"] = -1
    print(f"  WARNING: {TOPICS_PATH} not found -- using {NLP_ENRICHED_PATH} instead. "
          f"topic_id set to -1 for all records (no real topic signal). Run "
          f"05_topic_modelling.py and re-run this script for real topic features.")
else:
    print(f"  Neither {TOPICS_PATH} nor {NLP_ENRICHED_PATH} found -- run "
          f"04_nlp_pipeline.py (and ideally 05_topic_modelling.py) first.")
    raise SystemExit(1)

print(f"  Panel: {len(panel):,} rows | NLP records: {len(nlp):,} | "
      f"geocoded records: {len(geocoded):,}")
log("STAGE 1/3: DONE.\n")

# %% [markdown]
# ## 2. Merge sub-county assignment onto the NLP records, then compute
# decayed sentiment/topic features per panel row

# %%
log("STAGE 2/3: Merging geocoding + computing decayed NLP features "
    "(this is the slow step, same as conflict_persistence in Phase 3)...")

nlp_geocoded = nlp.merge(geocoded, on="Record_ID", how="left")
n_unmatched = nlp_geocoded["geo_subcounty"].isna().sum()
if n_unmatched > 0:
    print(f"  WARNING: {n_unmatched} NLP record(s) had no matching geocoded "
          f"Record_ID -- these won't contribute to the NLP panel features. "
          f"This usually means 06_spatial_feature_engineering.py was run on "
          f"a different version of conflict_cleaned.csv than 04/05 used -- "
          f"re-run 04 and 05 after 06 if so.")

panel = add_nlp_features(panel, nlp_geocoded, half_life_days=HALF_LIFE_DAYS,
                          subcounty_col="geo_subcounty")

n_with_signal = panel["decayed_sentiment"].notna().sum()
print(f"  {n_with_signal:,}/{len(panel):,} panel rows have NLP signal "
      f"(the rest are before any conflict record in that sub-county -- "
      f"same pattern as conflict_persistence)")
log("STAGE 2/3: DONE.\n")

# %% [markdown]
# ## 3. Save

# %%
log("STAGE 3/3: Saving full NLP-augmented panel...")
# No backup-copy needed -- the pre-NLP base panel already lives
# untouched in 06's own folder (INPUT_PANEL_PATH); this step's output
# is simply the complete version, in its own folder.
panel.to_csv(OUTPUT_PANEL_PATH, index=False)
log(f"STAGE 3/3: DONE. Saved -> {OUTPUT_PANEL_PATH}\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 3 (NLP PANEL FEATURES) COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
{OUTPUT_PANEL_PATH} now additionally includes:
  - decayed_sentiment: decay-weighted average sentiment of recent/
    ongoing conflict events in this sub-county-month (NaN if no
    conflict history yet)
  - dominant_topic_id: the BERTopic topic carrying the most decayed
    weight among recent events (-1 = outlier/no dominant theme)
  - topic_diversity: Shannon entropy of the decayed topic distribution
    -- low = one theme dominates, high = several themes co-occurring

NOTE: dominant_topic_id is a NUMBER (BERTopic's topic index), not the
human-readable name. Join against outputs/topic_info.csv (Topic ->
Name) if you want readable labels in your analysis/plots.

REMINDER (filed for Phase 6): BERTopic was fit on the full corpus,
including test-period documents -- refit using only training-period
data before using dominant_topic_id/topic_diversity in temporal
train/test validation.

NEXT: Phase 4 (exploratory spatial analysis) or Phase 6 (ML modelling)
can now both use the full panel, including NLP signal.
""")
