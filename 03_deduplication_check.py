# %% [markdown]
# # Phase 1c — Duplicate Detection
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Two different checks:
#   1. Exact duplicates across all six raw datasets (repeated rows or
#      repeated ID values from re-exports/re-merges).
#   2. Near-duplicate CONFLICT records specifically -- the same
#      real-world incident documented independently by two different
#      sources (plausible here since the conflict dataset was built
#      from GDELT, ACLED, ReliefWeb, Kenya Law, and NGO/media sources).
#      These are flagged for manual review only -- NEVER auto-dropped.
#
# Run this AFTER 01_phase1_data_audit.py and 02_conflict_cleaning.py,
# since it reuses their cleaned outputs where available.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pipeline_utils import log, Timer
from dedup_check import (
    check_exact_duplicates, find_near_duplicate_conflicts,
    inspect_id_collisions, resolve_id_collisions,
)
from conflict_cleaning import parse_dates

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Prefer the cleaned outputs from earlier phases if they exist; fall
# back to the raw file otherwise (e.g. if you're running this before
# 02_conflict_cleaning.py for some reason).
CONFLICT_CLEANED = OUT_DIR / "conflict_cleaned.csv"
CONFLICT_RAW = DATA_DIR / "Final_Main_kenya_land_water_conflicts.csv"

RAW_PATHS = {
    "census": DATA_DIR / "kenya_census_2019_subcounty_stats.csv",
    "wra_cleaned": OUT_DIR / "wra_cleaned.csv",
    "ndvi_filtered": OUT_DIR / "ndvi_filtered.csv",
    "chirps_filtered": OUT_DIR / "chirps_filtered.csv",
    "wrua_joined": OUT_DIR / "wrua_joined.csv",
}
ID_COLS = {
    "census": None,
    "wra_cleaned": "WRA_ID",
    "ndvi_filtered": None,
    "chirps_filtered": None,
    "wrua_joined": "WRUA_NAME",
}

timer = Timer()
timer.__enter__()
log("=== PHASE 1c: DUPLICATE DETECTION STARTED ===")

# %% [markdown]
# ## 1. Exact duplicate check across all datasets

# %%
log("STAGE 1/2: Checking exact duplicates across all datasets...")
print()
for name, path in RAW_PATHS.items():
    if not path.exists():
        print(f"  {name}: SKIPPED (file not found at {path} -- run earlier "
              f"phase scripts first)")
        continue
    df = pd.read_csv(path)
    check_exact_duplicates(df, id_col=ID_COLS[name], label=name)
    print()
log("STAGE 1/2: DONE.\n")

# %% [markdown]
# ## 2. Near-duplicate conflict record detection
#
# Flags candidate pairs where the SAME county, overlapping/nearby
# dates, AND similar incident text all agree -- these need a human to
# check `Source_URL`/`Full_Text_Description` and decide whether it's
# genuinely one event reported twice (merge/drop one) or two distinct
# events that happen to read similarly (keep both).

# %%
log("STAGE 2/2: Checking for near-duplicate conflict records...")
if CONFLICT_CLEANED.exists():
    conflict = pd.read_csv(CONFLICT_CLEANED, parse_dates=["Date_Start_parsed", "Date_End_parsed"])
else:
    print(f"  {CONFLICT_CLEANED} not found -- loading and parsing raw file instead")
    conflict = parse_dates(pd.read_csv(CONFLICT_RAW))

check_exact_duplicates(conflict, id_col="Record_ID", label="conflict (exact)")

# %% [markdown]
# ### 2a. Resolve Record_ID collisions before anything else
#
# If Stage 2 above reported rows sharing a Record_ID with DIFFERING
# content, that's very likely an ID-generation collision from data
# collection (two genuinely different records assigned the same ID),
# not true duplicate rows -- true duplicates would show up as exact
# full-row matches, which were checked separately above. This is
# resolved BEFORE the near-duplicate text scan below, since a
# non-unique Record_ID would make that scan's output ambiguous to
# review (you wouldn't know which physical row a flagged ID refers to).

# %%
id_collision_rows = conflict[conflict.duplicated(subset=["Record_ID"], keep=False)]
if len(id_collision_rows) > 0:
    print(f"  {len(id_collision_rows)} row(s) involved in Record_ID collisions -- "
          f"inspecting before resolving:")
    review_cols = [c for c in ["Record_ID", "County", "Date_Start", "Incident_Summary",
                                "Source_Name"] if c in conflict.columns]
    preview = inspect_id_collisions(conflict, display_cols=review_cols)
    preview.to_csv(OUT_DIR / "record_id_collisions_REVIEW.csv", index=False)
    print(f"  Full detail saved -> {OUT_DIR / 'record_id_collisions_REVIEW.csv'} "
          f"(worth a quick skim to confirm these are genuinely different "
          f"incidents, not literal re-entries with minor text edits)")

    conflict = resolve_id_collisions(conflict, id_col="Record_ID")
    conflict.to_csv(CONFLICT_CLEANED, index=False)
    print(f"  Re-saved {CONFLICT_CLEANED} with unique Record_IDs "
          f"(original values kept in 'Original_Record_ID')")
else:
    print("  No Record_ID collisions found.")

# %% [markdown]
# ### 2b. Near-duplicate text scan
#
# Now that IDs are guaranteed unique, scan for records that are
# similar enough in wording AND close enough in county/date to be the
# same real-world incident reported by two different sources.

# %%

candidates = find_near_duplicate_conflicts(conflict)
print(f"\n  {len(candidates)} candidate near-duplicate pair(s) found "
      f"(similarity >= 80, same county, overlapping or within 30 days)")

if len(candidates) > 0:
    candidates.to_csv(OUT_DIR / "conflict_near_duplicates_REVIEW.csv", index=False)
    print(f"  Saved -> {OUT_DIR / 'conflict_near_duplicates_REVIEW.csv'}")
    print(f"  Review each row: check Source_URL / Full_Text_Description for "
          f"record_id_a and record_id_b in your original data. If it's the "
          f"same real-world incident, decide which record to keep (usually "
          f"the higher Confidence_Score or more complete one) -- do this "
          f"manually, this script does not auto-merge or auto-drop anything.")
else:
    print("  No candidates found -- nothing to review.")

log("STAGE 2/2: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 1c COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
