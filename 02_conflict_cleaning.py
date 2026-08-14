# %% [markdown]
# # Phase 1b — Conflict Dataset Cleaning & Temporal Feature Engineering
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# This is the second half of Objective 1: cleaning the conflict dataset
# itself (your dependent variable) and implementing the agreed temporal
# methodology — onset / persistence / severity, exponential decay from
# `Date_Start`, composite/chronic records given a longer half-life
# instead of a hard `Date_End` cutoff (to avoid leaking future
# information into the train-on-earlier-years / test-on-later-years
# design planned for Phase 6).
#
# Run cell-by-cell in VS Code's Interactive Window, or top-to-bottom as
# a plain script.

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from output_paths import step_dir
from conflict_cleaning import (
    parse_dates, clean_counts, flag_composite_records,
    compute_severity_score, decayed_persistence,
    LONG_DURATION_THRESHOLD_DAYS, repair_mojibake,
)

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = step_dir("02_conflict_cleaning")

CONFLICT_PATH = DATA_DIR / "Final_Main_kenya_land_water_conflicts.csv"

# Decay half-lives to test as a sensitivity check, in days. These are
# hyperparameters, not facts -- reporting how much results change
# across this range is the point (see the sensitivity plot below and
# the discussion in the methodology).
HALF_LIVES_DAYS = [90, 180, 365]  # 3, 6, 12 months
COMPOSITE_MULTIPLIER = 3.0  # composite/chronic records decay this much slower

timer = Timer()
timer.__enter__()
log("=== PHASE 1b: CONFLICT DATA CLEANING STARTED ===")

# %% [markdown]
# ## 1. Load and parse dates
#
# Date_Start / Date_End arrive as "M/D/YYYY" strings — parsed to real
# datetimes so they can be compared and subtracted. Any value that
# fails to parse becomes NaT and is printed, not silently dropped.

# %%
def read_csv_robust(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError as e:
        log(f"[encoding fallback] {path}: UTF-8 failed ({e}); retrying with cp1252")
        try:
            return pd.read_csv(path, encoding="cp1252", **kwargs)
        except UnicodeDecodeError:
            log(f"[encoding fallback] {path}: cp1252 also failed; using latin-1")
            return pd.read_csv(path, encoding="latin-1", **kwargs)


log("STAGE 1/4: Loading conflict dataset and repairing text encoding...")
conflict = read_csv_robust(CONFLICT_PATH)
print(f"  Loaded {len(conflict):,} conflict records, {conflict.shape[1]} columns")

print("\n  Checking for double-encoding corruption (mojibake) in text fields:")
conflict = repair_mojibake(conflict)

conflict = parse_dates(conflict)
n_unparsed_start = conflict["Date_Start_parsed"].isna().sum()
n_unparsed_end = conflict["Date_End_parsed"].isna().sum()
print(f"  Date_Start: {n_unparsed_start:,} record(s) failed to parse")
print(f"  Date_End:   {n_unparsed_end:,} record(s) failed to parse")
log("STAGE 1/4: DONE.\n")

# %% [markdown]
# ## 2. Clean casualty and displacement counts
#
# Splits each of Casualties_Reported / Displaced_Persons into a numeric
# estimate, an `is_lower_bound` flag (values like "69+"), and an
# `is_unknown` flag ("Unknown" stays as a real missing value, NOT a 0 —
# coding it as 0 would understate severity exactly where it's likely
# highest: the large, imprecisely-counted chronic/composite records).

# %%
log("STAGE 2/4: Cleaning casualty and displacement counts...")
conflict = clean_counts(conflict)

for col in ["Casualties_Reported", "Displaced_Persons"]:
    n_lower = conflict[f"{col}_is_lower_bound"].sum()
    n_unknown = conflict[f"{col}_is_unknown"].sum()
    print(f"  {col}: {n_lower:,} record(s) are lower-bound estimates (e.g. '69+'), "
          f"{n_unknown:,} record(s) are 'Unknown' (kept as missing, not 0)")
log("STAGE 2/4: DONE.\n")

# %% [markdown]
# ## 3. Flag composite/chronic records and compute severity
#
# `Is_Composite` fires if EITHER the recorded duration exceeds
# {LONG_DURATION_THRESHOLD_DAYS} days OR the Notes field uses
# chronic/composite/long-running wording — both component flags are
# kept separately so you can see which signal drove each record.

# %%
log("STAGE 3/4: Flagging composite/chronic records and scoring severity...")
conflict = flag_composite_records(conflict)
conflict["severity_score"] = compute_severity_score(conflict)

n_composite = conflict["Is_Composite"].sum()
n_by_duration = conflict["is_long_duration"].sum()
n_by_wording = conflict["notes_mentions_chronic"].sum()
print(f"  Is_Composite: {n_composite:,} / {len(conflict):,} records "
      f"({n_composite/len(conflict):.1%})")
print(f"    - flagged by duration > {LONG_DURATION_THRESHOLD_DAYS} days: {n_by_duration:,}")
print(f"    - flagged by Notes wording (composite/chronic/etc.): {n_by_wording:,}")
print(f"  severity_score: computed for {conflict['severity_score'].notna().sum():,} "
      f"records; {conflict['severity_score'].isna().sum():,} left as missing "
      f"(both casualty and displacement counts were 'Unknown')")

conflict.to_csv(OUT_DIR / "conflict_cleaned.csv", index=False)
log(f"STAGE 3/4: DONE. Saved -> {OUT_DIR / 'conflict_cleaned.csv'}\n")

# %% [markdown]
# ## 4. Decay half-life sensitivity check
#
# `decayed_persistence()` in `src/conflict_cleaning.py` is the building
# block Phase 5 will call once per sub-county/month to build the ML
# panel. Here we sanity-check it by plotting the severity-weighted
# persistence curve for each target county across the 3 candidate
# half-lives, over the full observed time range — this is the plot to
# put directly in your methodology section as the sensitivity-check
# evidence.

# %%
log("STAGE 4/4: Generating decay half-life sensitivity plots...")

TARGET_COUNTIES = ["Nairobi", "Kiambu", "Machakos", "Turkana"]
timeline = pd.date_range(
    conflict["Date_Start_parsed"].min(), pd.Timestamp.today(), freq="MS"
)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, county in zip(axes.flat, TARGET_COUNTIES):
    subset = conflict[conflict["County"] == county]
    if subset.empty:
        ax.set_title(f"{county} (no records)")
        continue
    for hl in HALF_LIVES_DAYS:
        values = [
            decayed_persistence(subset, t, half_life_days=hl,
                                 composite_half_life_multiplier=COMPOSITE_MULTIPLIER,
                                 severity_col="severity_score")
            for t in timeline
        ]
        ax.plot(timeline, values, label=f"{hl}d half-life ({hl // 30}mo)")
    ax.set_title(county)
    ax.legend(fontsize=8)

fig.suptitle("Severity-weighted persistence — half-life sensitivity by county")
plt.tight_layout()
save_and_display(fig, OUT_DIR / "conflict_decay_sensitivity.png")

log("STAGE 4/4: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 1b COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. conflict_cleaned.csv          - parsed dates, numeric casualty/
                                      displacement estimates with
                                      lower-bound/unknown flags,
                                      Is_Composite flag, severity_score
  2. conflict_decay_sensitivity.png - persistence curves per county
                                      across 3 candidate half-lives

NEXT: review conflict_cleaned.csv -- especially any Date_Start/Date_End
that failed to parse (printed above) and confirm the Is_Composite split
between duration-based and wording-based flags looks right on the full
data before this feeds into Phase 5's ML panel.
""")
