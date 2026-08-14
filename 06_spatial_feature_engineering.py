# %% [markdown]
# # Phase 3 — Spatial Data Engineering: Building the ML Panel
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Assembles the final modelling table: one row per (Sub-county, Year,
# Month), joining population (static), water abstraction (static),
# WRUA governance (static), NDVI (monthly), rainfall (monthly), and the
# conflict onset/persistence/severity features (monthly, decayed).
#
# Run this after 01, 02, and 03 (needs their outputs) — 04/05 (NLP/
# topics) are not required for this script but their output can be
# joined in afterward if you want topic/sentiment features in the
# panel too (see the note at the end).

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from name_cleaning import build_canonical_lookup, TARGET_COUNTIES
from spatial_join import load_boundaries, join_points_to_subcounty
from conflict_cleaning import parse_dates, clean_counts, flag_composite_records, compute_severity_score
from panel_builder import (
    build_grid, add_static_population, add_static_abstraction,
    add_static_wrua_count, add_timevarying_ndvi, add_timevarying_rainfall,
    add_conflict_features,
)
from output_paths import step_dir

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = step_dir("06_spatial_feature_engineering")
STEP01_DIR = step_dir("01_phase1_data_audit")
STEP03_DIR = step_dir("03_deduplication_check")

CENSUS_PATH = DATA_DIR / "kenya_census_2019_subcounty_stats.csv"
CONFLICT_CLEANED = STEP03_DIR / "conflict_cleaned.csv"
WRA_CLEANED = STEP01_DIR / "wra_cleaned.csv"
NDVI_FILTERED = STEP01_DIR / "ndvi_filtered.csv"
CHIRPS_FILTERED = STEP01_DIR / "chirps_filtered.csv"
WRUA_JOINED = STEP01_DIR / "wrua_joined.csv"
BOUNDARIES_SHP = DATA_DIR / "subcounty_boundaries.shp"
BOUNDARY_COUNTY_COL = "COUNTY"
BOUNDARY_SUBCOUNTY_COL = "SUBCOUNTY"

# Panel date range -- adjust START_DATE to match your conflict data's
# earliest Date_Start once you've confirmed it on the full dataset.
START_DATE = "2004-01-01"
END_DATE = "2026-01-01"

HALF_LIFE_DAYS = 180  # see src/conflict_cleaning.py -- tunable, sensitivity-tested in 02

timer = Timer()
timer.__enter__()
log("=== PHASE 3: SPATIAL DATA ENGINEERING STARTED ===")

# %% [markdown]
# ## 1. Load all Phase 1/1b outputs

# %%
log("STAGE 1/5: Loading cleaned data from earlier phases...")
required = {"census": CENSUS_PATH, "conflict_cleaned": CONFLICT_CLEANED,
            "wra_cleaned": WRA_CLEANED, "ndvi_filtered": NDVI_FILTERED,
            "chirps_filtered": CHIRPS_FILTERED, "wrua_joined": WRUA_JOINED,
            "boundaries": BOUNDARIES_SHP}
missing = {k: v for k, v in required.items() if not v.exists()}
if missing:
    print("  Missing required file(s) -- run the earlier phase scripts first:")
    for k, v in missing.items():
        print(f"    {k}: {v}")
    raise SystemExit(1)

census = pd.read_csv(CENSUS_PATH)
conflict = pd.read_csv(CONFLICT_CLEANED)
wra_cleaned = pd.read_csv(WRA_CLEANED)
ndvi_filtered = pd.read_csv(NDVI_FILTERED)
chirps_filtered = pd.read_csv(CHIRPS_FILTERED)
wrua_joined = pd.read_csv(WRUA_JOINED)

lookup = build_canonical_lookup(census)
print(f"  Canonical sub-county list: {len(lookup.table)} sub-counties across {TARGET_COUNTIES}")
log("STAGE 1/5: DONE.\n")

# %% [markdown]
# ## 2. Geocode conflict records to sub-county
#
# Uses Latitude/Longitude via point-in-polygon (NOT the free-text
# Sub_Location field, which is ward/landmark-level and doesn't
# reliably match sub-county boundaries -- confirmed on real sample
# data: values like "Karura Forest - Kiambu Road" aren't sub-county
# names at all).

# %%
log("STAGE 2/5: Geocoding conflict records to sub-county...")
boundaries = load_boundaries(str(BOUNDARIES_SHP), county_col=BOUNDARY_COUNTY_COL,
                              subcounty_col=BOUNDARY_SUBCOUNTY_COL)

conflict = parse_dates(conflict) if "Date_Start_parsed" not in conflict.columns else conflict
conflict["Date_Start_parsed"] = pd.to_datetime(conflict["Date_Start_parsed"])
if "Is_Composite" not in conflict.columns:
    conflict = clean_counts(conflict)
    conflict = flag_composite_records(conflict)
    conflict["severity_score"] = compute_severity_score(conflict)

conflict_geocoded = join_points_to_subcounty(
    conflict, boundaries, county_col=BOUNDARY_COUNTY_COL, subcounty_col=BOUNDARY_SUBCOUNTY_COL,
    lon_col="Longitude", lat_col="Latitude", out_prefix="geo",
)
n_ungeocoded = conflict_geocoded["geo_subcounty"].isna().sum()
log(f"  {len(conflict_geocoded) - n_ungeocoded}/{len(conflict_geocoded)} conflict records "
    f"successfully geocoded to a sub-county")
conflict_geocoded[["Record_ID", "geo_county", "geo_subcounty"]].to_csv(
    OUT_DIR / "conflict_geocoded.csv", index=False
)
log(f"  Saved -> {OUT_DIR / 'conflict_geocoded.csv'} (Record_ID + sub-county, for reuse "
    f"by 08_nlp_panel_features.py)")
if n_ungeocoded > 0:
    print(f"  WARNING: {n_ungeocoded} record(s) could not be geocoded (missing/invalid "
          f"coordinates, or fall outside all 4 target county boundaries) -- these will NOT "
          f"contribute to the panel's conflict features. Review before proceeding if this "
          f"number is large relative to your total record count.")
log("STAGE 2/5: DONE.\n")

# %% [markdown]
# ## 3. Build the panel grid and join static features

# %%
log("STAGE 3/5: Building panel grid and joining static features...")
grid = build_grid(lookup, START_DATE, END_DATE)
print(f"  Grid: {len(grid):,} rows ({grid['SubCounty'].nunique()} sub-counties x "
      f"{grid['panel_date'].nunique()} months)")

grid = add_static_population(grid, census, lookup)
grid = add_static_abstraction(grid, wra_cleaned)
grid = add_static_wrua_count(grid, wrua_joined)
log("STAGE 3/5: DONE.\n")

# %% [markdown]
# ## 4. Join time-varying features (NDVI, rainfall, conflict decay)

# %%
log("STAGE 4/5: Joining time-varying features...")
grid = add_timevarying_ndvi(grid, ndvi_filtered, lookup)
grid = add_timevarying_rainfall(grid, chirps_filtered, lookup)

print("  Computing conflict onset/persistence/severity per panel row "
      "(this is the slowest step -- one decay calculation per sub-county-month)...")
grid = add_conflict_features(grid, conflict_geocoded, half_life_days=HALF_LIFE_DAYS,
                              subcounty_col="geo_subcounty")

grid.to_csv(OUT_DIR / "ml_panel.csv", index=False)
log(f"STAGE 4/5: DONE. Saved -> {OUT_DIR / 'ml_panel.csv'}\n")

# %% [markdown]
# ## 5. Visualization — panel completeness + conflict feature sanity check

# %%
log("STAGE 5/5: Generating panel summary visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

completeness = grid[["Population_Density_per_SqKm", "total_abstraction_m3_per_day",
                      "wrua_count", "Mean_NDVI", "Rainfall_mm",
                      "conflict_persistence"]].notna().mean() * 100
axes[0].barh(completeness.index, completeness.values, color="#264653")
axes[0].set_xlabel("% of panel rows with a value")
axes[0].set_title("Panel completeness by feature")
axes[0].set_xlim(0, 100)

top_subcounty = grid.groupby("SubCounty")["conflict_persistence"].max().idxmax()
example = grid[grid["SubCounty"] == top_subcounty].sort_values("panel_date")
axes[1].plot(example["panel_date"], example["conflict_persistence"], label="persistence")
axes[1].plot(example["panel_date"], example["conflict_onset"].astype(int) *
             example["conflict_persistence"].max(), "r.", label="onset month", markersize=4)
axes[1].set_title(f"Conflict persistence over time — {top_subcounty}")
axes[1].legend(fontsize=8)

plt.tight_layout()
save_and_display(fig, OUT_DIR / "panel_summary.png")

log("STAGE 5/5: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 3 COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
File written: {OUT_DIR / 'ml_panel.csv'} ({len(grid):,} rows)
  Static (repeated across all months for a sub-county):
    Population_Density_per_SqKm, Total_Population,
    total_abstraction_m3_per_day, wrua_count
  Time-varying (genuinely differ month to month):
    Mean_NDVI, Rainfall_mm, Rainfall_Anomaly_mm/Percent,
    conflict_onset, conflict_persistence, conflict_severity_weighted

NOTE: NLP-derived features (sentiment, topics) from 04/05 are NOT yet
joined into this panel -- they're at the conflict-RECORD level, and
need the same decay/aggregation treatment as conflict_persistence to
become sub-county-month features (e.g. "decayed dominant topic" or
"average sentiment of recent events"), rather than a simple merge.
Worth a dedicated step once you've decided how you want NLP signal
represented at the panel level.

NEXT: Phase 4 (exploratory spatial analysis -- hotspots, KDE, DBSCAN,
temporal trends) or Phase 6 (ML modelling) can both start from this panel.
""")
