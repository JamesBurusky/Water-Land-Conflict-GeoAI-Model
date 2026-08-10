# %% [markdown]
# # Phase 1 — Data Audit and Preprocessing
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# This script is cell-structured (`# %%` markers) for VS Code's Python
# Interactive Window — run cell-by-cell with Shift+Enter, or top-to-bottom
# as a plain script (`python 01_phase1_data_audit.py`).
#
# **Progress & completion:** every stage below prints a timestamped
# `[HH:MM:SS] START ...` / `[HH:MM:SS] DONE ...` pair, and the script
# prints a clear `ALL PHASES COMPLETE` banner at the very end. If the
# script stops printing and you never see that banner, something is
# genuinely still running (or has failed with a visible traceback) —
# you should never be left guessing.
#
# **Plotting note:** when run as a plain script (not inside VS Code's
# Interactive Window / Jupyter), this script does NOT open a pop-up
# plot window — it saves figures straight to `outputs/` and continues.
# Pop-up windows via `plt.show()` block the entire script until you
# manually close them, which looks exactly like a hang. If you're
# running cell-by-cell in VS Code's Interactive Window, figures will
# still display inline as normal.

# %%
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from name_cleaning import build_canonical_lookup, match_dataframe, TARGET_COUNTIES
from spatial_join import load_boundaries, join_wrua_to_subcounty

pd.set_option("display.max_columns", None)


def _running_interactively() -> bool:
    """True inside VS Code's Interactive Window / Jupyter, False when run
    as a plain script (python file.py) -- used to decide whether it's
    safe to call plt.show() without blocking the whole script."""
    try:
        get_ipython()  # noqa: F821 -- only exists inside a Jupyter/IPython kernel
        return True
    except NameError:
        return False


SHOW_PLOTS = _running_interactively()


def log(msg: str) -> None:
    """Timestamped progress print -- so the script never runs silently."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_and_display(fig, path: Path) -> None:
    """Save a figure to disk always; only pop up a window if we're in an
    interactive kernel (VS Code Interactive Window / Jupyter). In plain
    script mode this never blocks."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log(f"Saved figure -> {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


_SCRIPT_START = time.time()
log("=== PHASE 1 SCRIPT STARTED ===")

# %% [markdown]
# ## Config — dataset paths (already pointed at your full local files)

# %%
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PATHS = {
    "census": DATA_DIR / "kenya_census_2019_subcounty_stats.csv",
    "wra": DATA_DIR / "Abstraction_Data_James.xlsx",
    "conflict": DATA_DIR / "Final_Main_kenya_land_water_conflicts.csv",
    "ndvi": DATA_DIR / "Monthly_NDVI_Subcounty_2004_2025_Clean.csv",
    "chirps": DATA_DIR / "CHIRPS_Subcounty_Rainfall_Anomaly_2004_2025.csv",
    "wrua": DATA_DIR / "WRUAs-James.csv",
    "boundaries_shp": DATA_DIR / "subcounty_boundaries.shp",
}
BOUNDARY_COUNTY_COL = "COUNTY"        # attribute column name in your shapefile
BOUNDARY_SUBCOUNTY_COL = "SUBCOUNTY"  # attribute column name in your shapefile

# %% [markdown]
# ## 1. Load raw data
#
# Each file is read as UTF-8 first, with an automatic (and clearly
# logged) fallback to cp1252, then latin-1, only if UTF-8 genuinely
# fails to decode. This never blindly forces one encoding on every file
# — see the note in `read_csv_robust` for why that would be dangerous.

# %%
def read_csv_robust(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError as e:
        log(f"[encoding fallback] {path}: UTF-8 failed ({e}); retrying with cp1252")
        try:
            return pd.read_csv(path, encoding="cp1252", **kwargs)
        except UnicodeDecodeError:
            log(f"[encoding fallback] {path}: cp1252 also failed; using latin-1 "
                f"(lossless but may misrender a few characters -- spot-check "
                f"text fields after loading)")
            return pd.read_csv(path, encoding="latin-1", **kwargs)


log("STAGE 1/6: Loading raw datasets...")
census = read_csv_robust(PATHS["census"])
wra = pd.read_excel(PATHS["wra"])  # Excel handles text encoding internally, no issue here
conflict = read_csv_robust(PATHS["conflict"])
ndvi = read_csv_robust(PATHS["ndvi"])
chirps = read_csv_robust(PATHS["chirps"])
wrua = read_csv_robust(PATHS["wrua"])

print("\nDATASET SIZES LOADED:")
for name, df in [("census (KNBS population)", census), ("wra (water abstraction)", wra),
                  ("conflict (dependent variable)", conflict),
                  ("ndvi (vegetation index)", ndvi), ("chirps (rainfall)", chirps),
                  ("wrua (governance)", wrua)]:
    print(f"  - {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
log("STAGE 1/6: DONE loading raw datasets.\n")

# %% [markdown]
# ## 2. Build the canonical sub-county lookup
#
# This is the master list of "correct" sub-county names for our 4 target
# counties, built from the KNBS census data (the cleanest, most
# authoritative source). Every other dataset's sub-county spellings get
# matched against THIS list in the steps below.

# %%
log("STAGE 2/6: Building canonical sub-county name list from census data...")
lookup = build_canonical_lookup(census, county_col="County", subcounty_col="SubCounty")
print(f"  Canonical sub-county list covers {len(lookup.table)} sub-counties "
      f"across {TARGET_COUNTIES}")
log("STAGE 2/6: DONE.\n")

# %% [markdown]
# ## 3. Match WRA (water abstraction) sub-county names against the canonical list

# %%
log("STAGE 3/6: Matching WRA abstraction records to canonical sub-counties...")
wra_matched = match_dataframe(wra, subcounty_col="Sub_county", lookup=lookup,
                               county_col="County")

unmatched_wra = wra_matched[wra_matched["match_method"] == "unmatched"]
print(f"\n  WRA ABSTRACTION DATA — sub-county name matching results:")
print(f"  Total WRA records: {len(wra_matched):,}")
print(f"  Successfully matched to a known sub-county: "
      f"{len(wra_matched) - len(unmatched_wra):,}")
print(f"  UNMATCHED (could not confidently match): {len(unmatched_wra):,} "
      f"({len(unmatched_wra) / len(wra_matched):.1%})")

if len(unmatched_wra) > 0:
    print(f"\n  These are values from the WRA dataset's own 'Sub_county' column")
    print(f"  that could NOT be automatically matched to any of the standard")
    print(f"  KNBS sub-county names for Nairobi, Kiambu, Machakos, or Turkana.")
    print(f"  For EACH value below, decide: is it (a) a genuine spelling/format")
    print(f"  variant that should be added to MANUAL_ALIASES in")
    print(f"  src/name_cleaning.py, or (b) a real sub-county correctly named")
    print(f"  but missing from your census file (check your census file for it)?")
    print(f"\n  Unmatched 'Sub_county' values and how many WRA rows use each:")
    for val, count in unmatched_wra["Sub_county"].value_counts().items():
        print(f"    - {val!r}: {count} row(s)")

# Recover County where it was blank in the original WRA data, using the match
wra_matched["County_recovered"] = wra_matched["County"].fillna(
    wra_matched["matched_county"]
)
n_before = wra["County"].isna().sum()
n_after = wra_matched["County_recovered"].isna().sum()
print(f"\n  WRA 'County' column: {n_before:,} blank before cleaning -> "
      f"{n_after:,} blank after recovery from matched sub-county names.")

wra_matched.to_csv(OUT_DIR / "wra_cleaned.csv", index=False)
log(f"STAGE 3/6: DONE. Saved -> {OUT_DIR / 'wra_cleaned.csv'}\n")

# %% [markdown]
# ## 4. Filter NDVI and CHIRPS to the four target counties

# %%
log("STAGE 4/6: Filtering NDVI and CHIRPS to the 4 target counties...")
ndvi_filtered = ndvi[ndvi["ADM1_EN"].isin(TARGET_COUNTIES)].copy()
chirps_filtered = chirps[chirps["ADM1_EN"].isin(TARGET_COUNTIES)].copy()

print(f"\n  NDVI (vegetation): {len(ndvi):,} rows nationwide -> "
      f"{len(ndvi_filtered):,} rows in Nairobi/Kiambu/Machakos/Turkana")
print(f"  CHIRPS (rainfall): {len(chirps):,} rows nationwide -> "
      f"{len(chirps_filtered):,} rows in Nairobi/Kiambu/Machakos/Turkana")

ndvi_filtered.to_csv(OUT_DIR / "ndvi_filtered.csv", index=False)
chirps_filtered.to_csv(OUT_DIR / "chirps_filtered.csv", index=False)
log(f"STAGE 4/6: DONE. Saved -> {OUT_DIR / 'ndvi_filtered.csv'}, "
    f"{OUT_DIR / 'chirps_filtered.csv'}\n")

# %% [markdown]
# ## 5. WRUA spatial join against sub-county boundaries
#
# The WRUA dataset covers ALL of Kenya (not just our 4 counties), and it
# has no County column — so we attach County/Sub-county via a real
# point-in-polygon spatial join against the boundary shapefile, using
# each WRUA's centroid coordinates.

# %%
log("STAGE 5/6: Running WRUA spatial join against county boundaries...")
if PATHS["boundaries_shp"].exists():
    boundaries = load_boundaries(
        str(PATHS["boundaries_shp"]),
        county_col=BOUNDARY_COUNTY_COL,
        subcounty_col=BOUNDARY_SUBCOUNTY_COL,
    )
    wrua_joined = join_wrua_to_subcounty(
        wrua, boundaries,
        county_col=BOUNDARY_COUNTY_COL, subcounty_col=BOUNDARY_SUBCOUNTY_COL,
    )
    n_total = len(wrua_joined)
    n_unmatched = wrua_joined["joined_county"].isna().sum()
    n_matched = n_total - n_unmatched

    print(f"\n  WRUA (governance) spatial join results:")
    print(f"  Total WRUA records (nationwide): {n_total:,}")
    print(f"  Centroid falls INSIDE one of our 4 target counties: {n_matched:,}")
    print(f"  Centroid falls OUTSIDE our 4 target counties: {n_unmatched:,} "
          f"({n_unmatched / n_total:.1%})")
    print(f"  NOTE: a high 'outside' percentage here is EXPECTED and not an "
          f"error -- your WRUA dataset covers all 47 Kenyan counties, and we "
          f"only keep the 4 we're studying. What matters is that the 'inside' "
          f"group covers all 4 target counties, checked below:")
    if n_matched > 0:
        print(f"\n  Breakdown of the {n_matched:,} MATCHED WRUA records by county:")
        for county, count in wrua_joined["joined_county"].value_counts().items():
            print(f"    - {county}: {count}")
        missing_counties = set(TARGET_COUNTIES) - set(wrua_joined["joined_county"].dropna().unique())
        if missing_counties:
            print(f"\n  WARNING: no matched WRUAs at all in: {sorted(missing_counties)} "
                  f"-- worth checking whether this county genuinely has no WRUAs, "
                  f"or whether the boundary/CRS is off for it.")

    wrua_joined.to_csv(OUT_DIR / "wrua_joined.csv", index=False)
    log(f"STAGE 5/6: DONE. Saved -> {OUT_DIR / 'wrua_joined.csv'}\n")
else:
    print(f"\n  Shapefile not found at {PATHS['boundaries_shp']} -- update the "
          f"path in CONFIG. Skipping WRUA join for now.")
    wrua_joined = None
    boundaries = None
    log("STAGE 5/6: SKIPPED (no shapefile found).\n")

# %% [markdown]
# ## 6. Visualizations — data quality overview + WRUA spatial join map

# %%
log("STAGE 6/6: Generating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

match_counts = wra_matched["match_method"].value_counts()
axes[0].bar(match_counts.index, match_counts.values,
            color=["#2a9d8f", "#e9c46a", "#e76f51"][:len(match_counts)])
axes[0].set_title("WRA sub-county name match quality")
axes[0].set_ylabel("Number of records")
for i, v in enumerate(match_counts.values):
    axes[0].text(i, v + 0.05, str(v), ha="center")

county_counts = conflict["County"].value_counts().reindex(TARGET_COUNTIES).fillna(0)
axes[1].bar(county_counts.index, county_counts.values, color="#264653")
axes[1].set_title("Conflict records per county")
axes[1].set_ylabel("Record count")

plt.tight_layout()
save_and_display(fig, OUT_DIR / "phase1_data_quality_overview.png")

if wrua_joined is not None:
    fig, ax = plt.subplots(figsize=(7, 7))
    boundaries.plot(ax=ax, color="#f4f1de", edgecolor="#333", linewidth=0.6)
    matched_pts = wrua_joined.dropna(subset=["joined_county"])
    unmatched_pts = wrua_joined[wrua_joined["joined_county"].isna()]
    ax.scatter(matched_pts["CentroidX"], matched_pts["CentroidY"],
               c="#2a9d8f", s=25, label="Matched to a target county")
    ax.scatter(unmatched_pts["CentroidX"], unmatched_pts["CentroidY"],
               c="#e76f51", s=10, alpha=0.4, label="Outside target counties")
    ax.set_title("WRUA centroids vs. target county boundaries")
    ax.legend()
    plt.tight_layout()
    save_and_display(fig, OUT_DIR / "phase1_wrua_spatial_join.png")
else:
    print("  Skipped WRUA map -- no shapefile join available yet.")

log("STAGE 6/6: DONE.\n")

# %% [markdown]
# ## Summary

# %%
elapsed = time.time() - _SCRIPT_START
print("=" * 60)
print("ALL PHASES COMPLETE")
print(f"Total runtime: {elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. wra_cleaned.csv              - WRA data with matched/recovered County
                                     + Sub-county, plus a match_method
                                     column flagging every row's match
                                     quality (exact / fuzzy / unmatched)
  2. ndvi_filtered.csv             - NDVI filtered to the 4 target counties
  3. chirps_filtered.csv           - CHIRPS rainfall filtered to the 4 counties
  4. wrua_joined.csv               - WRUA records with County/Sub-county
                                     attached via spatial join
  5. phase1_data_quality_overview.png - match-quality + conflict-count chart
  6. phase1_wrua_spatial_join.png     - WRUA centroids vs. county boundaries map

If you see all 6 files above in your outputs/ folder and this banner
printed, Phase 1 ran to completion -- nothing is still running.
""")
