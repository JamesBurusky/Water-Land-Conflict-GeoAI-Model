# %% [markdown]
# # Phase 8 — Conflict Risk Mapping
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Turns the trained model (Phase 6) into an actual sub-county risk map
# -- Low/Medium/High categories plus the underlying probability -- and
# exports it in a form the Phase 9 dashboard can load directly.
#
# Run this after 10_ml_modeling.py (needs outputs/xgboost_model.joblib).

# %%
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from ml_prep import prepare_modelling_table
from spatial_join import load_boundaries
from risk_mapping import build_risk_layer

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PANEL_PATH = OUT_DIR / "ml_panel.csv"
MODEL_PATH = OUT_DIR / "xgboost_model.joblib"
BOUNDARIES_SHP = DATA_DIR / "subcounty_boundaries.shp"
BOUNDARY_COUNTY_COL = "COUNTY"
BOUNDARY_SUBCOUNTY_COL = "SUBCOUNTY"

LAG_MONTHS = 1  # MUST match 10_ml_modeling.py

timer = Timer()
timer.__enter__()
log("=== PHASE 8: CONFLICT RISK MAPPING STARTED ===")

# %% [markdown]
# ## 1. Load model, panel, and boundaries

# %%
log("STAGE 1/3: Loading model, panel, and boundaries...")
if not MODEL_PATH.exists():
    print(f"  {MODEL_PATH} not found -- run 10_ml_modeling.py first.")
    raise SystemExit(1)

saved = joblib.load(MODEL_PATH)
model, feature_cols = saved["model"], saved["feature_cols"]

panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
table, table_feature_cols = prepare_modelling_table(panel, lag_months=LAG_MONTHS)

if set(feature_cols) != set(table_feature_cols):
    print(f"\n  WARNING: features in the saved model don't match the panel's "
          f"current features -- re-run 10_ml_modeling.py to keep them in sync "
          f"before trusting this risk map.")

boundaries = load_boundaries(str(BOUNDARIES_SHP), county_col=BOUNDARY_COUNTY_COL,
                              subcounty_col=BOUNDARY_SUBCOUNTY_COL)
boundaries = boundaries.rename(columns={BOUNDARY_COUNTY_COL: "County",
                                         BOUNDARY_SUBCOUNTY_COL: "SubCounty"})

if len(table) == 0:
    print(f"\n  STOPPED: 0 rows in the modelling table after lagging/dropping "
          f"missing values. This is almost always an entirely-NaN feature "
          f"column (e.g. NDVI/CHIRPS coverage gap) -- check {PANEL_PATH} "
          f"column-by-column before re-running.")
    raise SystemExit(1)

log("STAGE 1/3: DONE.\n")

# %% [markdown]
# ## 2. Predict and categorize risk per sub-county
#
# Uses each sub-county's MOST RECENT complete feature row -- i.e. "what's
# the predicted risk of onset NEXT month, given the latest data we have".
# Risk category (Low/Medium/High) is assigned by RELATIVE tertiles across
# sub-counties, not fixed probability thresholds -- see src/risk_mapping.py
# for why (conflict onset is rare overall, so fixed thresholds calibrated
# for a balanced problem would call almost everywhere "Low" and hide real
# relative differences).

# %%
log("STAGE 2/3: Predicting and categorizing risk...")
layer = build_risk_layer(model, table, feature_cols, boundaries)

print("\n  Risk category counts:")
print(layer["risk_category"].value_counts(dropna=False).to_string())
print("\n  Top 5 highest-risk sub-counties:")
print(layer.dropna(subset=["risk_probability"]).nlargest(5, "risk_probability")
      [["County", "SubCounty", "risk_probability", "risk_category"]].to_string(index=False))

# Dashboard-ready exports: GeoJSON for anything that renders a map,
# plain CSV for anything that just wants the numbers.
layer.to_file(OUT_DIR / "conflict_risk_layer.geojson", driver="GeoJSON")
layer.drop(columns="geometry").to_csv(OUT_DIR / "conflict_risk_layer.csv", index=False)
log(f"STAGE 2/3: DONE. Saved -> {OUT_DIR / 'conflict_risk_layer.geojson'}, "
    f"{OUT_DIR / 'conflict_risk_layer.csv'}\n")

# %% [markdown]
# ## 3. Choropleth map

# %%
log("STAGE 3/3: Generating risk map...")

fig, ax = plt.subplots(figsize=(8, 8))
category_colors = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51",
                    "Lower": "#2a9d8f", "Higher": "#e76f51"}
# risk_category is a pandas Categorical (from qcut) -- fillna() on a
# Categorical rejects any value not already in its defined categories,
# so convert to plain string first rather than trying to add "No data"
# as a category just for this one plotting step.
risk_cat_str = layer["risk_category"].astype(object).fillna("No data")
layer.plot(
    ax=ax, color=[category_colors.get(c, "#dddddd") for c in risk_cat_str],
    edgecolor="#333", linewidth=0.5,
)
# Passing an explicit color list (needed for fixed, run-independent
# category colors) bypasses geopandas' automatic legend -- build one
# manually so the map is never ambiguous about what each color means.
import matplotlib.patches as mpatches
present_categories = [c for c in risk_cat_str.unique() if c != "No data"]
legend_handles = [mpatches.Patch(color=category_colors.get(c, "#dddddd"), label=c)
                   for c in present_categories]
if "No data" in risk_cat_str.unique():
    legend_handles.append(mpatches.Patch(color="#dddddd", label="No data"))
ax.legend(handles=legend_handles, loc="lower left")
ax.set_title("Conflict risk by sub-county (predicted, next month)")
ax.set_axis_off()
plt.tight_layout()
save_and_display(fig, OUT_DIR / "risk_map.png")

log("STAGE 3/3: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 8 COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. conflict_risk_layer.geojson - full risk layer with geometry, for Phase 9's dashboard
  2. conflict_risk_layer.csv - same data without geometry
  3. risk_map.png - choropleth visualization

REVIEW before writing up / handing to the dashboard:
  - Risk here is "next month, given the latest available data" -- state
    this explicitly in your thesis; it is NOT a long-range forecast.
  - Sub-counties with "No data" on the map have no complete recent feature
    history (e.g. an NDVI/rainfall gap) -- worth checking whether that's a
    genuine data gap or something fixable before presenting the map.
  - The tertile-based Low/Medium/High split is RELATIVE to this run's set
    of sub-counties -- re-running with more/fewer sub-counties, or after
    the underlying probabilities shift over time, will shift where the
    category boundaries fall. Report the underlying risk_probability
    alongside the category for anything precision-sensitive.
""")
