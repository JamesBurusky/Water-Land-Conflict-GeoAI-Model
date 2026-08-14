# %% [markdown]
# # Phase 8 — Conflict Risk Mapping (multi-horizon)
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Turns each of the trained models (Phase 6, one per forecast horizon)
# into an actual sub-county risk map -- Low/Medium/High categories plus
# the underlying probability -- exported per horizon so the dashboard
# can let a user step through 3-to-24-month-ahead predictions.
#
# Run this after 10_ml_modeling.py.

# %%
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from ml_prep import prepare_modelling_table
from spatial_join import load_boundaries
from risk_mapping import build_risk_layer
from output_paths import step_dir, horizon_subdir

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
STEP_NAME = "12_conflict_risk_mapping"

PANEL_PATH = step_dir("08_nlp_panel_features") / "ml_panel.csv"
BOUNDARIES_SHP = DATA_DIR / "subcounty_boundaries.shp"
BOUNDARY_COUNTY_COL = "COUNTY"
BOUNDARY_SUBCOUNTY_COL = "SUBCOUNTY"

HORIZONS_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]  # MUST match 10_ml_modeling.py's HORIZONS_MONTHS

CATEGORY_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51",
                    "Lower": "#2a9d8f", "Higher": "#e76f51"}

timer = Timer()
timer.__enter__()
log("=== PHASE 8: MULTI-HORIZON CONFLICT RISK MAPPING STARTED ===")

# %% [markdown]
# ## Load panel and boundaries once, reused for every horizon

# %%
log("Loading panel and boundaries...")
panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
boundaries = load_boundaries(str(BOUNDARIES_SHP), county_col=BOUNDARY_COUNTY_COL,
                              subcounty_col=BOUNDARY_SUBCOUNTY_COL)
boundaries = boundaries.rename(columns={BOUNDARY_COUNTY_COL: "County",
                                         BOUNDARY_SUBCOUNTY_COL: "SubCounty"})


def run_one_horizon(horizon_months: int) -> None:
    """Predicts, categorizes, and maps risk for ONE forecast horizon,
    using that horizon's own trained XGBoost model from 10_ml_modeling.py."""
    out_dir = horizon_subdir(STEP_NAME, horizon_months)
    model_path = horizon_subdir("10_ml_modeling", horizon_months) / "xgboost_model.joblib"
    log(f"--- HORIZON: {horizon_months} month(s) ahead -> {out_dir} ---")

    if not model_path.exists():
        print(f"  SKIPPED horizon={horizon_months}: {model_path} not found -- "
              f"run 10_ml_modeling.py first.")
        return

    saved = joblib.load(model_path)
    model, feature_cols = saved["model"], saved["feature_cols"]

    table, table_feature_cols = prepare_modelling_table(panel, lag_months=horizon_months)
    if set(feature_cols) != set(table_feature_cols):
        print(f"  WARNING: features in the saved model don't match the panel's current "
              f"features for horizon={horizon_months} -- re-run 10_ml_modeling.py to keep "
              f"them in sync before trusting this risk map.")

    if len(table) == 0:
        print(f"  SKIPPED horizon={horizon_months}: 0 rows in the modelling table after "
              f"lagging/dropping missing values (likely an entirely-NaN feature column) -- "
              f"check {PANEL_PATH} column-by-column before re-running.")
        return

    layer = build_risk_layer(model, table, feature_cols, boundaries)

    print(f"\n  Risk category counts (horizon={horizon_months}mo):")
    print(layer["risk_category"].value_counts(dropna=False).to_string())
    if layer["risk_probability"].notna().any():
        print(f"\n  Top 5 highest-risk sub-counties:")
        print(layer.dropna(subset=["risk_probability"]).nlargest(5, "risk_probability")
              [["County", "SubCounty", "risk_probability", "risk_category"]].to_string(index=False))

    layer.to_file(out_dir / "conflict_risk_layer.geojson", driver="GeoJSON")
    layer.drop(columns="geometry").to_csv(out_dir / "conflict_risk_layer.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    risk_cat_str = layer["risk_category"].astype(object).fillna("No data")
    layer.plot(
        ax=ax, color=[CATEGORY_COLORS.get(c, "#dddddd") for c in risk_cat_str],
        edgecolor="#333", linewidth=0.5,
    )
    present_categories = [c for c in risk_cat_str.unique() if c != "No data"]
    legend_handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#dddddd"), label=c)
                       for c in present_categories]
    if "No data" in risk_cat_str.unique():
        legend_handles.append(mpatches.Patch(color="#dddddd", label="No data"))
    ax.legend(handles=legend_handles, loc="lower left")
    ax.set_title(f"Conflict risk by sub-county - {horizon_months} month(s) ahead")
    ax.set_axis_off()
    plt.tight_layout()
    save_and_display(fig, out_dir / "risk_map.png")

    log(f"--- HORIZON {horizon_months} month(s): DONE. Saved -> "
        f"{out_dir / 'conflict_risk_layer.geojson'} ---\n")


# %% [markdown]
# ## Run every horizon

# %%
for h in HORIZONS_MONTHS:
    run_one_horizon(h)

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 8 (MULTI-HORIZON) COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
For each horizon in {HORIZONS_MONTHS}, outputs/{STEP_NAME}/horizon_<N>month/ contains:
  1. conflict_risk_layer.geojson - full risk layer with geometry, for the dashboard
  2. conflict_risk_layer.csv - same data without geometry
  3. risk_map.png - choropleth visualization

REVIEW before writing up / handing to the dashboard:
  - "1 month ahead" is the closest thing to a validated short-term
    forecast here; 3- and 6-month horizons are provided for comparison
    but check outputs/10_ml_modeling/horizon_comparison_summary.csv --
    if their metrics are notably worse, say so explicitly rather than
    presenting all three as equally trustworthy.
  - Sub-counties with "No data" have no complete recent feature history
    (e.g. an NDVI/rainfall gap) -- worth checking whether that's genuine
    or fixable.
  - The tertile-based Low/Medium/High split is RELATIVE to each horizon's
    own set of predictions -- a sub-county's category can differ between
    horizons even if you don't expect it to; that's expected given the
    relative (not fixed-threshold) categorization, not a bug.
""")
