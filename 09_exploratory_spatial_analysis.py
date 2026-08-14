# %% [markdown]
# # Phase 4 — Exploratory Spatial Analysis
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Hotspot analysis (Getis-Ord Gi*), Kernel Density Estimation, DBSCAN
# clustering, temporal trend analysis, and seasonal analysis — answers
# the spec's core questions: where do conflicts occur, when, and what
# factors appear associated.
#
# Run this after 06_spatial_feature_engineering.py (needs ml_panel.csv)
# and 02_conflict_cleaning.py (needs conflict_cleaned.csv for raw
# point coordinates).

# %%
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from spatial_join import load_boundaries
from spatial_analysis import (
    aggregate_subcounty_summary, compute_getis_ord,
    compute_kde_surface, compute_dbscan_clusters,
)
from temporal_analysis import yearly_trend, seasonal_pattern, county_yearly_trend
from output_paths import step_dir

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = step_dir("09_exploratory_spatial_analysis")

PANEL_PATH = step_dir("08_nlp_panel_features") / "ml_panel.csv"
CONFLICT_CLEANED = step_dir("03_deduplication_check") / "conflict_cleaned.csv"
BOUNDARIES_SHP = DATA_DIR / "subcounty_boundaries.shp"
BOUNDARY_COUNTY_COL = "COUNTY"
BOUNDARY_SUBCOUNTY_COL = "SUBCOUNTY"

HOTSPOT_VALUE_COL = "total_onset_months"  # which subcounty_summary column to test for hot/coldspots
K_NEIGHBORS = 4       # KNN weights -- see src/spatial_analysis.py for why not contiguity
DBSCAN_EPS_KM = 25    # tunable -- distance (km) within which points count as "close"
DBSCAN_MIN_SAMPLES = 3

timer = Timer()
timer.__enter__()
log("=== PHASE 4: EXPLORATORY SPATIAL ANALYSIS STARTED ===")

# %% [markdown]
# ## 1. Load data

# %%
log("STAGE 1/5: Loading panel and conflict point data...")
panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
conflict = pd.read_csv(CONFLICT_CLEANED)
boundaries = load_boundaries(str(BOUNDARIES_SHP), county_col=BOUNDARY_COUNTY_COL,
                              subcounty_col=BOUNDARY_SUBCOUNTY_COL)
boundaries = boundaries.rename(columns={BOUNDARY_COUNTY_COL: "County",
                                         BOUNDARY_SUBCOUNTY_COL: "SubCounty"})
print(f"  Panel: {len(panel):,} rows | Conflict records: {len(conflict):,} | "
      f"Boundary sub-counties: {len(boundaries)}")
log("STAGE 1/5: DONE.\n")

# %% [markdown]
# ## 2. Getis-Ord Gi* hotspot analysis
#
# Uses KNN spatial weights (not Queen contiguity) since Turkana is
# geographically isolated from the other 3 counties — see the note in
# src/spatial_analysis.py for why contiguity weights would leave it
# with zero neighbors.

# %%
log("STAGE 2/5: Running Getis-Ord Gi* hotspot analysis...")
summary = aggregate_subcounty_summary(panel)
try:
    gi_result = compute_getis_ord(summary, boundaries, value_col=HOTSPOT_VALUE_COL,
                                    k_neighbors=K_NEIGHBORS)
    print(f"\n  Hotspot categories for '{HOTSPOT_VALUE_COL}':")
    print(gi_result["hotspot_category"].value_counts().to_string())
    gi_result.drop(columns="geometry").to_csv(OUT_DIR / "hotspot_analysis.csv", index=False)
    print(f"  Saved -> {OUT_DIR / 'hotspot_analysis.csv'}")
except ValueError as e:
    print(f"  SKIPPED: {e}")
    gi_result = None
log("STAGE 2/5: DONE.\n")

# %% [markdown]
# ## 3. Kernel Density Estimation + DBSCAN clustering
#
# Both run on raw conflict point coordinates (Latitude/Longitude),
# not sub-county aggregates -- these methods work at the individual-
# incident level to reveal fine-grained spatial concentration that
# sub-county boundaries could mask (a hotspot near a boundary edge, or
# multiple distinct clusters within one large sub-county like Turkana
# South).

# %%
log("STAGE 3/5: Computing KDE surface and DBSCAN clusters...")
lons = conflict["Longitude"].to_numpy()
lats = conflict["Latitude"].to_numpy()

X, Y, Z = compute_kde_surface(lons, lats)
cluster_labels = compute_dbscan_clusters(lons, lats, eps_km=DBSCAN_EPS_KM,
                                          min_samples=DBSCAN_MIN_SAMPLES)
conflict["dbscan_cluster"] = cluster_labels

n_clusters = len(set(cluster_labels) - {-1, -2})
n_noise = (cluster_labels == -1).sum()
print(f"  DBSCAN: {n_clusters} cluster(s) found, {n_noise} point(s) marked as noise "
      f"(too isolated to belong to any cluster at eps={DBSCAN_EPS_KM}km, "
      f"min_samples={DBSCAN_MIN_SAMPLES} -- adjust these if this looks off)")

conflict.to_csv(OUT_DIR / "conflict_with_clusters.csv", index=False)
log(f"STAGE 3/5: DONE. Saved -> {OUT_DIR / 'conflict_with_clusters.csv'}\n")

# %% [markdown]
# ## 4. Temporal trend and seasonal analysis

# %%
log("STAGE 4/5: Computing temporal trend and seasonal patterns...")
trend = yearly_trend(panel, value_col="conflict_onset", agg="sum")
print(f"\n  Yearly trend (total onset months/year): slope={trend.attrs['trend_slope']:.3f}, "
      f"p-value={trend.attrs['trend_pvalue']:.4f}, R²={trend.attrs['trend_r_squared']:.3f}")
if trend.attrs["trend_pvalue"] < 0.05:
    direction = "increasing" if trend.attrs["trend_slope"] > 0 else "decreasing"
    print(f"  -> Statistically significant {direction} trend (p < 0.05)")
else:
    print(f"  -> No statistically significant year-over-year trend (p >= 0.05) -- "
          f"don't claim a trend in your write-up without noting this")

seasonal = seasonal_pattern(panel, value_col="conflict_onset", agg="sum")
county_trend = county_yearly_trend(panel, value_col="conflict_onset", agg="sum")

trend.to_csv(OUT_DIR / "yearly_trend.csv", index=False)
seasonal.to_csv(OUT_DIR / "seasonal_pattern.csv", index=False)
log(f"STAGE 4/5: DONE. Saved -> {OUT_DIR / 'yearly_trend.csv'}, "
    f"{OUT_DIR / 'seasonal_pattern.csv'}\n")

# %% [markdown]
# ## 5. Visualizations

# %%
log("STAGE 5/5: Generating visualizations...")

# 5a. Hotspot map
if gi_result is not None:
    fig, ax = plt.subplots(figsize=(7, 7))
    boundaries.plot(ax=ax, color="#eee", edgecolor="#999", linewidth=0.5)
    gi_result.plot(column="gi_zscore", cmap="RdBu_r", legend=True, ax=ax,
                    edgecolor="#333", linewidth=0.5)
    ax.set_title(f"Getis-Ord Gi* hotspots — {HOTSPOT_VALUE_COL}")
    plt.tight_layout()
    save_and_display(fig, OUT_DIR / "hotspot_map.png")

# 5b. KDE heatmap + DBSCAN clusters
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].contourf(X, Y, Z, levels=20, cmap="YlOrRd")
axes[0].scatter(lons, lats, c="black", s=8, alpha=0.6)
axes[0].set_title("Conflict density (KDE)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

scatter = axes[1].scatter(lons, lats, c=cluster_labels, cmap="tab10", s=30)
axes[1].set_title(f"DBSCAN clusters (eps={DBSCAN_EPS_KM}km, min_samples={DBSCAN_MIN_SAMPLES})")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(scatter, ax=axes[1], label="Cluster ID (-1/-2 = noise)")

plt.tight_layout()
save_and_display(fig, OUT_DIR / "kde_dbscan.png")

# 5c. Temporal trend + seasonal pattern
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(trend["Year"], trend["conflict_onset"], marker="o")
z = np.polyfit(trend["Year"], trend["conflict_onset"], 1)
axes[0].plot(trend["Year"], np.poly1d(z)(trend["Year"]), "r--", alpha=0.7, label="linear trend")
axes[0].set_title("Conflict onset months per year")
axes[0].set_xlabel("Year"); axes[0].legend()

axes[1].bar(seasonal["Month"], seasonal["conflict_onset_total"], color="#2a9d8f")
axes[1].set_title("Conflict onset by calendar month (all years combined)")
axes[1].set_xlabel("Month"); axes[1].set_xticks(range(1, 13))

plt.tight_layout()
save_and_display(fig, OUT_DIR / "temporal_seasonal.png")

log("STAGE 5/5: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 4 COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. hotspot_analysis.csv / hotspot_map.png  - Getis-Ord Gi* results
  2. conflict_with_clusters.csv / kde_dbscan.png - KDE + DBSCAN
  3. yearly_trend.csv, seasonal_pattern.csv / temporal_seasonal.png

REVIEW before writing up:
  - HOTSPOT_VALUE_COL is currently '{HOTSPOT_VALUE_COL}' -- worth re-running
    with 'conflict_persistence' or 'conflict_severity_weighted' too and
    comparing whether the same sub-counties come up hot/cold.
  - DBSCAN_EPS_KM ({DBSCAN_EPS_KM}) and min_samples ({DBSCAN_MIN_SAMPLES}) are
    tunable -- try a couple of values and report whichever produces the
    most interpretable clusters, noting the choice in your methodology.
  - K_NEIGHBORS ({K_NEIGHBORS}) for Getis-Ord affects Turkana's neighbors
    specifically (see the module docstring) -- worth a one-line
    limitation note in your methodology given the non-contiguous study area.
""")
