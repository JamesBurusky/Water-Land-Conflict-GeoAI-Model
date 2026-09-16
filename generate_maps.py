"""
generate_maps.py

Standalone script -- NOT part of the numbered pipeline. Generates the
spatial result figures Chapter Four is currently missing: a hotspot
choropleth, a predictive risk map, a DBSCAN point-cluster map, and a
raw conflict-point map, all built from your own real outputs.

Run this from your project root (the same folder that contains data/
and outputs/), after the full pipeline (through at least stage 12) has
been run.

Requirements (install once):
    pip install geopandas matplotlib pandas scikit-learn --break-system-packages

Usage:
    python generate_maps.py

Each map is generated independently with its own try/except: if one
input file is missing, that map is skipped with a clear message and
the script continues to the others, rather than the whole run failing
because of one missing file.

Outputs are written to outputs/14_chapter4_figures/, ready to drop
into Chapter Four. Every map includes a north arrow, a scale bar, a
visible map frame, and a legend placed entirely outside the map's
plotted area (never overlapping the map itself, including the small,
tightly-clustered Nairobi/Kiambu/Machakos area).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths -- adjust these if your folder layout differs from the pipeline's
# established structure.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "14_chapter4_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_PATH = DATA_DIR / "subcounty_boundaries.shp"
HOTSPOT_PATH = OUTPUTS_DIR / "09_exploratory_spatial_analysis" / "hotspot_analysis.csv"
CONFLICT_CLEANED_PATH = OUTPUTS_DIR / "03_deduplication_check" / "conflict_cleaned.csv"
RISK_LAYER_DIR = OUTPUTS_DIR / "12_conflict_risk_mapping"
RISK_LAYER_HORIZON_MONTHS = 3  # which horizon's risk layer to map; change to map a different one

FONT = "serif"


def log(msg):
    print(f"  {msg}")


def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not read {path} with utf-8, cp1252, or latin-1.")


def load_boundaries():
    if not BOUNDARY_PATH.exists():
        log(f"SKIPPED (boundary file not found at {BOUNDARY_PATH} -- "
            f"edit BOUNDARY_PATH at the top of this script if it lives elsewhere)")
        return None
    gdf = gpd.read_file(BOUNDARY_PATH)
    if not gdf.geometry.is_valid.all():
        gdf["geometry"] = gdf.geometry.buffer(0)
    for col in ("COUNTY", "SUBCOUNTY"):
        if col not in gdf.columns:
            log(f"SKIPPED (boundary file has no '{col}' column -- found: {list(gdf.columns)})")
            return None
    gdf["_join_key"] = (gdf["SUBCOUNTY"].astype(str).str.strip().str.lower()
                         + "|" + gdf["COUNTY"].astype(str).str.strip().str.lower())
    return gdf


def find_risk_layer():
    """Looks in the exact documented location first (outputs/12_.../horizon_Nmonth/),
    then falls back to a recursive search anywhere under the risk-mapping
    output folder, so a naming variation doesn't silently produce nothing."""
    specific = RISK_LAYER_DIR / f"horizon_{RISK_LAYER_HORIZON_MONTHS}month"
    if specific.exists():
        hits = list(specific.glob("*.geojson"))
        if hits:
            return hits[0]
    hits = list(RISK_LAYER_DIR.rglob("*.geojson"))
    hits = [h for h in hits if f"{RISK_LAYER_HORIZON_MONTHS}month" in str(h)] or hits
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Shared cartographic elements -- every map gets the same treatment, so a
# map frame, north arrow, and scale bar are never forgotten on any of them.
# ---------------------------------------------------------------------------
def finalise_map(fig, ax, gdf_for_extent, title, legend_handles=None, legend_title=None):
    """Applies a consistent set of map elements: a visible frame (kept, not
    switched off), axis tick labels showing lat/lon, a north arrow, a scale
    bar, and -- critically -- a legend placed in reserved figure space
    OUTSIDE the plotted map extent, never overlapping the map itself."""
    minx, miny, maxx, maxy = gdf_for_extent.total_bounds
    pad_x = (maxx - minx) * 0.15
    pad_y = (maxy - miny) * 0.15
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # Keep the frame visible (a plain ax.set_axis_off() would remove it
    # entirely, which is the "no map frame" problem being fixed here).
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    ax.set_xlabel("Longitude", family=FONT, fontsize=9)
    ax.set_ylabel("Latitude", family=FONT, fontsize=9)
    ax.tick_params(labelsize=7.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family(FONT)

    # North arrow -- placed using AXES-FRACTION coordinates (0-1, fixed
    # relative to the plot frame), not data coordinates. This is the
    # actual fix for the overlap bug: data-coordinate placement near
    # "top-left" put the arrow directly on top of Turkana, since Turkana
    # genuinely IS in the top-left of these maps. Axes-fraction placement
    # guarantees a fixed screen position in the corner regardless of
    # where the real geographic data happens to fall.
    ax.annotate("N", xy=(0.035, 0.95), xytext=(0.035, 0.80),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", fontsize=11, family=FONT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", linewidth=1.3, color="black"),
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75))

    # Scale bar -- anchored in axes-fraction space (bottom-left corner)
    # for the same reason, while its LENGTH is still computed correctly
    # in real kilometres via the local degrees-to-km conversion.
    mean_lat = (miny + maxy) / 2
    km_per_deg_lon = 111.32 * np.cos(np.radians(mean_lat))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    bar_km = 50 if (xlim[1] - xlim[0]) < 3 else 100
    bar_len_deg = bar_km / km_per_deg_lon
    bar_x0 = xlim[0] + 0.035 * (xlim[1] - xlim[0])
    bar_y = ylim[0] + 0.035 * (ylim[1] - ylim[0])
    tick_h = 0.008 * (ylim[1] - ylim[0])
    ax.plot([bar_x0, bar_x0 + bar_len_deg], [bar_y, bar_y], color="black", linewidth=2.2, solid_capstyle="butt")
    for xe in (bar_x0, bar_x0 + bar_len_deg):
        ax.plot([xe, xe], [bar_y - tick_h, bar_y + tick_h], color="black", linewidth=1.3)
    ax.text(bar_x0 + bar_len_deg / 2, bar_y + tick_h * 2.2, f"{bar_km} km",
            ha="center", va="bottom", fontsize=7.5, family=FONT,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

    ax.set_title(title, family=FONT, fontsize=12, pad=10)

    # Legend in its own reserved margin to the right of the map -- NEVER
    # inside the axes, so it cannot sit on top of Nairobi/Kiambu/Machakos
    # or any other feature regardless of where they fall in the frame.
    if legend_handles:
        fig.subplots_adjust(right=0.72)
        ax.legend(handles=legend_handles, title=legend_title, loc="center left",
                  bbox_to_anchor=(1.04, 0.5), fontsize=8.5, title_fontsize=9.5,
                  prop={"family": FONT}, frameon=True, edgecolor="black")


# ---------------------------------------------------------------------------
# Map 1: Hotspot choropleth (Getis-Ord Gi*) -- diverging red/blue colour
# scheme, the standard convention for hot/cold spot mapping.
# ---------------------------------------------------------------------------
def make_hotspot_map(boundaries):
    if boundaries is None:
        return
    if not HOTSPOT_PATH.exists():
        log(f"SKIPPED hotspot map ({HOTSPOT_PATH} not found)")
        return
    try:
        hs = read_csv_robust(HOTSPOT_PATH)
        subcounty_col = next((c for c in ["SubCounty", "SUBCOUNTY", "sub_county"] if c in hs.columns), None)
        county_col = next((c for c in ["County_x", "County", "COUNTY", "County_y"] if c in hs.columns), None)
        cat_col = next((c for c in ["hotspot_category", "classification"] if c in hs.columns), None)
        if not (subcounty_col and county_col and cat_col):
            log(f"SKIPPED hotspot map (expected columns not found -- has: {list(hs.columns)})")
            return

        hs["_join_key"] = (hs[subcounty_col].astype(str).str.strip().str.lower()
                            + "|" + hs[county_col].astype(str).str.strip().str.lower())
        merged = boundaries.merge(hs[["_join_key", cat_col]], on="_join_key", how="left")
        n_matched = merged[cat_col].notna().sum()
        log(f"Hotspot map: matched {n_matched} / {len(hs)} hotspot_analysis.csv rows to boundary sub-counties")
        if n_matched == 0:
            log("SKIPPED hotspot map (no rows matched -- check SUBCOUNTY/COUNTY spelling agreement)")
            return

        # Diverging colour scheme: red family = hotspot, blue family =
        # coldspot, neutral grey = not significant/outside scope.
        style_map = {
            "Hotspot (99% confidence)": dict(color="#7f0000", label="Hotspot (99%)"),
            "Hotspot (90% confidence)": dict(color="#e34a33", label="Hotspot (90%)"),
            "Not significant": dict(color="#f0f0f0", label="Not significant"),
            "Coldspot (90% confidence)": dict(color="#a6bddb", label="Coldspot (90%)"),
            "Coldspot (95% confidence)": dict(color="#3690c0", label="Coldspot (95%)"),
            "Coldspot (99% confidence)": dict(color="#034e7b", label="Coldspot (99%)"),
        }

        fig, ax = plt.subplots(figsize=(7.8, 8.6))
        boundaries.plot(ax=ax, facecolor="#fafafa", edgecolor="#bbbbbb", linewidth=0.4)  # outside-scope context
        for cat, style in style_map.items():
            sub = merged[merged[cat_col] == cat]
            if len(sub) == 0:
                continue
            sub.plot(ax=ax, facecolor=style["color"], edgecolor="black", linewidth=0.7)
        merged.dissolve().boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

        legend_handles = [mpatches.Patch(facecolor=s["color"], edgecolor="black", label=s["label"])
                           for s in style_map.values()]
        finalise_map(fig, ax, boundaries, "Getis-Ord Gi* Hotspot Classification by Sub-County",
                     legend_handles, "Classification")
        out_path = FIG_DIR / "map_hotspot_classification.png"
        plt.savefig(out_path, dpi=230, facecolor="white", bbox_inches="tight")
        plt.close()
        log(f"SAVED {out_path}")
    except Exception as e:
        log(f"SKIPPED hotspot map (error: {e})")


# ---------------------------------------------------------------------------
# Map 2: Predictive risk map -- sequential yellow-orange-red colour scheme,
# the standard convention for risk/hazard intensity.
# ---------------------------------------------------------------------------
def make_risk_map(boundaries):
    risk_path = find_risk_layer()
    if risk_path is None:
        log(f"SKIPPED risk map (no .geojson found under {RISK_LAYER_DIR} or its horizon_*month subfolders)")
        return
    try:
        risk = gpd.read_file(risk_path)
        risk_col = next((c for c in ["risk_probability", "risk_score", "predicted_probability", "probability"]
                          if c in risk.columns), None)
        if risk_col is None:
            log(f"SKIPPED risk map (no recognisable risk column in {risk_path.name} -- has: {list(risk.columns)})")
            return

        fig, ax = plt.subplots(figsize=(7.8, 8.6))
        if boundaries is not None:
            boundaries.plot(ax=ax, facecolor="#fafafa", edgecolor="#bbbbbb", linewidth=0.4)
        plot = risk.plot(column=risk_col, ax=ax, cmap="YlOrRd", edgecolor="black", linewidth=0.5)
        finalise_map(fig, ax, risk, f"Predicted Conflict Risk by Sub-County ({RISK_LAYER_HORIZON_MONTHS}-Month Horizon)")
        fig.subplots_adjust(right=0.72)  # reserve the same right margin used for legends elsewhere,
                                          # so the colorbar added below doesn't overlap the map

        # Colorbar added as its own explicitly-positioned axes, added
        # AFTER finalise_map has already fixed the main map's extent and
        # title -- geopandas' built-in legend=True mechanism was found to
        # conflict with the explicit xlim/ylim calls above, leaving the
        # title floating with a large empty gap above the actual map.
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(
            vmin=risk[risk_col].min(), vmax=risk[risk_col].max()))
        sm._A = []
        fig.canvas.draw()  # forces matplotlib to resolve the equal-aspect axes
                            # position before we read it, so the colorbar below
                            # is anchored to where the map ACTUALLY ended up,
                            # not a hardcoded guess that assumes a specific
                            # aspect ratio.
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.76, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Predicted risk", family=FONT, fontsize=9)
        cbar.ax.tick_params(labelsize=7.5)
        out_path = FIG_DIR / "map_predicted_risk.png"
        plt.savefig(out_path, dpi=230, facecolor="white", bbox_inches="tight")
        plt.close()
        log(f"SAVED {out_path} (source: {risk_path})")
    except Exception as e:
        log(f"SKIPPED risk map (error: {e})")


# ---------------------------------------------------------------------------
# Map 3: DBSCAN point-cluster map -- recomputed directly with the same
# parameters documented in Chapter Three (25 km radius, min 3 points,
# UTM 37N); each cluster gets a distinct colour, noise points are hollow.
# ---------------------------------------------------------------------------
def make_dbscan_map(boundaries):
    if not CONFLICT_CLEANED_PATH.exists():
        log(f"SKIPPED DBSCAN map ({CONFLICT_CLEANED_PATH} not found)")
        return
    try:
        from sklearn.cluster import DBSCAN

        df = read_csv_robust(CONFLICT_CLEANED_PATH, low_memory=False)
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon"] if c in df.columns), None)
        if not (lat_col and lon_col):
            log(f"SKIPPED DBSCAN map (no latitude/longitude columns found -- has: {list(df.columns)})")
            return

        df = df.dropna(subset=[lat_col, lon_col]).copy()
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        gdf_utm = gdf.to_crs("EPSG:32637")

        coords = np.column_stack([gdf_utm.geometry.x, gdf_utm.geometry.y])
        labels = DBSCAN(eps=25000, min_samples=3).fit_predict(coords)
        gdf["_cluster"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        log(f"DBSCAN: {n_clusters} clusters found; {n_noise} / {len(labels)} points classified as noise")

        fig, ax = plt.subplots(figsize=(7.8, 8.6))
        if boundaries is not None:
            boundaries.plot(ax=ax, facecolor="#fafafa", edgecolor="#bbbbbb", linewidth=0.4)

        cmap = plt.get_cmap("tab20")
        legend_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                                  markeredgecolor="#999999", markersize=6, label="Noise (unclustered)")]
        noise = gdf[gdf["_cluster"] == -1]
        ax.scatter(noise.geometry.x, noise.geometry.y, s=10, facecolor="none",
                   edgecolor="#999999", linewidth=0.6)
        for i, cl in enumerate(sorted(c for c in gdf["_cluster"].unique() if c != -1)):
            sub = gdf[gdf["_cluster"] == cl]
            color = cmap(i % 20)
            ax.scatter(sub.geometry.x, sub.geometry.y, s=16, facecolor=color, edgecolor="black", linewidth=0.3)
            legend_handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
                                          markeredgecolor="black", markersize=6, label=f"Cluster {cl}"))

        finalise_map(fig, ax, gdf, f"DBSCAN Point Clusters (25 km radius, min. 3 points) -- {n_clusters} clusters found",
                     legend_handles, "Cluster")
        out_path = FIG_DIR / "map_dbscan_clusters.png"
        plt.savefig(out_path, dpi=230, facecolor="white", bbox_inches="tight")
        plt.close()
        log(f"SAVED {out_path}")
    except Exception as e:
        log(f"SKIPPED DBSCAN map (error: {e})")


# ---------------------------------------------------------------------------
# Map 4: Raw conflict-point map
# ---------------------------------------------------------------------------
def make_point_map(boundaries):
    if not CONFLICT_CLEANED_PATH.exists():
        log(f"SKIPPED point map ({CONFLICT_CLEANED_PATH} not found)")
        return
    try:
        df = read_csv_robust(CONFLICT_CLEANED_PATH, low_memory=False)
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon"] if c in df.columns), None)
        if not (lat_col and lon_col):
            log(f"SKIPPED point map (no latitude/longitude columns found)")
            return
        df = df.dropna(subset=[lat_col, lon_col])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

        fig, ax = plt.subplots(figsize=(7.8, 8.6))
        if boundaries is not None:
            boundaries.plot(ax=ax, facecolor="#fafafa", edgecolor="#bbbbbb", linewidth=0.4)
        ax.scatter(df[lon_col], df[lat_col], s=10, facecolor="#c0392b", edgecolor="none", alpha=0.6)
        legend_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#c0392b",
                                  markeredgecolor="none", markersize=6, label="Conflict record")]
        finalise_map(fig, ax, gdf, f"Geocoded Conflict Records (n = {len(df):,})", legend_handles, None)
        out_path = FIG_DIR / "map_conflict_points.png"
        plt.savefig(out_path, dpi=230, facecolor="white", bbox_inches="tight")
        plt.close()
        log(f"SAVED {out_path}")
    except Exception as e:
        log(f"SKIPPED point map (error: {e})")


def main():
    print("Generating spatial result maps...\n")
    boundaries = load_boundaries()

    print("\n[1/4] Hotspot classification map")
    make_hotspot_map(boundaries)

    print("\n[2/4] Predictive risk map")
    make_risk_map(boundaries)

    print("\n[3/4] DBSCAN point-cluster map")
    make_dbscan_map(boundaries)

    print("\n[4/4] Raw conflict-point map")
    make_point_map(boundaries)

    print(f"\nDone. Any maps saved are in {FIG_DIR}/")
    print("Maps that were skipped list the missing file above -- point the relevant "
          "path constant at the top of this script to the correct location and re-run.")


if __name__ == "__main__":
    main()
