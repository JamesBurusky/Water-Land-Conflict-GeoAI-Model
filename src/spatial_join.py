"""
spatial_join.py

Attaches County/Sub-county identity to the WRUA governance dataset via a
real point-in-polygon spatial join, using the WRUA centroid coordinates
(CentroidX, CentroidY) against your admin boundary shapefile.

Why spatial join instead of name matching
-------------------------------------------
The WRUA dataset has no County/Sub-county column at all -- it's
organized by hydrological basin (Basin, Subbasin, SubBasinManagementUnitCode),
which doesn't correspond 1:1 with administrative boundaries. A WRUA can
straddle a sub-county border. Point-in-polygon on the centroid is the
correct approach here, but note the limitation explicitly: a WRUA whose
CENTROID falls in Sub-county A but whose polygon extends into Sub-county
B will be attributed entirely to A. For WRUAs far larger than the
sub-counties they sit in (a few in the sample exceed 1,000 km2 --
"Kalikuvu" at 1,317 km2, "Lagha Kokani" at 6,367 km2), this is a real
approximation worth naming in your limitations section. A more precise
(but heavier) alternative is an area-weighted overlay -- splitting each
WRUA polygon by sub-county boundary and apportioning by area of overlap
-- which is worth doing if WRUA-level variables turn out to matter a lot
in Phase 7's SHAP results.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from name_cleaning import TARGET_COUNTIES


def load_boundaries(shapefile_path: str, county_col: str,
                     subcounty_col: str) -> gpd.GeoDataFrame:
    """
    Load a sub-county boundary shapefile/GeoJSON and restrict to the
    four target counties. Reprojects to WGS84 (EPSG:4326) if needed,
    since the WRUA CentroidX/CentroidY appear to be lon/lat already
    (values in the ~35-39 / ~-3-4.5 range match Kenya in WGS84).
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError(
            "Shapefile has no CRS defined. Confirm the projection with "
            "the source (e.g. KNBS/IEBC) before proceeding -- joining "
            "against an unknown CRS will silently misplace every point."
        )
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf[county_col].isin(TARGET_COUNTIES)].reset_index(drop=True)
    return gdf


def join_points_to_subcounty(
    df: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    county_col: str,
    subcounty_col: str,
    lon_col: str,
    lat_col: str,
    out_prefix: str = "joined",
) -> pd.DataFrame:
    """
    General point-in-polygon join: works for WRUA centroids
    (CentroidX/CentroidY) or conflict records (Longitude/Latitude) --
    anything with a lon/lat pair. Rows with missing coordinates, or
    whose point falls outside all four target counties, get NaN for
    the joined columns and are KEPT (not dropped), so nothing
    disappears from the audit trail silently.
    """
    lon_numeric = pd.to_numeric(df[lon_col], errors="coerce")
    lat_numeric = pd.to_numeric(df[lat_col], errors="coerce")

    bad_coords = df[
        (lon_numeric.isna() & df[lon_col].notna()) |
        (lat_numeric.isna() & df[lat_col].notna())
    ]
    if len(bad_coords) > 0:
        id_col = next((c for c in ["Record_ID", "WRUA_NAME", "WRA_ID"] if c in df.columns), None)
        print(f"  WARNING: {len(bad_coords)} row(s) have a NON-NUMERIC value in "
              f"'{lon_col}' or '{lat_col}' -- this usually means a CSV parsing "
              f"issue upstream (e.g. an unescaped comma in a text field shifting "
              f"columns), not a genuinely missing coordinate. These rows are "
              f"treated as missing coordinates below, but the source file is "
              f"worth checking directly.")
        for _, row in bad_coords.iterrows():
            label = row[id_col] if id_col else "(no ID column found)"
            print(f"    {label}: {lon_col}={row[lon_col]!r}, {lat_col}={row[lat_col]!r}")

    has_coords = lon_numeric.notna() & lat_numeric.notna()
    n_missing = (~has_coords).sum()
    if n_missing > 0 and len(bad_coords) < n_missing:
        print(f"  {n_missing - len(bad_coords)} additional row(s) missing "
              f"coordinates entirely (genuinely blank, not malformed) -- "
              f"excluded from spatial join, kept in output with NaN joined columns")

    boundaries_renamed = boundaries[[county_col, subcounty_col, "geometry"]].rename(
        columns={county_col: f"{out_prefix}_county", subcounty_col: f"{out_prefix}_subcounty"}
    )
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
                  for xy in zip(lon_numeric, lat_numeric)],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf, boundaries_renamed, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return pd.DataFrame(joined.drop(columns="geometry"))


def join_wrua_to_subcounty(
    wrua_df: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    county_col: str,
    subcounty_col: str,
    x_col: str = "CentroidX",
    y_col: str = "CentroidY",
) -> pd.DataFrame:
    """Thin wrapper over join_points_to_subcounty preserving the
    original WRUA-specific column names used by earlier phase scripts."""
    result = join_points_to_subcounty(
        wrua_df, boundaries, county_col, subcounty_col,
        lon_col=x_col, lat_col=y_col, out_prefix="joined",
    )
    return result
