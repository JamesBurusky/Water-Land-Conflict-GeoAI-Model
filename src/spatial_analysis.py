"""
spatial_analysis.py

Phase 4 steps 1-3: hotspot analysis (Getis-Ord Gi*), Kernel Density
Estimation, and DBSCAN clustering.

Why KNN weights instead of contiguity (Queen/Rook) for Getis-Ord
--------------------------------------------------------------------
Standard practice for Gi* uses Queen/Rook contiguity weights (sub-
counties sharing a border are neighbors). That breaks down here:
Turkana is geographically isolated from Nairobi/Kiambu/Machakos --
there is no shared border, so contiguity weights would leave Turkana's
sub-counties with ZERO neighbors (an "island"), and esda's Gi*
cannot compute a meaningful z-score for an island. K-nearest-neighbor
weights (each sub-county's k geographically nearest others, regardless
of whether they touch) avoid this -- every unit gets neighbors. The
tradeoff, worth stating in your methodology: KNN neighbors for a
Turkana sub-county will be OTHER Turkana sub-counties (nothing else is
close), while a Nairobi sub-county's KNN neighbors will be genuine
adjacent neighbors. This is a reasonable, commonly-used compromise for
non-contiguous study areas, not a hidden flaw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd


def aggregate_subcounty_summary(panel: pd.DataFrame,
                                 value_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Collapses the monthly panel to one row per sub-county -- the
    cross-sectional summary Getis-Ord Gi* needs (Gi* operates on a
    single attribute value per spatial unit, not a time series).

    Default aggregation: sum for conflict_onset (total onset months),
    mean for everything else (average persistence/severity/sentiment
    level over the study period).
    """
    value_cols = value_cols or [
        "conflict_onset", "conflict_persistence", "conflict_severity_weighted",
        "decayed_sentiment", "topic_diversity",
    ]
    agg = {}
    for col in value_cols:
        if col not in panel.columns:
            continue
        agg[col] = "sum" if col == "conflict_onset" else "mean"

    summary = panel.groupby(["County", "SubCounty"]).agg(agg).reset_index()
    if "conflict_onset" in summary.columns:
        summary = summary.rename(columns={"conflict_onset": "total_onset_months"})
    return summary


def compute_getis_ord(
    subcounty_summary: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    value_col: str,
    subcounty_col: str = "SubCounty",
    k_neighbors: int = 4,
) -> gpd.GeoDataFrame:
    """
    Computes Getis-Ord Gi* z-scores and p-values for value_col across
    sub-counties, using KNN spatial weights (see module docstring for
    why, not Queen contiguity).

    Returns a GeoDataFrame with the original geometry plus:
      - gi_zscore: positive = hotspot (high values surrounded by high
        values), negative = coldspot
      - gi_pvalue: statistical significance of that z-score
      - hotspot_category: human-readable label at conventional
        confidence thresholds (90/95/99%), for direct use in a choropleth
    """
    from libpysal.weights import KNN
    from esda.getisord import G_Local

    merged = boundaries.merge(subcounty_summary, left_on=subcounty_col,
                               right_on=subcounty_col, how="inner")
    merged = merged.dropna(subset=[value_col]).reset_index(drop=True)

    if len(merged) < k_neighbors + 1:
        raise ValueError(f"Only {len(merged)} sub-counties have data for '{value_col}' "
                          f"-- need at least {k_neighbors + 1} for k={k_neighbors} "
                          f"nearest-neighbor weights. Lower k_neighbors or check for "
                          f"missing values in this column.")

    w = KNN.from_dataframe(merged, k=k_neighbors)
    w.transform = "r"

    gi = G_Local(merged[value_col].astype(float).values, w, permutations=999)

    merged["gi_zscore"] = gi.Zs
    merged["gi_pvalue"] = gi.p_sim

    def categorize(row):
        if row["gi_pvalue"] > 0.10:
            return "Not significant"
        conf = "99%" if row["gi_pvalue"] < 0.01 else ("95%" if row["gi_pvalue"] < 0.05 else "90%")
        kind = "Hotspot" if row["gi_zscore"] > 0 else "Coldspot"
        return f"{kind} ({conf} confidence)"

    merged["hotspot_category"] = merged.apply(categorize, axis=1)
    return merged


def compute_kde_surface(
    lons: np.ndarray, lats: np.ndarray,
    grid_size: int = 100, bandwidth: str | float = "scott",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kernel Density Estimation over raw conflict point coordinates.
    Returns (X, Y, Z) grid arrays ready for a contour/heatmap plot.
    bandwidth: 'scott' (default, scipy's rule-of-thumb) or a float to
    override -- worth trying a couple of values, since KDE bandwidth
    is a classic case where the "right" answer depends on how smooth
    vs. locally detailed you want the surface to look; there's no
    single objectively correct choice.

    Coordinates are coerced to numeric explicitly -- if the source
    column has even one non-numeric value (e.g. a CSV parsing issue
    upstream shifting a text value into the coordinate column, as
    happened with the WRUA data in Phase 3), pandas/numpy silently
    types the WHOLE array as object, which breaks np.isnan() with a
    cryptic error rather than identifying the bad row. Coercing here
    makes that failure mode explicit and reports exactly how many
    rows were affected.
    """
    lons = pd.to_numeric(pd.Series(lons), errors="coerce").to_numpy()
    lats = pd.to_numeric(pd.Series(lats), errors="coerce").to_numpy()

    not_nan = ~(np.isnan(lons) | np.isnan(lats))
    in_range = (np.abs(lats) <= 90) & (np.abs(lons) <= 180)
    valid = not_nan & in_range

    n_nan = (~not_nan).sum()
    n_out_of_range = (not_nan & ~in_range).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} coordinate pair(s) were non-numeric or missing "
              f"and excluded from the KDE surface.")
    if n_out_of_range > 0:
        print(f"  WARNING: {n_out_of_range} coordinate pair(s) were numeric but "
              f"OUT OF VALID RANGE (|lat|>90 or |lon|>180) and excluded -- likely "
              f"a unit error or swapped lat/lon column upstream, worth checking "
              f"those specific rows directly. An out-of-range value left in would "
              f"otherwise silently stretch the whole KDE grid to cover a huge, "
              f"mostly-empty area.")
    lons, lats = lons[valid], lats[valid]

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([lons, lats]), bw_method=bandwidth)

    lon_grid = np.linspace(lons.min() - 0.1, lons.max() + 0.1, grid_size)
    lat_grid = np.linspace(lats.min() - 0.1, lats.max() + 0.1, grid_size)
    X, Y = np.meshgrid(lon_grid, lat_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    return X, Y, Z


def compute_dbscan_clusters(
    lons: np.ndarray, lats: np.ndarray,
    eps_km: float = 5.0, min_samples: int = 3,
) -> np.ndarray:
    """
    DBSCAN clustering on conflict point locations. Projects lon/lat to
    UTM Zone 37N (EPSG:32737, covers Kenya) BEFORE clustering, so eps
    is a real distance in kilometers -- running DBSCAN directly on
    degrees would make eps meaningless (a degree of longitude is a very
    different real distance near the equator vs. further from it, and
    lon/lat degrees aren't even equal to each other in real distance).

    Returns cluster labels (-1 = noise, not part of any cluster --
    this is DBSCAN's standard convention, not a data quality flag).
    """
    from sklearn.cluster import DBSCAN
    import pyproj

    lons = pd.to_numeric(pd.Series(lons), errors="coerce").to_numpy()
    lats = pd.to_numeric(pd.Series(lats), errors="coerce").to_numpy()

    # Two separate validity checks: NaN (non-numeric/missing) and
    # out-of-range (a number, but not a physically valid coordinate --
    # e.g. a swapped lat/lon, a stray large value, or a unit error
    # upstream). Both are real, distinct data-quality signals worth
    # reporting separately rather than lumping into one count.
    not_nan = ~(np.isnan(lons) | np.isnan(lats))
    in_range = (np.abs(lats) <= 90) & (np.abs(lons) <= 180)
    valid = not_nan & in_range

    n_nan = (~not_nan).sum()
    n_out_of_range = (not_nan & ~in_range).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} coordinate pair(s) were non-numeric or missing "
              f"and excluded from DBSCAN clustering.")
    if n_out_of_range > 0:
        print(f"  WARNING: {n_out_of_range} coordinate pair(s) were numeric but "
              f"OUT OF VALID RANGE (|lat|>90 or |lon|>180) and excluded -- this "
              f"usually means a unit error or swapped lat/lon column upstream, "
              f"worth checking those specific rows directly.")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
    x_m, y_m = transformer.transform(lons[valid], lats[valid])

    finite = np.isfinite(x_m) & np.isfinite(y_m)
    n_nonfinite = (~finite).sum()
    if n_nonfinite > 0:
        print(f"  WARNING: {n_nonfinite} coordinate pair(s) produced a non-finite "
              f"value after UTM projection and were excluded -- likely a coordinate "
              f"technically in range but nowhere near Kenya (e.g. (0, 0), or a "
              f"digit-entry typo).")

    # Rebuild `valid` to mark exactly the rows that survived BOTH the
    # range check and the finite-after-projection check, so labels[valid]
    # below lines up correctly with coords_m.
    valid_indices = np.flatnonzero(valid)[finite]
    valid = np.zeros(len(lons), dtype=bool)
    valid[valid_indices] = True
    coords_m = np.column_stack([x_m[finite], y_m[finite]])

    db = DBSCAN(eps=eps_km * 1000, min_samples=min_samples).fit(coords_m)

    labels = np.full(len(lons), -2)  # -2 marks originally-invalid (NaN) coords, distinct from DBSCAN's own -1 noise label
    labels[valid] = db.labels_
    return labels
