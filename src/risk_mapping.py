"""
risk_mapping.py

Phase 8: turns the trained model into an actual sub-county risk map --
Low/Medium/High categories plus the underlying probability, exported
in a form the Phase 9 dashboard can consume directly.

Design choice: risk is computed from each sub-county's MOST RECENT
available lagged feature row -- i.e. "given what we know as of the
latest data, what's the predicted risk of onset next month". This is
the genuinely actionable question for a decision-support tool
(the spec's stated purpose), as opposed to re-scoring historical
months that already happened.

Risk categorization uses RELATIVE tertiles across sub-counties (bottom
third = Low, middle third = Medium, top third = High) rather than
fixed absolute probability thresholds (e.g. <0.1/0.1-0.3/>0.3). This
matters because conflict onset is rare overall (confirmed in Phase 6:
well under 1% of sub-county-months) -- fixed thresholds calibrated for
a balanced problem would likely call almost every sub-county "Low"
and hide meaningful relative differences. Tertiles guarantee the map
actually distinguishes relatively higher- and lower-risk areas, which
is what a risk map is FOR. The raw probability is always kept
alongside the category, so absolute magnitude isn't lost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd


def get_latest_features(panel_with_lags: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    For each sub-county, takes the single most recent row that has
    complete (non-NaN) values for every feature_cols -- this is "the
    latest point in time we can actually make a prediction from",
    which may not be the panel's literal last month if recent months
    are missing NDVI/rainfall data (a real, common satellite-data lag).
    """
    complete = panel_with_lags.dropna(subset=feature_cols)
    latest = (
        complete.sort_values("panel_date")
        .groupby(["County", "SubCounty"])
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def predict_risk(model, latest: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Adds risk_probability (P(onset) for the month after each
    sub-county's latest available data point)."""
    if len(latest) == 0:
        raise ValueError(
            "No sub-county has a complete feature row to predict from -- "
            "get_latest_features() returned 0 rows. This usually means one "
            "or more feature_cols are entirely NaN across the whole panel "
            "(e.g. an NDVI/CHIRPS coverage gap) -- check each column in the "
            "panel individually with df[col].notna().sum() before re-running."
        )
    out = latest.copy()
    out["risk_probability"] = model.predict_proba(out[feature_cols])[:, 1]
    return out


def categorize_risk(df: pd.DataFrame, prob_col: str = "risk_probability") -> pd.DataFrame:
    """
    Assigns Low/Medium/High via tertiles of risk_probability ACROSS
    sub-counties -- see module docstring for why relative, not fixed,
    thresholds. Ties (e.g. many sub-counties at exactly 0 probability,
    plausible given a rare-event target) are handled by pandas.qcut's
    duplicates='drop', which can collapse to fewer than 3 bins if the
    distribution is too degenerate for 3 -- reported explicitly rather
    than silently producing a 2-category "map".
    """
    out = df.copy()
    try:
        out["risk_category"] = pd.qcut(
            out[prob_col], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
        )
    except ValueError as e:
        print(f"  WARNING: could not split into 3 distinct risk tertiles ({e}) -- "
              f"this usually means most sub-counties have identical/near-identical "
              f"predicted probabilities (plausible if the model is under-trained "
              f"or the target is extremely rare). Falling back to a coarser split.")
        out["risk_category"] = pd.qcut(
            out[prob_col], q=2, labels=["Lower", "Higher"], duplicates="drop"
        )
    return out


def build_risk_layer(
    model, panel_with_lags: pd.DataFrame, feature_cols: list[str],
    boundaries: gpd.GeoDataFrame, county_col: str = "County", subcounty_col: str = "SubCounty",
) -> gpd.GeoDataFrame:
    """Full pipeline: latest features -> prediction -> categorization ->
    merged with boundary geometry, ready to export as GeoJSON for the
    Phase 9 dashboard."""
    latest = get_latest_features(panel_with_lags, feature_cols)
    predicted = predict_risk(model, latest, feature_cols)
    categorized = categorize_risk(predicted)

    layer = boundaries.merge(
        categorized, left_on=[county_col, subcounty_col], right_on=["County", "SubCounty"],
        how="left",
    )
    n_missing = layer["risk_probability"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} sub-county polygon(s) in the boundary file "
              f"have no matching prediction (no complete feature history yet, or a "
              f"name mismatch with the panel) -- these will show as blank on the map.")
    return layer
