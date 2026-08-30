"""
temporal_analysis.py

Phase 4 steps 4-5: temporal trend analysis and seasonal analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def yearly_trend(panel: pd.DataFrame, value_col: str = "conflict_onset",
                  agg: str = "sum") -> pd.DataFrame:
    """
    Aggregates a panel column to one value per year (default: total
    onset months per year, across all sub-counties), and fits a simple
    linear trend (scipy.stats.linregress) -- reports slope, p-value,
    and R^2 so trend significance is stated explicitly rather than
    eyeballed off a chart.
    """
    yearly = panel.groupby("Year")[value_col].agg(agg).reset_index()
    result = stats.linregress(yearly["Year"], yearly[value_col])
    yearly.attrs["trend_slope"] = result.slope
    yearly.attrs["trend_pvalue"] = result.pvalue
    yearly.attrs["trend_r_squared"] = result.rvalue ** 2
    return yearly


def seasonal_pattern(panel: pd.DataFrame, value_col: str = "conflict_onset",
                      agg: str = "sum") -> pd.DataFrame:
    """
    Aggregates a panel column by CALENDAR month (1-12), collapsing
    across all years -- reveals whether conflict activity clusters in
    particular months regardless of year (e.g. dry-season months),
    distinct from yearly_trend's year-over-year direction.
    """
    monthly = panel.groupby("Month")[value_col].agg(["sum", "mean", "std"]).reset_index()
    monthly.columns = ["Month", f"{value_col}_total", f"{value_col}_mean", f"{value_col}_std"]
    return monthly


def county_yearly_trend(panel: pd.DataFrame, value_col: str = "conflict_onset",
                         agg: str = "sum") -> pd.DataFrame:
    """Same as yearly_trend but broken out per county, for comparing
    whether the four target counties are trending the same direction
    or diverging."""
    return panel.groupby(["County", "Year"])[value_col].agg(agg).reset_index()
