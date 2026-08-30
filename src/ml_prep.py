"""
ml_prep.py

Phase 6 step 1: preparing the panel for modelling. Two problems
solved here, both about preventing the model from "seeing the answer"
in a way that would inflate offline metrics without being usable for
real forecasting.

PROBLEM 1: same-month leakage in decay features
--------------------------------------------------
conflict_persistence(t) is 1.0 in the EXACT month a conflict record's
Date_Start falls in -- confirmed by direct test. Using it to predict
conflict_onset(t) (the target) means the model is shown the answer
disguised as a feature. The same applies to conflict_severity_weighted,
decayed_sentiment, topic_diversity -- anything derived from the
same-month event.

FIX: every dynamic feature is LAGGED by lag_months (default 1) before
being used as a predictor -- predict month t's onset using only
information genuinely available as of month t-1 (or earlier). This
is what makes the resulting model an actual forecasting model rather
than a same-day nowcast dressed up as one.

PROBLEM 2: random train/test splitting on temporal data
--------------------------------------------------------
A random 80/20 split would let the model train on some of 2020's data
and test on other parts of 2020 -- meaning it could "learn" from
patterns that are only knowable in hindsight for that period. The
project spec explicitly calls for training on earlier years and
testing on later years; temporal_train_test_split implements exactly
that split, not a random one.
"""

from __future__ import annotations

import pandas as pd

# Columns that reflect the CURRENT month's own conflict activity and
# therefore MUST be lagged before use as predictors. Static structural
# columns (population, abstraction, wrua_count) don't need lagging --
# they don't change month to month, so there's no "future" information
# in them to leak.
DYNAMIC_FEATURE_COLS = [
    "conflict_persistence", "conflict_severity_weighted",
    "decayed_sentiment", "topic_diversity",
    "Mean_NDVI", "Rainfall_mm", "Rainfall_Anomaly_mm", "Rainfall_Anomaly_Percent",
]

STATIC_FEATURE_COLS = [
    "Population_Density_per_SqKm", "Total_Population",
    "total_abstraction_m3_per_day", "wrua_count",
]

TARGET_COL = "conflict_onset"


def add_lagged_features(panel: pd.DataFrame, lag_months: int = 1,
                         dynamic_cols: list[str] | None = None) -> pd.DataFrame:
    """
    For each sub-county, shifts every dynamic feature column forward
    by lag_months -- row t gets the value that was true at t-lag_months.
    The FIRST lag_months rows of each sub-county's time series become
    NaN (no prior data exists yet) and should be dropped before
    modelling (see prepare_modelling_table below), not imputed with 0
    -- imputing would incorrectly claim "no conflict risk" for a
    period that's actually just outside the observed data window.
    """
    dynamic_cols = dynamic_cols or DYNAMIC_FEATURE_COLS
    out = panel.sort_values(["SubCounty", "panel_date"]).copy()

    present_cols = [c for c in dynamic_cols if c in out.columns]
    missing_cols = [c for c in dynamic_cols if c not in out.columns]
    if missing_cols:
        print(f"  NOTE: {missing_cols} not found in panel -- skipping (fine if "
              f"you haven't run 08_nlp_panel_features.py yet)")

    for col in present_cols:
        out[f"{col}_lag{lag_months}"] = out.groupby("SubCounty")[col].shift(lag_months)

    return out


def prepare_modelling_table(
    panel: pd.DataFrame, lag_months: int = 1,
    dynamic_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full prep: lag dynamic features, drop rows with no lagged history
    yet (the first lag_months months of each sub-county's series), and
    return (table, feature_column_names) ready for train/test splitting.
    """
    dynamic_cols = dynamic_cols or DYNAMIC_FEATURE_COLS
    static_cols = static_cols or STATIC_FEATURE_COLS

    lagged = add_lagged_features(panel, lag_months=lag_months, dynamic_cols=dynamic_cols)
    lagged_feature_names = [f"{c}_lag{lag_months}" for c in dynamic_cols if c in panel.columns]
    static_feature_names = [c for c in static_cols if c in panel.columns]
    feature_cols = lagged_feature_names + static_feature_names

    n_before = len(lagged)
    lagged = lagged.dropna(subset=lagged_feature_names + [target_col])
    n_after = len(lagged)
    print(f"  Dropped {n_before - n_after:,} row(s) with no lagged history yet "
          f"(first {lag_months} month(s) of each sub-county's series) or missing "
          f"target -- {n_after:,} rows remain for modelling")

    return lagged, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame, cutoff_year: int, year_col: str = "Year"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits strictly by year: train = everything BEFORE cutoff_year,
    test = cutoff_year onward. This is a hard temporal boundary, not a
    random split -- matches the project spec's "train on earlier
    years, test on later years" design and is what makes the
    evaluation a genuine test of forecasting ability rather than
    interpolation within a single time period.
    """
    train = df[df[year_col] < cutoff_year].copy()
    test = df[df[year_col] >= cutoff_year].copy()
    print(f"  Train: {len(train):,} rows (years < {cutoff_year}) | "
          f"Test: {len(test):,} rows (years >= {cutoff_year})")
    if len(train) == 0 or len(test) == 0:
        print(f"  WARNING: one side of the split is empty -- cutoff_year={cutoff_year} "
              f"may be outside the data's actual year range")
    return train, test
