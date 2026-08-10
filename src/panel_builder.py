"""
panel_builder.py

Phase 3: assembles the unit of analysis for the ML model -- one row
per (Sub-county, Year, Month) -- by joining every source dataset onto
a full grid spanning the study period.

Two different kinds of features get joined differently, and mixing
them up would silently produce a misleading panel:

  - STATIC / structural features (population density, total water
    abstraction permitted, count of WRUAs): these datasets have no
    time dimension of their own (a 2019 census, a snapshot of current
    WRA permits, current WRUA registrations). They're repeated
    identically across every month for a given sub-county. This is a
    real, worth-stating limitation: the panel cannot capture e.g.
    population growth or new abstraction permits issued over time --
    only cross-sectional structural pressure, held constant.

  - TIME-VARYING features (NDVI, rainfall, conflict onset/persistence/
    severity): these genuinely change month to month and are joined
    on (Sub-county, Year, Month), not just Sub-county.

Sub-county name matching happens at build-time, not before: NDVI/
CHIRPS's ADM2_EN naming may not exactly match the canonical KNBS list,
so match_dataframe() from name_cleaning.py is applied here too, with
unmatched rows reported (not silently dropped).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from name_cleaning import CanonicalLookup, match_dataframe, normalize_name
from conflict_cleaning import decayed_persistence, onset_flag


def build_grid(lookup: CanonicalLookup, start_date: str, end_date: str) -> pd.DataFrame:
    """Every (County, SubCounty) x every month in [start_date, end_date]."""
    months = pd.date_range(start_date, end_date, freq="MS")
    subcounties = lookup.table[["County", "SubCounty"]].drop_duplicates()
    grid = subcounties.merge(pd.DataFrame({"panel_date": months}), how="cross")
    grid["Year"] = grid["panel_date"].dt.year
    grid["Month"] = grid["panel_date"].dt.month
    return grid


def add_static_population(grid: pd.DataFrame, census: pd.DataFrame,
                           lookup: CanonicalLookup) -> pd.DataFrame:
    """Joins 2019 census population density onto every month (static --
    see module docstring)."""
    pop = census[["SubCounty", "Population_Density_per_SqKm",
                  "Total_Population"]].drop_duplicates(subset=["SubCounty"])
    out = grid.merge(pop, on="SubCounty", how="left")
    n_missing = out["Population_Density_per_SqKm"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} panel row(s) missing population data "
              f"(sub-county not found in census) -- check for a canonical "
              f"list / census mismatch")
    return out


def add_static_abstraction(grid: pd.DataFrame, wra_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Sums total permitted abstraction (M3/Day) per sub-county from the
    Phase 1-cleaned WRA data (using matched_subcounty, not the raw
    messy Sub_county field). Static across all months -- see module
    docstring for why (WRA data is a permit snapshot, not a time series).
    """
    total_col = [c for c in wra_cleaned.columns if "Total" in c and "M3" in c]
    if not total_col:
        raise ValueError("Could not find a 'Total (M3/Day)' column in wra_cleaned -- "
                          "check the column name matches what Phase 1 produced.")
    total_col = total_col[0]

    subcounty_col = "matched_subcounty" if "matched_subcounty" in wra_cleaned.columns else "Sub_county"
    agg = wra_cleaned.groupby(subcounty_col, dropna=True)[total_col].sum().reset_index()
    agg.columns = ["SubCounty", "total_abstraction_m3_per_day"]

    out = grid.merge(agg, on="SubCounty", how="left")
    out["total_abstraction_m3_per_day"] = out["total_abstraction_m3_per_day"].fillna(0)
    return out


def add_static_wrua_count(grid: pd.DataFrame, wrua_joined: pd.DataFrame) -> pd.DataFrame:
    """Count of WRUAs per sub-county (governance capacity proxy),
    static across all months -- institutional presence doesn't
    meaningfully change month to month at this data's resolution."""
    subcounty_col = "joined_subcounty" if "joined_subcounty" in wrua_joined.columns else "geo_subcounty"
    agg = wrua_joined.groupby(subcounty_col, dropna=True).size().reset_index(name="wrua_count")
    agg.columns = ["SubCounty", "wrua_count"]

    out = grid.merge(agg, on="SubCounty", how="left")
    out["wrua_count"] = out["wrua_count"].fillna(0)
    return out


def _match_remote_sensing_subcounty(df: pd.DataFrame, lookup: CanonicalLookup,
                                     subcounty_col: str, county_col: str,
                                     label: str) -> pd.DataFrame:
    """NDVI/CHIRPS's ADM2_EN naming may not match the canonical KNBS
    list exactly -- match it here, same logic as Phase 1's WRA
    matching, and report anything unmatched rather than silently
    dropping it from the panel."""
    matched = match_dataframe(df, subcounty_col=subcounty_col, lookup=lookup,
                               county_col=county_col)
    n_unmatched = (matched["match_method"] == "unmatched").sum()
    if n_unmatched > 0:
        print(f"  WARNING: {label}: {n_unmatched} row(s) could not be matched "
              f"to a canonical sub-county -- these will be missing from the "
              f"panel. Unmatched values: "
              f"{matched.loc[matched['match_method']=='unmatched', subcounty_col].unique().tolist()}")
    matched["SubCounty"] = matched["matched_subcounty"]
    return matched


def add_timevarying_ndvi(grid: pd.DataFrame, ndvi_filtered: pd.DataFrame,
                          lookup: CanonicalLookup) -> pd.DataFrame:
    matched = _match_remote_sensing_subcounty(
        ndvi_filtered, lookup, subcounty_col="ADM2_EN", county_col="ADM1_EN", label="NDVI"
    )
    agg = matched.groupby(["SubCounty", "Year", "Month"], dropna=True)["Mean_NDVI"].mean().reset_index()
    return grid.merge(agg, on=["SubCounty", "Year", "Month"], how="left")


def add_timevarying_rainfall(grid: pd.DataFrame, chirps_filtered: pd.DataFrame,
                              lookup: CanonicalLookup) -> pd.DataFrame:
    matched = _match_remote_sensing_subcounty(
        chirps_filtered, lookup, subcounty_col="ADM2_EN", county_col="ADM1_EN", label="CHIRPS"
    )
    agg_cols = [c for c in ["Rainfall_mm", "Rainfall_Anomaly_mm", "Rainfall_Anomaly_Percent"]
                if c in matched.columns]
    agg = matched.groupby(["SubCounty", "Year", "Month"], dropna=True)[agg_cols].mean().reset_index()
    return grid.merge(agg, on=["SubCounty", "Year", "Month"], how="left")


def _decay_weights(events: pd.DataFrame, eval_date: pd.Timestamp,
                    half_life_days: float, composite_multiplier: float,
                    start_col: str, composite_col: str):
    """
    Shared decay-weight computation used by both conflict_persistence
    (via decayed_persistence in conflict_cleaning.py) and the NLP
    aggregation functions below -- same leakage-safe rule: only events
    with Date_Start <= eval_date contribute.
    """
    past = events[events[start_col] <= eval_date]
    if len(past) == 0:
        return past, np.array([])
    days_elapsed = (eval_date - past[start_col]).dt.days.clip(lower=0)
    effective_half_life = np.where(
        past[composite_col], half_life_days * composite_multiplier, half_life_days
    )
    decay = np.exp(-np.log(2) * days_elapsed / effective_half_life)
    return past, decay


def add_nlp_features(
    grid: pd.DataFrame,
    conflict_nlp_geocoded: pd.DataFrame,
    half_life_days: float = 180,
    composite_multiplier: float = 3.0,
    subcounty_col: str = "geo_subcounty",
    start_col: str = "Date_Start_parsed",
    composite_col: str = "Is_Composite",
) -> pd.DataFrame:
    """
    Adds two NLP-derived panel features, both using the SAME decay
    weighting as conflict_persistence (same half-life, same leakage
    rule: only events with Date_Start <= eval_date contribute) so all
    three "how much conflict signal is active here" features are
    directly comparable:

      - decayed_sentiment: a decay-WEIGHTED AVERAGE (not sum) of
        sentiment_score across recent/ongoing events -- answers "what's
        the emotional tone of what's currently active here", not "how
        much" (that's what conflict_persistence already covers).
        NaN when there are no past events yet (distinct from 0/neutral,
        which would be a real claim about tone).

      - dominant_topic_id / topic_diversity: among past events, finds
        which BERTopic topic carries the most decayed weight (the
        "type of conflict currently dominant" in this sub-county-month),
        and the Shannon entropy of the decayed topic distribution
        (topic_diversity) -- low entropy means one theme dominates,
        high entropy means several themes are co-occurring. Both NaN
        when there's no past signal yet.

    Requires conflict_nlp_geocoded to have sentiment_score, topic_id,
    Date_Start_parsed, Is_Composite already present (i.e. built from
    conflict_topics.csv / conflict_nlp_enriched.csv, merged with the
    sub-county assignment from Phase 3's geocoding step).
    """
    conflict_nlp_geocoded = conflict_nlp_geocoded.copy()
    conflict_nlp_geocoded[start_col] = pd.to_datetime(conflict_nlp_geocoded[start_col])
    grouped = {sc: grp for sc, grp in conflict_nlp_geocoded.groupby(subcounty_col)}

    sentiment_vals, dominant_topic_vals, diversity_vals = [], [], []

    for row in grid.itertuples(index=False):
        events = grouped.get(row.SubCounty)
        if events is None or len(events) == 0:
            sentiment_vals.append(np.nan)
            dominant_topic_vals.append(np.nan)
            diversity_vals.append(np.nan)
            continue

        past, decay = _decay_weights(events, row.panel_date, half_life_days,
                                      composite_multiplier, start_col, composite_col)
        if len(past) == 0 or decay.sum() == 0:
            sentiment_vals.append(np.nan)
            dominant_topic_vals.append(np.nan)
            diversity_vals.append(np.nan)
            continue

        sent = past["sentiment_score"].fillna(0).to_numpy()
        sentiment_vals.append(float((decay * sent).sum() / decay.sum()))

        topics = past["topic_id"].fillna(-1).astype(int).to_numpy()
        topic_weight = {}
        for t, w in zip(topics, decay):
            topic_weight[t] = topic_weight.get(t, 0.0) + w
        total_w = sum(topic_weight.values())
        probs = np.array([w / total_w for w in topic_weight.values()])
        dominant_topic_vals.append(max(topic_weight, key=topic_weight.get))
        diversity_vals.append(max(0.0, float(-(probs * np.log(probs + 1e-12)).sum())))

    out = grid.copy()
    out["decayed_sentiment"] = sentiment_vals
    out["dominant_topic_id"] = dominant_topic_vals
    out["topic_diversity"] = diversity_vals
    return out


def add_conflict_features(
    grid: pd.DataFrame,
    conflict_geocoded: pd.DataFrame,
    half_life_days: float = 180,
    composite_multiplier: float = 3.0,
    subcounty_col: str = "geo_subcounty",
    start_col: str = "Date_Start_parsed",
    severity_col: str = "severity_score",
    composite_col: str = "Is_Composite",
) -> pd.DataFrame:
    """
    Computes, for every (SubCounty, Year, Month) row:
      - conflict_onset: 1 if any conflict record's Date_Start falls in
        this exact month, in this sub-county
      - conflict_persistence: decayed occurrence-weighted score (the
        Phase 1b decayed_persistence function)
      - conflict_severity_weighted: same, weighted by severity_score

    Uses the sub-county each conflict record was spatially joined to
    (from the point-in-polygon join against Latitude/Longitude), NOT
    the free-text Sub_Location field, which is ward/landmark-level and
    doesn't reliably match sub-county boundaries (confirmed on real
    sample data -- see README).
    """
    conflict_geocoded = conflict_geocoded.copy()
    conflict_geocoded[start_col] = pd.to_datetime(conflict_geocoded[start_col])

    onset_vals, persistence_vals, severity_vals = [], [], []
    grouped = {sc: grp for sc, grp in conflict_geocoded.groupby(subcounty_col)}

    for row in grid.itertuples(index=False):
        sc = row.SubCounty
        eval_date = row.panel_date
        events = grouped.get(sc)
        if events is None or len(events) == 0:
            onset_vals.append(False)
            persistence_vals.append(0.0)
            severity_vals.append(0.0)
            continue
        onset_vals.append(onset_flag(events, row.Year, row.Month, start_col=start_col))
        persistence_vals.append(decayed_persistence(
            events, eval_date, half_life_days=half_life_days,
            composite_half_life_multiplier=composite_multiplier,
            start_col=start_col, composite_col=composite_col,
        ))
        severity_vals.append(decayed_persistence(
            events, eval_date, half_life_days=half_life_days,
            composite_half_life_multiplier=composite_multiplier,
            start_col=start_col, composite_col=composite_col,
            severity_col=severity_col,
        ))

    out = grid.copy()
    out["conflict_onset"] = onset_vals
    out["conflict_persistence"] = persistence_vals
    out["conflict_severity_weighted"] = severity_vals
    return out
