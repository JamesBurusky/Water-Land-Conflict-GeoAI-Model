"""
land_water_analysis.py

Addresses the explicit research question raised early in this project:
the study covers BOTH land and water conflicts, and part of its purpose
is understanding whether/how they relate. No column in the source data
says "this is a land conflict" or "this is a water conflict" -- so this
module builds that classification from the text (NLP_Keywords,
falling back to Incident_Summary), then analyzes how the resulting
categories relate to severity, chronicity, county, and time.

IMPORTANT: this is a keyword heuristic, not a ground-truth label. Some
terms are genuinely ambiguous (e.g. "pasture" relates to land use but
is frequently a water-scarcity-driven conflict; "catchment" is a land
feature defined by its relationship to water). The keyword lists below
are deliberately exposed as named constants so they're easy to review
and adjust -- treat the resulting categories as a documented,
reproducible heuristic in your methodology, not an authoritative
ground truth.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

LAND_KEYWORDS = [
    "title deed", "title deeds", "eviction", "displacement", "private land",
    "compulsory acquisition", "land use", "land right", "land rights",
    "boundary", "parcel", "landbuying", "land-buying", "land buying",
    "forest excision", "buffer zone", "grazing", "pasture", "communal land",
    "grabbing", "land grabbing", "registry", "easement", "compensation",
    "resettlement", "informal settlement",
]

WATER_KEYWORDS = [
    "river", "water", "wetland", "stream", "irrigation", "riparian",
    "catchment", "dam", "water pan", "pipeline", "lake", "drought",
    "fishing", "borehole", "aquifer", "spring", "flood", "rainfall",
    "abstraction", "reservoir", "swamp", "water pollution",
]


def _compile_pattern(keywords: list[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


LAND_PATTERN = _compile_pattern(LAND_KEYWORDS)
WATER_PATTERN = _compile_pattern(WATER_KEYWORDS)


def classify_domain(text: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (domain, matched_land_terms, matched_water_terms) for one
    record's text. domain is one of: "Land-only", "Water-only",
    "Mixed", "Neither" -- matched terms are returned too so a reviewer
    can see exactly why a record was classified a given way, rather
    than trusting the label blindly.
    """
    if not isinstance(text, str) or not text.strip():
        return "Neither", [], []
    land_matches = sorted(set(m.lower() for m in LAND_PATTERN.findall(text)))
    water_matches = sorted(set(m.lower() for m in WATER_PATTERN.findall(text)))
    has_land, has_water = len(land_matches) > 0, len(water_matches) > 0
    if has_land and has_water:
        domain = "Mixed"
    elif has_land:
        domain = "Land-only"
    elif has_water:
        domain = "Water-only"
    else:
        domain = "Neither"
    return domain, land_matches, water_matches


def add_domain_classification(df: pd.DataFrame,
                               keywords_col: str = "NLP_Keywords",
                               fallback_text_col: str = "Incident_Summary") -> pd.DataFrame:
    """
    Classifies every record. Prefers NLP_Keywords (already-extracted,
    cleaner terms from Phase 2) and falls back to Incident_Summary for
    any record missing keywords, so every record with SOME text gets a
    domain rather than defaulting to "Neither" unnecessarily.
    """
    out = df.copy()
    text_source = out[keywords_col].fillna("")
    if fallback_text_col in out.columns:
        needs_fallback = text_source.str.strip() == ""
        text_source = text_source.where(~needs_fallback, out[fallback_text_col].fillna(""))

    results = text_source.map(classify_domain)
    out["conflict_domain"] = [r[0] for r in results]
    out["matched_land_terms"] = [", ".join(r[1]) for r in results]
    out["matched_water_terms"] = [", ".join(r[2]) for r in results]
    return out


def domain_severity_summary(df: pd.DataFrame, domain_col: str = "conflict_domain",
                             severity_col: str = "severity_score") -> pd.DataFrame:
    """Mean/median/count of severity_score per domain -- answers "are
    mixed land-water conflicts more severe than single-domain ones?" """
    if severity_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(domain_col)[severity_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "n_records"})
    )


def domain_composite_crosstab(df: pd.DataFrame, domain_col: str = "conflict_domain",
                               composite_col: str = "Is_Composite") -> tuple[pd.DataFrame, dict]:
    """
    Cross-tabulates conflict domain against chronic/composite status,
    plus a chi-square test of independence -- answers "are mixed land-
    water conflicts more likely to become chronic than single-domain
    ones?" with an actual significance test, not just eyeballed counts.
    Returns (crosstab, test_result_dict); test_result is empty if the
    table is too sparse for a meaningful test (any expected cell < 5,
    the standard rule of thumb for chi-square validity).
    """
    if composite_col not in df.columns:
        return pd.DataFrame(), {}
    crosstab = pd.crosstab(df[domain_col], df[composite_col])
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return crosstab, {}
    chi2, p, dof, expected = chi2_contingency(crosstab)
    if (expected < 5).any():
        return crosstab, {
            "chi2": chi2, "p_value": p, "dof": dof,
            "warning": "at least one expected cell count < 5 -- chi-square "
                       "result may not be reliable with this little data",
        }
    return crosstab, {"chi2": chi2, "p_value": p, "dof": dof}


def domain_by_county(df: pd.DataFrame, domain_col: str = "conflict_domain",
                      county_col: str = "County") -> pd.DataFrame:
    """Cross-tab of domain by county -- do certain counties skew toward
    land-only, water-only, or mixed conflicts?"""
    return pd.crosstab(df[county_col], df[domain_col])


def domain_yearly_trend(df: pd.DataFrame, domain_col: str = "conflict_domain",
                         date_col: str = "Date_Start_parsed") -> pd.DataFrame:
    """Yearly count per domain -- has the land/water/mixed balance
    shifted over time?"""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    years = dates.dt.year
    return pd.crosstab(years, df[domain_col]).sort_index()
