"""
data_access.py

THE key architectural piece: every route handler calls functions in
this module, never reads a file directly. Right now every function
reads from outputs/*.csv or *.geojson produced by the pipeline
scripts. When you migrate to Postgres, only this file changes --
each function's SIGNATURE and RETURN TYPE (a pandas DataFrame, or a
GeoDataFrame for spatial data) stays the same, so nothing in routers/
or the frontend needs to change. A function like get_panel() becomes
`pd.read_sql(query, conn)` instead of `pd.read_csv(path)`, with the
same filter arguments turning into a WHERE clause instead of pandas
boolean indexing.

Caching: file reads are cached in-process with a simple TTL (see
config.CACHE_TTL_SECONDS) -- the pipeline re-runs occasionally, not
per-request, so re-parsing a multi-thousand-row CSV on every single
API call would be wasted work. The cache is intentionally dumb (no
invalidation signal from the pipeline) -- restart the API process (or
wait out the TTL) after a fresh pipeline run to pick up new data. This
whole cache disappears naturally once Postgres is the source of truth
(the DB itself is the shared, always-current state).
"""
from __future__ import annotations

import time
import datetime
from functools import wraps
from pathlib import Path

import pandas as pd
import geopandas as gpd

from . import config


def _json_safe_value(value):
    """Converts one cell value to something JSON can represent:
    NaN/NaT/None all become None; Timestamps become ISO strings;
    everything else passes through unchanged."""
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # pd.isna() can't evaluate some types (e.g. lists) -- fine, just pass through
    return value


def df_to_json_records(df: pd.DataFrame) -> list[dict]:
    """
    Converts a DataFrame to a list of JSON-safe dicts. Deliberately
    fixes values AFTER calling to_dict(), not before with
    df.where(df.notna(), None) -- that approach silently fails on
    numeric columns, because a float64 column can't actually hold
    Python's None (numpy has no such value for a float array), so
    pandas converts any assigned None right back to NaN, and the bad
    value survives to break FastAPI's strict JSON encoding anyway.
    Operating on the plain Python dicts from to_dict() sidesteps this
    entirely. Use this everywhere a route returns DataFrame rows.
    """
    records = df.to_dict(orient="records")
    return [{k: _json_safe_value(v) for k, v in record.items()} for record in records]


def geodf_to_json_safe_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    Same problem as df_to_json_records, applied to a GeoDataFrame's
    GeoJSON representation: __geo_interface__ can carry raw NaN floats
    in a feature's properties, which breaks strict JSON encoding.
    Fixes properties in the already-built geo_interface dict (plain
    Python objects at that point), sidestepping the same pandas
    dtype-coercion trap.
    """
    geo = gdf.__geo_interface__
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        feature["properties"] = {k: _json_safe_value(v) for k, v in props.items()}
    return geo


def _ttl_cache(ttl_seconds: int):
    """Minimal TTL cache -- avoids adding a dependency (e.g. cachetools)
    for something this simple. Keyed on function args, since several
    of these functions are called with no arguments (whole-file reads)."""
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return value
            value = func(*args, **kwargs)
            cache[key] = (value, now)
            return value
        return wrapper
    return decorator


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_risk_layer() -> gpd.GeoDataFrame | None:
    """The Phase 8 output -- risk category + probability per sub-county,
    with geometry. This is what the map's choropleth layer renders."""
    path = config.OUTPUTS_DIR / "conflict_risk_layer.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    # panel_date (and any other datetime column) isn't natively JSON-
    # serializable -- convert once here so every route that calls this
    # function gets clean JSON automatically, rather than each route
    # needing its own datetime-handling logic.
    for col in gdf.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        gdf[col] = gdf[col].astype(str)
    return gdf


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_panel() -> pd.DataFrame | None:
    """The full Phase 3 ML panel -- one row per (sub-county, year, month)."""
    path = config.OUTPUTS_DIR / "ml_panel.csv"
    return _read_csv_if_exists(path, parse_dates=["panel_date"])


def query_panel(
    county: str | None = None, subcounty: str | None = None,
    year_min: int | None = None, year_max: int | None = None,
) -> pd.DataFrame:
    """
    Filtered panel query -- this signature is exactly what becomes a
    SQL WHERE clause when this migrates to Postgres. Every filter
    argument is optional and additive (AND'd together).
    """
    df = get_panel()
    if df is None:
        return pd.DataFrame()
    if county:
        df = df[df["County"] == county]
    if subcounty:
        df = df[df["SubCounty"] == subcounty]
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]
    return df


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_conflicts_geocoded() -> pd.DataFrame | None:
    """
    Individual conflict records with their sub-county assignment and
    coordinates -- what the map's point layer and the conflict list/
    search view use. Joins the cleaned conflict data (dates, severity,
    Is_Composite) with the geocoding result from Phase 3, and topic/
    sentiment data if available (graceful degradation if Phase 2 NLP
    steps haven't been run).
    """
    cleaned_path = config.OUTPUTS_DIR / "conflict_cleaned.csv"
    geocoded_path = config.OUTPUTS_DIR / "conflict_geocoded.csv"
    if not cleaned_path.exists() or not geocoded_path.exists():
        return None

    cleaned = pd.read_csv(cleaned_path)
    geocoded = pd.read_csv(geocoded_path)
    merged = cleaned.merge(geocoded, on="Record_ID", how="left")

    # Optional enrichment -- attach if present, skip silently if not
    topics_safe_path = config.OUTPUTS_DIR / "conflict_topics_temporal_safe.csv"
    topics_path = config.OUTPUTS_DIR / "conflict_topics.csv"
    topics_source = topics_safe_path if topics_safe_path.exists() else topics_path
    if topics_source.exists():
        topics_full = pd.read_csv(topics_source)
        keep_cols = [c for c in ["Record_ID", "topic_id", "topic_label",
                                  "sentiment_score", "sentiment_label"]
                     if c in topics_full.columns]
        merged = merged.merge(topics_full[keep_cols], on="Record_ID",
                               how="left", suffixes=("", "_topic"))

    return merged


def query_conflicts(
    county: str | None = None, subcounty: str | None = None,
    year_min: int | None = None, year_max: int | None = None,
    topic_id: int | None = None, search: str | None = None,
) -> pd.DataFrame:
    """
    Filtered conflict record query, including free-text search over
    Incident_Summary -- this is the "search" half of the requested
    filter+search behavior. Search is a simple case-insensitive
    substring match here (fine for file-based data at this scale);
    swapping to Postgres later would naturally upgrade this to
    full-text search (to_tsvector) without changing the function's
    calling convention.
    """
    df = get_conflicts_geocoded()
    if df is None:
        return pd.DataFrame()

    if county:
        df = df[df["geo_county"] == county]
    if subcounty:
        df = df[df["geo_subcounty"] == subcounty]
    if year_min is not None:
        df = df[pd.to_datetime(df["Date_Start"]).dt.year >= year_min]
    if year_max is not None:
        df = df[pd.to_datetime(df["Date_Start"]).dt.year <= year_max]
    if topic_id is not None and "topic_id" in df.columns:
        df = df[df["topic_id"] == topic_id]
    if search:
        text_col = "Incident_Summary" if "Incident_Summary" in df.columns else None
        if text_col:
            df = df[df[text_col].fillna("").str.contains(search, case=False, na=False)]
    return df


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_model_metrics() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "model_comparison.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_shap_summary() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "shap_feature_importance.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_topic_info() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "topic_info.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_hotspot_analysis() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "hotspot_analysis.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_yearly_trend() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "yearly_trend.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_seasonal_pattern() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.OUTPUTS_DIR / "seasonal_pattern.csv")


def get_available_filters() -> dict:
    """
    Distinct values for populating filter dropdowns in the UI --
    counties, sub-counties, year range, topics. Computed from whatever
    data is actually available rather than hardcoded, so the filter
    options never drift out of sync with the real data.
    """
    panel = get_panel()
    conflicts = get_conflicts_geocoded()
    topics = get_topic_info()

    counties, subcounties, year_min, year_max = [], [], None, None
    if panel is not None and len(panel) > 0:
        counties = sorted(panel["County"].dropna().unique().tolist())
        subcounties = sorted(panel["SubCounty"].dropna().unique().tolist())
        year_min, year_max = int(panel["Year"].min()), int(panel["Year"].max())

    topic_options = []
    if topics is not None and len(topics) > 0:
        topic_options = topics[topics["Topic"] != -1][["Topic", "Name"]].to_dict("records")

    return {
        "counties": counties,
        "subcounties": subcounties,
        "year_min": year_min,
        "year_max": year_max,
        "topics": topic_options,
    }
