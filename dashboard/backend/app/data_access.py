"""
data_access.py

THE key architectural piece: every route handler calls functions in
this module, never reads a file directly. Right now every function
reads from the pipeline's outputs/<step_name>/ folders (see
src/output_paths.py in the pipeline itself for the convention this
mirrors). When you migrate to Postgres, only this file changes --
each function's SIGNATURE and RETURN TYPE (a pandas DataFrame, or a
GeoDataFrame for spatial data) stays the same, so nothing in routers/
or the frontend needs to change.

Caching: file reads are cached in-process with a simple TTL (see
config.CACHE_TTL_SECONDS) -- the pipeline re-runs occasionally, not
per-request, so re-parsing a multi-thousand-row CSV on every single
API call would be wasted work. Restart the API process after a fresh
pipeline run to pick up new data immediately.
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
        pass
    return value


def df_to_json_records(df: pd.DataFrame) -> list[dict]:
    """
    Converts a DataFrame to a list of JSON-safe dicts. Fixes values
    AFTER calling to_dict(), not before with df.where(df.notna(), None)
    -- that approach silently fails on numeric columns (a float64
    column can't actually hold Python's None; pandas converts any
    assigned None right back to NaN, and the bad value survives to
    break FastAPI's strict JSON encoding anyway). Use this everywhere
    a route returns DataFrame rows.
    """
    records = df.to_dict(orient="records")
    return [{k: _json_safe_value(v) for k, v in record.items()} for record in records]


def geodf_to_json_safe_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Same fix as df_to_json_records, applied to a GeoDataFrame's
    GeoJSON representation."""
    geo = gdf.__geo_interface__
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        feature["properties"] = {k: _json_safe_value(v) for k, v in props.items()}
    return geo


def _ttl_cache(ttl_seconds: int):
    """Minimal TTL cache, keyed on function args (including horizon_months
    where relevant, so each horizon's data is cached separately)."""
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


def _validate_horizon(horizon_months: int) -> int:
    if horizon_months not in config.FORECAST_HORIZONS_MONTHS:
        raise ValueError(
            f"horizon_months={horizon_months} is not one of the available "
            f"forecast horizons {config.FORECAST_HORIZONS_MONTHS}"
        )
    return horizon_months


# ---------------------------------------------------------------------
# Spatial: risk layer (horizon-aware) and hotspots
# ---------------------------------------------------------------------

@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_risk_layer(horizon_months: int = 1) -> gpd.GeoDataFrame | None:
    """The Phase 8 output for ONE forecast horizon -- risk category +
    probability per sub-county, with geometry. This is what the map's
    choropleth layer renders; horizon_months selects which of the
    1/3/6-month-ahead predictions to show."""
    _validate_horizon(horizon_months)
    path = config.horizon_dir("12_conflict_risk_mapping", horizon_months) / "conflict_risk_layer.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    for col in gdf.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        gdf[col] = gdf[col].astype(str)
    return gdf


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_hotspot_analysis() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.step_dir("09_exploratory_spatial_analysis") / "hotspot_analysis.csv")


# ---------------------------------------------------------------------
# Panel (Phase 3, NLP-augmented version from step 08)
# ---------------------------------------------------------------------

@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_panel() -> pd.DataFrame | None:
    """The full, NLP-augmented ML panel -- one row per (sub-county,
    year, month). Lives in step 08's folder (06 has the pre-NLP base
    version; 08 is the complete one every downstream consumer wants)."""
    path = config.step_dir("08_nlp_panel_features") / "ml_panel.csv"
    return _read_csv_if_exists(path, parse_dates=["panel_date"])


def query_panel(
    county: str | None = None, subcounty: str | None = None,
    year_min: int | None = None, year_max: int | None = None,
) -> pd.DataFrame:
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


# ---------------------------------------------------------------------
# Conflict records
# ---------------------------------------------------------------------

def _resolve_topics_source() -> tuple[pd.DataFrame | None, str | None, str | None, Path | None]:
    """
    Picks ONE topic model's output and returns everything needed to
    use it consistently: (per-record dataframe, topic_id column name,
    topic_label column name, matching topic_info file path).

    This exists because of a real bug found during review: reading the
    per-record topic_id from one source while reading topic names/counts
    from a DIFFERENT, non-matching source silently produces nonsense --
    a topic label in a tooltip that doesn't appear in the themes list,
    "themes" that filter zero records, a topic whose name doesn't match
    its members. Concretely, two distinct mismatches were found and are
    both avoided here:

    1. 05_topic_modelling.py's saved topic_info.csv is a snapshot taken
       BEFORE outlier reassignment (reduce_outliers -> update_topics),
       which can change a topic's representative words/name as
       previously-unclustered documents join it. The per-record
       topic_id/topic_label columns in conflict_topics.csv DO reflect
       the post-reassignment state, so topic_info.csv can describe a
       topic differently than the records actually assigned to it now.
       Fixed by preferring topic_id_coarse/topic_label_coarse +
       topic_info_coarse.csv, which are computed together, at the same
       final moment, and are self-consistent by construction (also
       fewer, more coherent themes -- 10 vs ~20 -- for the same reason
       05_topic_modelling.py's own docs recommend the coarse set for
       any presentation-facing use).
    2. If 07_topic_refit_temporal_safe.py has been run, it takes
       priority (leakage-safe) -- but it fits an ENTIRELY SEPARATE
       topic model with its own numbering; topic "3" there is not
       topic "3" from 05's model. Reading 05's topic_info while
       displaying 07's topic_id values would be comparing two
       unrelated numbering schemes. Fixed by reading
       topic_info_temporal_safe.csv (07's own saved topic info) when
       07's conflict data is the one in use.
    """
    step07_dir = config.step_dir("07_topic_refit_temporal_safe")
    step05_dir = config.step_dir("05_topic_modelling")

    topics_safe_path = step07_dir / "conflict_topics_temporal_safe.csv"
    if topics_safe_path.exists():
        df = pd.read_csv(topics_safe_path)
        info_path = step07_dir / "topic_info_temporal_safe.csv"
        return df, "topic_id", "topic_label", (info_path if info_path.exists() else None)

    topics_path = step05_dir / "conflict_topics.csv"
    if topics_path.exists():
        df = pd.read_csv(topics_path)
        if "topic_id_coarse" in df.columns:
            info_path = step05_dir / "topic_info_coarse.csv"
            return df, "topic_id_coarse", "topic_label_coarse", (info_path if info_path.exists() else None)
        # Coarse reduction step hasn't run yet (older/partial run) --
        # fall back to the fine set, at least self-consistent with
        # itself even if topic_info.csv might be stale relative to it.
        info_path = step05_dir / "topic_info.csv"
        return df, "topic_id", "topic_label", (info_path if info_path.exists() else None)

    return None, None, None, None


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_conflicts_geocoded() -> pd.DataFrame | None:
    """
    Individual conflict records with their sub-county assignment and
    coordinates. Joins step 03's cleaned conflict data (dates,
    severity, Is_Composite) with step 06's geocoding result, and step
    07/05's topic/sentiment data if available.
    """
    cleaned_path = config.step_dir("03_deduplication_check") / "conflict_cleaned.csv"
    geocoded_path = config.step_dir("06_spatial_feature_engineering") / "conflict_geocoded.csv"
    if not cleaned_path.exists() or not geocoded_path.exists():
        return None

    cleaned = pd.read_csv(cleaned_path)
    geocoded = pd.read_csv(geocoded_path)
    merged = cleaned.merge(geocoded, on="Record_ID", how="left")

    # Coerce coordinates to numeric explicitly -- if even ONE row has a
    # non-numeric value, pandas silently infers the WHOLE COLUMN as
    # object/string dtype, breaking the map's numeric coordinate check
    # for every record, not just the bad one.
    for col in ["Latitude", "Longitude"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    topics_full, topic_id_col, topic_label_col, _ = _resolve_topics_source()
    if topics_full is not None:
        keep_cols = [c for c in [topic_id_col, topic_label_col, "sentiment_score", "sentiment_label"]
                     if c and c in topics_full.columns]
        topics_subset = topics_full[["Record_ID"] + keep_cols].rename(
            columns={topic_id_col: "topic_id", topic_label_col: "topic_label"}
        )
        merged = merged.merge(topics_subset, on="Record_ID", how="left", suffixes=("", "_topic"))

    # Land/water domain classification, if step 13 has run
    domain_path = config.step_dir("13_land_water_relationship_analysis") / "conflict_with_domain.csv"
    if domain_path.exists():
        domain_full = pd.read_csv(domain_path)
        keep_cols = [c for c in ["Record_ID", "conflict_domain", "matched_land_terms", "matched_water_terms"]
                     if c in domain_full.columns]
        merged = merged.merge(domain_full[keep_cols], on="Record_ID", how="left")

    return merged


def query_conflicts(
    county: str | None = None, subcounty: str | None = None,
    year_min: int | None = None, year_max: int | None = None,
    topic_id: int | None = None, search: str | None = None,
    domain: str | None = None,
) -> pd.DataFrame:
    """Filtered conflict record query, including free-text search over
    Incident_Summary and now optional filtering by land/water domain."""
    df = get_conflicts_geocoded()
    if df is None:
        return pd.DataFrame()

    if county:
        df = df[df["geo_county"] == county]
    if subcounty:
        df = df[df["geo_subcounty"] == subcounty]
    if year_min is not None or year_max is not None:
        # errors="coerce" is required, not optional, on this specific
        # dataset -- confirmed by direct reproduction: a handful of
        # rows have genuinely corrupted Date_Start values (upstream
        # data issue, documented separately), and pd.to_datetime()
        # RAISES on those without coerce rather than skipping them.
        # That exception was propagating out of every year-filtered
        # request as a 500 error -- which explains the reported "year
        # filter doesn't work, I still see too many records": the
        # request was failing silently from the user's point of view,
        # and the frontend was left showing its last successful
        # (unfiltered) result rather than the filtered one that never
        # arrived. Coerced-to-NaT rows fall out of the filter entirely
        # (NaT >= / <= any year is False), which is correct: a record
        # with no usable date can't be said to fall inside a chosen
        # year range.
        parsed_dates = pd.to_datetime(df["Date_Start"], format="%m/%d/%Y", errors="coerce")
        year_mask = pd.Series(True, index=df.index)
        if year_min is not None:
            year_mask &= parsed_dates.dt.year >= year_min
        if year_max is not None:
            year_mask &= parsed_dates.dt.year <= year_max
        df = df[year_mask]
    if topic_id is not None and "topic_id" in df.columns:
        df = df[df["topic_id"] == topic_id]
    if domain and "conflict_domain" in df.columns:
        df = df[df["conflict_domain"] == domain]
    if search:
        # Searches every field a place name or keyword could plausibly
        # live in -- not just Incident_Summary. A ward name like
        # "Mathare" or "Kamukunji" is far more likely to appear in
        # Sub_Location or Full_Text_Description than in the (often
        # more abstracted) Incident_Summary, and searching only the
        # latter silently returned zero results for exactly that kind
        # of query. Record_ID is included too, so pasting an ID works
        # as a search the same way a keyword does.
        search_cols = [c for c in ["Record_ID", "Sub_Location", "Incident_Summary",
                                    "Full_Text_Description", "NLP_Keywords"]
                       if c in df.columns]
        if search_cols:
            mask = pd.Series(False, index=df.index)
            for col in search_cols:
                mask = mask | df[col].astype(str).str.contains(search, case=False, na=False, regex=False)
            df = df[mask]
    return df


# ---------------------------------------------------------------------
# Model results (horizon-aware): metrics, confusion matrices, calibration
# ---------------------------------------------------------------------

@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_model_metrics(horizon_months: int = 1) -> pd.DataFrame | None:
    """Accuracy/precision/recall/F1/ROC-AUC/PR-AUC for all three models
    (baseline, comparison, final) at ONE forecast horizon."""
    _validate_horizon(horizon_months)
    return _read_csv_if_exists(
        config.horizon_dir("10_ml_modeling", horizon_months) / "model_comparison.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_confusion_matrices(horizon_months: int = 1) -> pd.DataFrame | None:
    """TN/FP/FN/TP per model at one horizon, explicitly labeled."""
    _validate_horizon(horizon_months)
    return _read_csv_if_exists(
        config.horizon_dir("10_ml_modeling", horizon_months) / "confusion_matrices.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_calibration(horizon_months: int = 1) -> pd.DataFrame | None:
    """Reliability/calibration curve data per model at one horizon."""
    _validate_horizon(horizon_months)
    return _read_csv_if_exists(
        config.horizon_dir("10_ml_modeling", horizon_months) / "calibration.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_horizon_comparison_summary() -> pd.DataFrame | None:
    """All three models' metrics across ALL forecast horizons in one
    table -- the evidence for how performance degrades as the
    forecast window lengthens."""
    return _read_csv_if_exists(
        config.step_dir("10_ml_modeling") / "horizon_comparison_summary.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_shap_summary() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.step_dir("11_shap_interpretability") / "shap_feature_importance.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_topic_info() -> pd.DataFrame | None:
    """
    Themes for the dashboard's "Themes" tab. Deliberately cross-checked
    against the LIVE topic_id values actually present in
    get_conflicts_geocoded() before returning -- found during review:
    topic_info.csv can list a topic that zero current records are
    actually assigned to (e.g. from a stale/partial pipeline run, or a
    topic that only existed pre-consolidation), and the dashboard would
    show it as a clickable theme that silently filters to nothing. This
    check makes that impossible regardless of which pipeline run or
    script produced the files on disk -- a universal safety net, not a
    fix for one specific staleness cause.
    """
    _, _, _, info_path = _resolve_topics_source()
    info = _read_csv_if_exists(info_path) if info_path else None
    if info is None or "Topic" not in info.columns:
        return info

    conflicts = get_conflicts_geocoded()
    if conflicts is None or "topic_id" not in conflicts.columns:
        return info

    live_topic_ids = set(conflicts["topic_id"].dropna().unique())
    return info[info["Topic"].isin(live_topic_ids)].reset_index(drop=True)


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_yearly_trend() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.step_dir("09_exploratory_spatial_analysis") / "yearly_trend.csv")


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_seasonal_pattern() -> pd.DataFrame | None:
    return _read_csv_if_exists(config.step_dir("09_exploratory_spatial_analysis") / "seasonal_pattern.csv")


# ---------------------------------------------------------------------
# Land-water relationship analysis
# ---------------------------------------------------------------------

@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_land_water_severity() -> pd.DataFrame | None:
    return _read_csv_if_exists(
        config.step_dir("13_land_water_relationship_analysis") / "severity_by_domain.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_land_water_county_crosstab() -> pd.DataFrame | None:
    return _read_csv_if_exists(
        config.step_dir("13_land_water_relationship_analysis") / "domain_by_county.csv"
    )


@_ttl_cache(config.CACHE_TTL_SECONDS)
def get_land_water_yearly_trend() -> pd.DataFrame | None:
    return _read_csv_if_exists(
        config.step_dir("13_land_water_relationship_analysis") / "domain_yearly_trend.csv"
    )


# ---------------------------------------------------------------------
# Filter options
# ---------------------------------------------------------------------

def get_pipeline_status() -> dict:
    """
    Full picture of what's actually been run -- for EVERY pipeline
    step, and for the horizon-aware steps (10, 12), a per-horizon
    breakdown. This is what turns "I got a 404" into "step 10 hasn't
    been run for the 6-month horizon yet" -- the frontend uses this to
    show a clear status list and to disable/grey out UI for horizons
    that don't have data, rather than letting the user click into a
    dead end and see a bare error.
    """
    steps = {
        "01_phase1_data_audit": (config.step_dir("01_phase1_data_audit") / "wra_cleaned.csv").exists(),
        "02_conflict_cleaning": (config.step_dir("02_conflict_cleaning") / "conflict_cleaned.csv").exists(),
        "03_deduplication_check": (config.step_dir("03_deduplication_check") / "conflict_cleaned.csv").exists(),
        "04_nlp_pipeline": (config.step_dir("04_nlp_pipeline") / "conflict_nlp_enriched.csv").exists(),
        "05_topic_modelling": (config.step_dir("05_topic_modelling") / "conflict_topics.csv").exists(),
        "06_spatial_feature_engineering": (config.step_dir("06_spatial_feature_engineering") / "ml_panel.csv").exists(),
        "07_topic_refit_temporal_safe": (config.step_dir("07_topic_refit_temporal_safe") / "conflict_topics_temporal_safe.csv").exists(),
        "08_nlp_panel_features": (config.step_dir("08_nlp_panel_features") / "ml_panel.csv").exists(),
        "09_exploratory_spatial_analysis": (config.step_dir("09_exploratory_spatial_analysis") / "yearly_trend.csv").exists(),
        "11_shap_interpretability": (config.step_dir("11_shap_interpretability") / "shap_feature_importance.csv").exists(),
        "13_land_water_relationship_analysis": (config.step_dir("13_land_water_relationship_analysis") / "conflict_with_domain.csv").exists(),
    }

    horizons_10 = {
        h: (config.horizon_dir("10_ml_modeling", h) / "xgboost_model.joblib").exists()
        for h in config.FORECAST_HORIZONS_MONTHS
    }
    horizons_12 = {
        h: (config.horizon_dir("12_conflict_risk_mapping", h) / "conflict_risk_layer.geojson").exists()
        for h in config.FORECAST_HORIZONS_MONTHS
    }

    return {
        "outputs_dir": str(config.OUTPUTS_DIR.resolve()),
        "outputs_dir_exists": config.OUTPUTS_DIR.exists(),
        "data_dir": str(config.DATA_DIR.resolve()),
        "steps": steps,
        "step_10_ml_modeling_horizons": horizons_10,
        "step_12_conflict_risk_mapping_horizons": horizons_12,
        "horizon_comparison_available": get_horizon_comparison_summary() is not None,
    }


def get_available_filters() -> dict:
    """Distinct values for populating filter dropdowns -- counties,
    sub-counties, year range, topics, land/water domains, and the
    available forecast horizons. Computed from whatever data is
    actually available rather than hardcoded."""
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

    domains = []
    n_unparseable_dates = 0
    if conflicts is not None and "conflict_domain" in conflicts.columns:
        domains = sorted(conflicts["conflict_domain"].dropna().unique().tolist())
    if conflicts is not None and "Date_Start" in conflicts.columns:
        parsed = pd.to_datetime(conflicts["Date_Start"], format="%m/%d/%Y", errors="coerce")
        n_unparseable_dates = int(conflicts["Date_Start"].notna().sum() - parsed.notna().sum())

    return {
        "counties": counties,
        "subcounties": subcounties,
        "year_min": year_min,
        "year_max": year_max,
        "topics": topic_options,
        "domains": domains,
        "horizons_months": config.FORECAST_HORIZONS_MONTHS,
        "n_unparseable_dates": n_unparseable_dates,
    }
