from fastapi import APIRouter, Query

from .. import data_access

router = APIRouter(prefix="/api/panel", tags=["panel"])


@router.get("")
def panel(
    county: str | None = None,
    subcounty: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = Query(1000, le=10000),
):
    """
    Filtered rows from the sub-county x month x year ML panel --
    powers time-series charts (rainfall, NDVI, conflict persistence
    over time) that respond to the map's region selection and any
    date-range filter.
    """
    df = data_access.query_panel(county=county, subcounty=subcounty,
                                  year_min=year_min, year_max=year_max)
    df = df.sort_values("panel_date").head(limit)
    return data_access.df_to_json_records(df)


@router.get("/filters")
def available_filters():
    """Distinct counties/sub-counties/year range/topics -- populates
    the filter sidebar's dropdown options from real data."""
    return data_access.get_available_filters()


@router.get("/summary")
def panel_summary(
    county: str | None = None,
    subcounty: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
):
    """
    Aggregated stats for a KPI/summary card (total onset months, avg
    persistence, etc.) for the currently selected region/date range --
    cheaper for the frontend than pulling the full filtered panel just
    to sum it.
    """
    df = data_access.query_panel(county=county, subcounty=subcounty,
                                  year_min=year_min, year_max=year_max)
    if len(df) == 0:
        return {}

    def safe_float(series):
        """Returns None instead of a raw NaN (which breaks strict JSON
        encoding) when a column is missing or entirely empty."""
        if series is None or not series.notna().any():
            return None
        return float(series.mean())

    return {
        "total_months": len(df),
        "total_onset_months": int(df["conflict_onset"].sum()) if "conflict_onset" in df else None,
        "avg_persistence": safe_float(df.get("conflict_persistence")),
        "avg_ndvi": safe_float(df.get("Mean_NDVI")),
        "avg_rainfall_anomaly_pct": safe_float(df.get("Rainfall_Anomaly_Percent")),
    }
