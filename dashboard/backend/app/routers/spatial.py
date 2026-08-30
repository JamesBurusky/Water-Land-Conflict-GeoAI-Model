from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from .. import config, data_access

router = APIRouter(prefix="/api/spatial", tags=["spatial"])


@router.get("/risk-layer")
def risk_layer(horizon_months: int = Query(1, description="Forecast horizon: 1, 3, or 6 months ahead")):
    """GeoJSON: sub-county polygons + risk_probability + risk_category
    for ONE forecast horizon. This is the map's main choropleth layer."""
    try:
        gdf = data_access.get_risk_layer(horizon_months=horizon_months)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if gdf is None:
        expected_path = config.horizon_dir("12_conflict_risk_mapping", horizon_months) / "conflict_risk_layer.geojson"
        raise HTTPException(
            404,
            f"Risk layer for the {horizon_months}-month horizon not found. "
            f"Expected file: {expected_path.resolve()} -- run 10_ml_modeling.py then "
            f"12_conflict_risk_mapping.py, or check /api/status for the full picture "
            f"of what's been run so far."
        )
    geo = data_access.geodf_to_json_safe_geojson(gdf)
    return JSONResponse(content=geo)


@router.get("/hotspots")
def hotspots():
    """Getis-Ord Gi* hotspot/coldspot classification per sub-county
    (Phase 4) -- a second, complementary map layer/toggle option."""
    df = data_access.get_hotspot_analysis()
    if df is None:
        expected_path = config.step_dir("09_exploratory_spatial_analysis") / "hotspot_analysis.csv"
        raise HTTPException(
            404,
            f"Hotspot analysis not found. Expected file: {expected_path.resolve()} -- "
            f"run 09_exploratory_spatial_analysis.py first (this file specifically is "
            f"only produced if the Getis-Ord analysis had enough sub-counties with data "
            f"to run -- check that script's own output for a 'SKIPPED' message)."
        )
    return data_access.df_to_json_records(df)
