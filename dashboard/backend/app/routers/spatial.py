from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import data_access

router = APIRouter(prefix="/api/spatial", tags=["spatial"])


@router.get("/risk-layer")
def risk_layer():
    """GeoJSON: sub-county polygons + risk_probability + risk_category.
    This is the map's main choropleth layer."""
    gdf = data_access.get_risk_layer()
    if gdf is None:
        raise HTTPException(404, "Risk layer not found -- run 12_conflict_risk_mapping.py first.")
    # __geo_interface__ can carry raw NaN floats in properties (e.g. a
    # sub-county with no prediction yet), which breaks Starlette's
    # strict JSON encoding the same way a plain DataFrame's NaN does --
    # same fix, applied via the shared helper before building the GeoJSON.
    geo = data_access.geodf_to_json_safe_geojson(gdf)
    return JSONResponse(content=geo)


@router.get("/hotspots")
def hotspots():
    """Getis-Ord Gi* hotspot/coldspot classification per sub-county
    (Phase 4) -- a second, complementary map layer/toggle option."""
    df = data_access.get_hotspot_analysis()
    if df is None:
        raise HTTPException(404, "Hotspot analysis not found -- run 09_exploratory_spatial_analysis.py first.")
    return data_access.df_to_json_records(df)
