"""
main.py

Entry point. Run locally with:
    uvicorn app.main:app --reload --port 8000
(from the dashboard/backend/ directory)

Interactive API docs auto-generated at /docs once running -- useful
for the frontend developer (you) to see exactly what's available
without reading route code.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import spatial, panel, conflicts, analytics

app = FastAPI(
    title="GeoAI Conflict Dashboard API",
    description="Serves data from the water-land conflict prediction "
                 "pipeline (file-based now, Postgres later -- see data_access.py).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(spatial.router)
app.include_router(panel.router)
app.include_router(conflicts.router)
app.include_router(analytics.router)


@app.get("/api/status")
def pipeline_status():
    """
    Full per-step, per-horizon status of the pipeline -- what the
    frontend uses to show a clear checklist and to disable/grey out
    horizons that have no data yet, instead of surfacing a bare 404
    with no explanation of what's actually missing.
    """
    from . import data_access
    status = data_access.get_pipeline_status()
    if not status["outputs_dir_exists"]:
        status["diagnostic"] = (
            f"outputs_dir ({status['outputs_dir']}) does not exist at all -- "
            f"either the pipeline hasn't been run yet, or OUTPUTS_DIR is "
            f"pointed at the wrong location. Check the OUTPUTS_DIR/DATA_DIR "
            f"environment variables if you've set them explicitly."
        )
    return status


@app.get("/api/health")
def health():
    """Simple liveness check -- also reports which pipeline outputs
    are currently available, so the frontend can gracefully hide
    panels for phases that haven't been run yet rather than erroring."""
    from . import data_access
    horizons_available = {
        h: data_access.get_risk_layer(horizon_months=h) is not None
        for h in config.FORECAST_HORIZONS_MONTHS
    }
    return {
        "status": "ok",
        "outputs_dir": str(config.OUTPUTS_DIR.resolve()),
        "forecast_horizons_months": config.FORECAST_HORIZONS_MONTHS,
        "horizons_available": horizons_available,
        "available": {
            "risk_layer": any(horizons_available.values()),
            "panel": data_access.get_panel() is not None,
            "conflicts": data_access.get_conflicts_geocoded() is not None,
            "model_metrics": data_access.get_model_metrics() is not None,
            "shap": data_access.get_shap_summary() is not None,
            "topics": data_access.get_topic_info() is not None,
            "hotspots": data_access.get_hotspot_analysis() is not None,
            "yearly_trend": data_access.get_yearly_trend() is not None,
            "seasonal_pattern": data_access.get_seasonal_pattern() is not None,
            "land_water_analysis": data_access.get_land_water_severity() is not None,
        },
    }
