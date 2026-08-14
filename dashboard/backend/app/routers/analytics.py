from fastapi import APIRouter, HTTPException, Query

from .. import config, data_access

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _not_found(label: str, expected_path, run_this: str) -> HTTPException:
    """Consistent, actionable 404 body: WHAT is missing, WHERE it was
    expected, and WHAT to run -- so a 404 is never a mystery."""
    return HTTPException(
        404,
        f"{label} not found. Expected file: {expected_path.resolve()} -- "
        f"run {run_this}, or check /api/status for the full picture of "
        f"what's been run so far."
    )


@router.get("/model-metrics")
def model_metrics(horizon_months: int = Query(1, description="Forecast horizon: 1, 3, or 6 months ahead")):
    """Accuracy/precision/recall/F1/ROC-AUC/PR-AUC for all three models
    (baseline/comparison/final) at ONE forecast horizon (Phase 6)."""
    try:
        df = data_access.get_model_metrics(horizon_months=horizon_months)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if df is None:
        path = config.horizon_dir("10_ml_modeling", horizon_months) / "model_comparison.csv"
        raise _not_found(f"Model metrics for the {horizon_months}-month horizon", path, "10_ml_modeling.py")
    return data_access.df_to_json_records(df)


@router.get("/confusion-matrices")
def confusion_matrices(horizon_months: int = Query(1, description="Forecast horizon: 1, 3, or 6 months ahead")):
    """TN/FP/FN/TP per model at one forecast horizon, explicitly labeled."""
    try:
        df = data_access.get_confusion_matrices(horizon_months=horizon_months)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if df is None:
        path = config.horizon_dir("10_ml_modeling", horizon_months) / "confusion_matrices.csv"
        raise _not_found(f"Confusion matrices for the {horizon_months}-month horizon", path, "10_ml_modeling.py")
    return data_access.df_to_json_records(df)


@router.get("/calibration")
def calibration(horizon_months: int = Query(1, description="Forecast horizon: 1, 3, or 6 months ahead")):
    """Reliability/calibration curve data per model at one forecast
    horizon -- is a "70% predicted risk" really observed ~70% of the time?"""
    try:
        df = data_access.get_calibration(horizon_months=horizon_months)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if df is None:
        path = config.horizon_dir("10_ml_modeling", horizon_months) / "calibration.csv"
        raise _not_found(f"Calibration data for the {horizon_months}-month horizon", path, "10_ml_modeling.py")
    return data_access.df_to_json_records(df)


@router.get("/horizon-comparison")
def horizon_comparison():
    """All three models' metrics across ALL forecast horizons (1/3/6
    months) in one table -- lets the dashboard show how performance
    degrades as the forecast window lengthens."""
    df = data_access.get_horizon_comparison_summary()
    if df is None:
        path = config.step_dir("10_ml_modeling") / "horizon_comparison_summary.csv"
        raise _not_found("Horizon comparison summary", path, "10_ml_modeling.py (produces all horizons at once)")
    return data_access.df_to_json_records(df)


@router.get("/shap-importance")
def shap_importance():
    """SHAP feature importance (Phase 7) -- for a bar/beeswarm chart."""
    df = data_access.get_shap_summary()
    if df is None:
        path = config.step_dir("11_shap_interpretability") / "shap_feature_importance.csv"
        raise _not_found("SHAP results", path, "11_shap_interpretability.py")
    return data_access.df_to_json_records(df)


@router.get("/topics")
def topics():
    """Topic modelling results (Phase 2) -- topic sizes and representative
    words, for a themes breakdown panel."""
    df = data_access.get_topic_info()
    if df is None:
        path = config.step_dir("05_topic_modelling") / "topic_info.csv"
        raise _not_found("Topic info", path, "05_topic_modelling.py (needs internet access)")
    return data_access.df_to_json_records(df)


@router.get("/yearly-trend")
def yearly_trend():
    """Yearly conflict trend (Phase 4) -- for the temporal chart panel."""
    df = data_access.get_yearly_trend()
    if df is None:
        path = config.step_dir("09_exploratory_spatial_analysis") / "yearly_trend.csv"
        raise _not_found("Yearly trend", path, "09_exploratory_spatial_analysis.py")
    return data_access.df_to_json_records(df)


@router.get("/seasonal-pattern")
def seasonal_pattern():
    """Calendar-month seasonal pattern (Phase 4)."""
    df = data_access.get_seasonal_pattern()
    if df is None:
        path = config.step_dir("09_exploratory_spatial_analysis") / "seasonal_pattern.csv"
        raise _not_found("Seasonal pattern", path, "09_exploratory_spatial_analysis.py")
    return data_access.df_to_json_records(df)


@router.get("/land-water-severity")
def land_water_severity():
    """Mean/median severity per land/water conflict domain -- are mixed
    conflicts more severe than single-domain ones?"""
    df = data_access.get_land_water_severity()
    if df is None:
        path = config.step_dir("13_land_water_relationship_analysis") / "severity_by_domain.csv"
        raise _not_found("Land-water analysis", path, "13_land_water_relationship_analysis.py")
    return data_access.df_to_json_records(df)


@router.get("/land-water-by-county")
def land_water_by_county():
    """Land/water domain distribution per county."""
    df = data_access.get_land_water_county_crosstab()
    if df is None:
        path = config.step_dir("13_land_water_relationship_analysis") / "domain_by_county.csv"
        raise _not_found("Land-water analysis", path, "13_land_water_relationship_analysis.py")
    return data_access.df_to_json_records(df)


@router.get("/land-water-yearly-trend")
def land_water_yearly_trend():
    """Land/water domain balance over time."""
    df = data_access.get_land_water_yearly_trend()
    if df is None:
        path = config.step_dir("13_land_water_relationship_analysis") / "domain_yearly_trend.csv"
        raise _not_found("Land-water analysis", path, "13_land_water_relationship_analysis.py")
    return data_access.df_to_json_records(df)
