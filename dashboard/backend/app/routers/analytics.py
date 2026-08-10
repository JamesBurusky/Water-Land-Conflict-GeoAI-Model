from fastapi import APIRouter, HTTPException

from .. import data_access

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/model-metrics")
def model_metrics():
    """Accuracy/precision/recall/F1/ROC-AUC/PR-AUC per model (Phase 6)."""
    df = data_access.get_model_metrics()
    if df is None:
        raise HTTPException(404, "Model metrics not found -- run 10_ml_modeling.py first.")
    return data_access.df_to_json_records(df)


@router.get("/shap-importance")
def shap_importance():
    """SHAP feature importance (Phase 7) -- for a bar/beeswarm chart."""
    df = data_access.get_shap_summary()
    if df is None:
        raise HTTPException(404, "SHAP results not found -- run 11_shap_interpretability.py first.")
    return data_access.df_to_json_records(df)


@router.get("/topics")
def topics():
    """Topic modelling results (Phase 2) -- topic sizes and representative
    words, for a themes breakdown panel."""
    df = data_access.get_topic_info()
    if df is None:
        raise HTTPException(404, "Topic info not found -- run 05_topic_modelling.py first.")
    return data_access.df_to_json_records(df)


@router.get("/yearly-trend")
def yearly_trend():
    """Yearly conflict trend (Phase 4) -- for the temporal chart panel."""
    df = data_access.get_yearly_trend()
    if df is None:
        raise HTTPException(404, "Yearly trend not found -- run 09_exploratory_spatial_analysis.py first.")
    return data_access.df_to_json_records(df)


@router.get("/seasonal-pattern")
def seasonal_pattern():
    """Calendar-month seasonal pattern (Phase 4)."""
    df = data_access.get_seasonal_pattern()
    if df is None:
        raise HTTPException(404, "Seasonal pattern not found -- run 09_exploratory_spatial_analysis.py first.")
    return data_access.df_to_json_records(df)
