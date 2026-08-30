"""
shap_analysis.py

Phase 7: SHAP interpretability for the final XGBoost model. Answers
the spec's question directly -- "which factors contribute most to
conflict risk" -- with per-feature, per-record attribution rather than
just a global importance ranking (which XGBoost's built-in
feature_importances_ already gave us in Phase 6, but that only says
HOW MUCH a feature matters on average, not HOW -- SHAP adds direction
and per-record variation, which is what a policy interpretation needs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    """
    TreeExplainer is exact (not approximate) for tree ensembles like
    XGBoost -- no sampling, no kernel approximation -- and fast, since
    it exploits the tree structure directly rather than treating the
    model as a black box. Returns a shap.Explanation object (not just
    a raw array) so downstream plotting functions have access to
    feature names and base values automatically.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer


def summarize_feature_importance(shap_values, feature_names: list[str]) -> pd.DataFrame:
    """
    Mean ABSOLUTE SHAP value per feature -- the standard SHAP-based
    importance ranking (magnitude of impact, regardless of direction).
    Also reports mean SIGNED SHAP value separately, since a feature
    can have large absolute impact while being ambiguous in direction
    (helps vs. hurts depending on its own value) -- collapsing both
    into one number would hide that distinction.
    """
    values = shap_values.values
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary


def generate_policy_interpretation(summary: pd.DataFrame, top_n: int = 5) -> str:
    """
    Turns the SHAP summary table into plain-language sentences for the
    thesis discussion section -- states DIRECTION explicitly (higher
    values of the feature push risk up or down), since that's the part
    a raw importance bar chart doesn't communicate on its own.
    """
    lines = [f"Top {top_n} factors contributing to predicted conflict risk "
             f"(by mean absolute SHAP value):\n"]
    for i, row in summary.head(top_n).iterrows():
        direction = ("HIGHER values of this feature INCREASE predicted risk"
                     if row["mean_signed_shap"] > 0 else
                     "HIGHER values of this feature DECREASE predicted risk"
                     if row["mean_signed_shap"] < 0 else
                     "direction is mixed/ambiguous across records")
        lines.append(f"{i + 1}. {row['feature']}: mean |SHAP| = {row['mean_abs_shap']:.4f} -- {direction}")
    return "\n".join(lines)
