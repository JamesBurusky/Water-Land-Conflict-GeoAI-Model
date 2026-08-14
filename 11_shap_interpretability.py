# %% [markdown]
# # Phase 7 — Model Interpretability (SHAP)
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Answers "which factors contribute most to conflict risk" with
# per-record, directional attribution -- not just the average
# importance ranking XGBoost's feature_importances_ already gave us in
# Phase 6.
#
# Run this after 10_ml_modeling.py (needs the horizon's xgboost_model.joblib).

# %%
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from ml_prep import prepare_modelling_table, temporal_train_test_split
from shap_analysis import compute_shap_values, summarize_feature_importance, generate_policy_interpretation
from output_paths import step_dir, horizon_subdir

pd.set_option("display.max_columns", None)

PANEL_PATH = step_dir("08_nlp_panel_features") / "ml_panel.csv"

# SHAP is run for ONE horizon (the primary/default forecast window) --
# not looped across all three like 10_ml_modeling.py and
# 12_conflict_risk_mapping.py, to keep this step's output focused on a
# single, clearly-stated model rather than tripling the number of SHAP
# plots. Change SHAP_HORIZON_MONTHS if a different horizon is your
# primary one for the thesis narrative.
SHAP_HORIZON_MONTHS = 1
OUT_DIR = step_dir("11_shap_interpretability")
MODEL_DIR = horizon_subdir("10_ml_modeling", SHAP_HORIZON_MONTHS)
MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

LAG_MONTHS = SHAP_HORIZON_MONTHS  # MUST match the horizon of MODEL_PATH above
CUTOFF_YEAR = 2022     # MUST match 10_ml_modeling.py's CUTOFF_YEAR
TOP_N_DEPENDENCE_PLOTS = 4

timer = Timer()
timer.__enter__()
log("=== PHASE 7: SHAP INTERPRETABILITY STARTED ===")

# %% [markdown]
# ## 1. Load model and rebuild the matching test set

# %%
log("STAGE 1/3: Loading model and rebuilding test set...")
if not MODEL_PATH.exists():
    print(f"  {MODEL_PATH} not found -- run 10_ml_modeling.py first.")
    raise SystemExit(1)

saved = joblib.load(MODEL_PATH)
model, feature_cols = saved["model"], saved["feature_cols"]
print(f"  Loaded model with {len(feature_cols)} features: {feature_cols}")

panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
table, table_feature_cols = prepare_modelling_table(panel, lag_months=LAG_MONTHS)

if set(feature_cols) != set(table_feature_cols):
    print(f"\n  WARNING: features in the saved model don't match features "
          f"currently produced by prepare_modelling_table() -- the panel may "
          f"have changed since 10_ml_modeling.py was last run (e.g. NLP "
          f"features added/removed). Re-run 10_ml_modeling.py to keep the "
          f"model and panel in sync before trusting SHAP results below.")

train, test = temporal_train_test_split(table, cutoff_year=CUTOFF_YEAR)
X_test = test[feature_cols]

if len(X_test) == 0:
    print(f"\n  STOPPED: test set is empty (0 rows for years >= {CUTOFF_YEAR}). "
          f"SHAP needs at least some test rows to explain. This is usually the "
          f"same root cause as an empty modelling table -- check that "
          f"CUTOFF_YEAR={CUTOFF_YEAR} actually falls within your panel's date "
          f"range, and that the features listed above aren't entirely NaN.")
    raise SystemExit(1)

print(f"  Test set for SHAP: {len(X_test):,} rows")
log("STAGE 1/3: DONE.\n")

# %% [markdown]
# ## 2. Compute SHAP values and summarize

# %%
log("STAGE 2/3: Computing SHAP values (TreeExplainer -- exact, not sampled)...")
shap_values, explainer = compute_shap_values(model, X_test)

summary = summarize_feature_importance(shap_values, feature_cols)
print("\n  Feature importance (mean |SHAP|) and direction:")
print(summary.to_string(index=False))

interpretation = generate_policy_interpretation(summary, top_n=min(5, len(feature_cols)))
print(f"\n{interpretation}")

summary.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)
with open(OUT_DIR / "shap_policy_interpretation.txt", "w") as f:
    f.write(interpretation)
log(f"STAGE 2/3: DONE. Saved -> {OUT_DIR / 'shap_feature_importance.csv'}, "
    f"{OUT_DIR / 'shap_policy_interpretation.txt'}\n")

# %% [markdown]
# ## 3. Visualizations — summary (beeswarm), bar chart, dependence plots

# %%
log("STAGE 3/3: Generating SHAP visualizations...")

fig = plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
save_and_display(plt.gcf(), OUT_DIR / "shap_summary_beeswarm.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(summary["feature"], summary["mean_abs_shap"], color="#2a9d8f")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP feature importance")
ax.invert_yaxis()
plt.tight_layout()
save_and_display(fig, OUT_DIR / "shap_importance_bar.png")

top_features = summary.head(TOP_N_DEPENDENCE_PLOTS)["feature"].tolist()
n_plots = len(top_features)
if n_plots > 0:
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    axes = [axes] if n_plots == 1 else axes
    for ax, feat in zip(axes, top_features):
        shap.dependence_plot(feat, shap_values.values, X_test, ax=ax, show=False)
    plt.tight_layout()
    save_and_display(fig, OUT_DIR / "shap_dependence_plots.png")

log("STAGE 3/3: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 7 COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. shap_feature_importance.csv - mean |SHAP| and mean signed SHAP per feature
  2. shap_policy_interpretation.txt - plain-language summary for your thesis
  3. shap_summary_beeswarm.png - per-record SHAP distribution per feature
  4. shap_importance_bar.png - mean |SHAP| bar chart
  5. shap_dependence_plots.png - how predicted risk changes across each
     top feature's own value range, colored by its strongest interacting feature

REVIEW before writing up:
  - The beeswarm plot is the richest single figure for your Chapter 4/7 --
    it shows both magnitude AND direction AND the spread across records,
    not just an average.
  - Cross-check shap_policy_interpretation.txt's direction claims against
    domain knowledge (e.g. does "higher rainfall anomaly -> higher risk"
    make sense for drought-driven conflict, or would you expect the
    opposite sign?) -- a surprising direction is worth investigating
    before reporting it as a finding, since it could reflect a genuine
    counterintuitive pattern OR a data/feature construction issue.
""")
