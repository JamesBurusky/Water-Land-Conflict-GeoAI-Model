# %% [markdown]
# # Phase 6 — Machine Learning Model Development
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Baseline (Logistic Regression), comparison (Random Forest), and
# final (XGBoost) models predicting conflict onset one month ahead.
#
# CRITICAL: features are LAGGED (see src/ml_prep.py) -- conflict_
# persistence at the exact onset month is always inflated by the
# onset event itself (confirmed by direct test: it's the value that
# results FROM the event, not a predictor of it), so every dynamic
# feature is shifted to only use information from before the month
# being predicted. Skipping this would produce deceptively good
# offline metrics that don't reflect real forecasting ability.
#
# Run this after 08_nlp_panel_features.py.

# %%
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from ml_prep import prepare_modelling_table, temporal_train_test_split, DYNAMIC_FEATURE_COLS
from ml_models import (
    train_logistic_regression, train_random_forest, train_xgboost, evaluate_model,
)

pd.set_option("display.max_columns", None)

OUT_DIR = Path("outputs")
PANEL_PATH = OUT_DIR / "ml_panel.csv"

LAG_MONTHS = 1        # predict onset at t using features from t - LAG_MONTHS
CUTOFF_YEAR = 2022     # train on years < CUTOFF_YEAR, test on years >= CUTOFF_YEAR -- ADJUST based on your data's actual year range and where you want the split
CV_SPLITS = 5

timer = Timer()
timer.__enter__()
log("=== PHASE 6: ML MODEL DEVELOPMENT STARTED ===")

# %% [markdown]
# ## 1. Load panel and prepare leakage-safe modelling table

# %%
log("STAGE 1/4: Loading panel and building lagged feature table...")
panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
table, feature_cols = prepare_modelling_table(panel, lag_months=LAG_MONTHS)

if len(table) == 0:
    print(f"\n  STOPPED: 0 rows remain after lagging and dropping missing values. "
          f"This almost always means one or more feature columns are entirely "
          f"NaN across the whole panel (e.g. NDVI/CHIRPS coverage gaps) -- check "
          f"each column in {PANEL_PATH} individually with df[col].notna().sum() "
          f"before re-running.")
    raise SystemExit(1)

print(f"\n  Features used: {feature_cols}")

target_balance = table["conflict_onset"].value_counts()
print(f"\n  Overall target balance: {target_balance.to_dict()} "
      f"({target_balance.get(True, 0) / len(table):.2%} positive)")
log("STAGE 1/4: DONE.\n")

# %% [markdown]
# ## 2. Temporal train/test split

# %%
log(f"STAGE 2/4: Splitting train (years < {CUTOFF_YEAR}) / test (years >= {CUTOFF_YEAR})...")
train, test = temporal_train_test_split(table, cutoff_year=CUTOFF_YEAR)

X_train, y_train = train[feature_cols], train["conflict_onset"].astype(int)
X_test, y_test = test[feature_cols], test["conflict_onset"].astype(int)

train_balance = y_train.value_counts()
test_balance = y_test.value_counts()
print(f"\n  Train positive rate: {train_balance.get(1, 0)}/{len(y_train)} "
      f"({train_balance.get(1, 0) / max(len(y_train), 1):.2%})")
print(f"  Test positive rate: {test_balance.get(1, 0)}/{len(y_test)} "
      f"({test_balance.get(1, 0) / max(len(y_test), 1):.2%})")

if len(set(y_test)) < 2:
    print(f"\n  WARNING: test set has only ONE class present -- ROC-AUC/PR-AUC "
          f"are undefined and metrics below will be misleading (e.g. accuracy=1.0 "
          f"is meaningless if the model just always predicts the majority class). "
          f"This means CUTOFF_YEAR={CUTOFF_YEAR} leaves too few/no positive "
          f"examples in the test period -- lower CUTOFF_YEAR or check your data's "
          f"actual onset distribution by year before trusting results below.")
log("STAGE 2/4: DONE.\n")

# %% [markdown]
# ## 3. Train and evaluate all three models

# %%
log(f"STAGE 3/4: Training models (cv_splits={CV_SPLITS}, this may take a while "
    f"on the full dataset)...")

print("\n--- Logistic Regression (baseline) ---")
lr_model, lr_scaler = train_logistic_regression(X_train, y_train, cv_splits=CV_SPLITS)
lr_metrics, lr_cm, lr_proba = evaluate_model(lr_model, lr_scaler, X_test, y_test, "LogisticRegression")

print("\n--- Random Forest (comparison) ---")
rf_model, rf_scaler = train_random_forest(X_train, y_train, cv_splits=CV_SPLITS)
rf_metrics, rf_cm, rf_proba = evaluate_model(rf_model, rf_scaler, X_test, y_test, "RandomForest")

print("\n--- XGBoost (final) ---")
xgb_model, xgb_scaler = train_xgboost(X_train, y_train, cv_splits=CV_SPLITS)
xgb_metrics, xgb_cm, xgb_proba = evaluate_model(xgb_model, xgb_scaler, X_test, y_test, "XGBoost")

results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
print("\n  Model comparison:")
print(results.to_string(index=False))
results.to_csv(OUT_DIR / "model_comparison.csv", index=False)

import joblib
joblib.dump({"model": xgb_model, "scaler": xgb_scaler, "feature_cols": feature_cols},
            OUT_DIR / "xgboost_model.joblib")
log(f"STAGE 3/4: DONE. Saved -> {OUT_DIR / 'model_comparison.csv'}, "
    f"{OUT_DIR / 'xgboost_model.joblib'}\n")

# %% [markdown]
# ## 4. Visualizations — ROC/PR curves, confusion matrices, feature importance

# %%
log("STAGE 4/4: Generating evaluation visualizations...")

has_both_classes = len(set(y_test)) > 1

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

if has_both_classes:
    for name, proba, color in [("LR", lr_proba, "#264653"), ("RF", rf_proba, "#2a9d8f"),
                                 ("XGB", xgb_proba, "#e76f51")]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=name, color=color)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC curves"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()

    for name, proba, color in [("LR", lr_proba, "#264653"), ("RF", rf_proba, "#2a9d8f"),
                                 ("XGB", xgb_proba, "#e76f51")]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=name, color=color)
    axes[1].set_title("Precision-Recall curves")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
else:
    axes[0].text(0.5, 0.5, "Test set has only one\nclass -- ROC undefined",
                 ha="center", va="center"); axes[0].axis("off")
    axes[1].text(0.5, 0.5, "Test set has only one\nclass -- PR curve undefined",
                 ha="center", va="center"); axes[1].axis("off")

importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
axes[2].barh(importances.index, importances.values, color="#2a9d8f")
axes[2].set_title("XGBoost feature importance")

plt.tight_layout()
save_and_display(fig, OUT_DIR / "model_evaluation.png")

log("STAGE 4/4: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 6 COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. model_comparison.csv - accuracy/precision/recall/F1/ROC-AUC/PR-AUC per model
  2. xgboost_model.joblib - the fitted final model (load with joblib.load)
  3. model_evaluation.png - ROC curves, PR curves, feature importance

REVIEW before writing up:
  - CUTOFF_YEAR ({CUTOFF_YEAR}) determines the whole train/test split --
    check the train/test positive-rate printout above; if either side has
    too few positive examples, the split isn't giving you a meaningful test.
  - LAG_MONTHS ({LAG_MONTHS}) is your forecast horizon -- 1 month ahead by
    default. Longer horizons (3, 6 months) are worth testing too and
    reporting how performance degrades with horizon length, per the
    spec's emphasis on genuine forecasting.
  - REMINDER (still outstanding, filed earlier): dominant_topic_id/
    topic_diversity were built from a BERTopic model fit on the FULL
    corpus including test-period text. For a fully rigorous temporal
    evaluation, these two features should be excluded or rebuilt from a
    topic model refit on training-period text only -- flagging this
    explicitly rather than silently leaving it unresolved.

NEXT: Phase 7 (SHAP interpretability) can now load outputs/xgboost_model.joblib.
""")
