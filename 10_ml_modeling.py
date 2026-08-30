# %% [markdown]
# # Phase 6 — Machine Learning Model Development (multi-horizon)
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Trains Logistic Regression (baseline), Random Forest (comparison),
# and XGBoost (final) at EIGHT forecast horizons -- 3 through 24 months
# ahead -- so you can report how predictive performance degrades as
# the forecast window lengthens, per the project spec's emphasis on
# genuine forecasting rather than same-period nowcasting.
#
# Every model at every horizon is saved (not just the final XGBoost),
# along with its confusion matrix and calibration curve, into its own
# subfolder: outputs/10_ml_modeling/horizon_<N>month/.
#
# CRITICAL leakage note (unchanged from the original version): features
# are LAGGED -- conflict_persistence at the exact month an event starts
# is always inflated by that same event, so every dynamic feature is
# shifted to only use information from before the month being
# predicted. This is what makes the horizon actually mean "N months
# ahead" rather than "same month, dressed up as a forecast."
#
# Run this after 08_nlp_panel_features.py.

# %%
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, file-saving backend -- set BEFORE pyplot
    # is imported, and before pipeline_utils or anything else can import it first.
    # Without this, matplotlib auto-selects an interactive GUI backend (TkAgg on
    # Windows, since Tkinter ships with the standard installer), which creates
    # Tkinter-backed figure objects even though this pipeline only ever calls
    # plt.savefig(), never plt.show(). Those Tk objects are not thread-safe, and
    # this project's model training uses joblib parallelism (n_jobs=-1 in
    # GridSearchCV/RandomizedSearchCV) -- when Python garbage-collects a leftover
    # Tk figure from a worker thread instead of the main thread, Tkinter raises
    # "RuntimeError: main thread is not in main loop" during its own cleanup.
    # Confirmed as the cause of exactly that error during a real Windows run of
    # this pipeline. Agg has no GUI/threading dependency at all, so this class of
    # error becomes structurally impossible, not just less likely.
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from ml_prep import prepare_modelling_table, temporal_train_test_split
from ml_models import (
    train_logistic_regression, train_random_forest, train_xgboost, evaluate_model,
    find_optimal_threshold,
    confusion_matrix_to_frame, compute_calibration,
)
from output_paths import step_dir, horizon_subdir

pd.set_option("display.max_columns", None)

STEP_NAME = "10_ml_modeling"
PANEL_PATH = step_dir("08_nlp_panel_features") / "ml_panel.csv"

HORIZONS_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]  # 3-month steps out to 2 years -- widened from
                                                    # [1, 3, 6] specifically because those three were
                                                    # too close together to show a meaningful reliability
                                                    # trend (the "almost flat" lines were partly a real
                                                    # finding -- performance genuinely doesn't degrade much
                                                    # over just 1-6 months -- and partly an artifact of not
                                                    # testing far enough out to see it change)
CUTOFF_YEAR = 2022            # train on years < CUTOFF_YEAR, test on years >= CUTOFF_YEAR -- ADJUST to your data's actual year range
CV_SPLITS = 5

timer = Timer()
timer.__enter__()
log("=== PHASE 6: MULTI-HORIZON ML MODEL DEVELOPMENT STARTED ===")

# %% [markdown]
# ## Load panel once, reused for every horizon

# %%
log("Loading panel...")
panel = pd.read_csv(PANEL_PATH, parse_dates=["panel_date"])
print(f"  Panel: {len(panel):,} rows")


def run_one_horizon(horizon_months: int) -> dict | None:
    """
    Full train/evaluate/save cycle for one forecast horizon. Returns a
    dict of headline metrics per model (used to build the cross-horizon
    comparison at the end) or None if this horizon couldn't be run
    (e.g. an empty modelling table).
    """
    out_dir = horizon_subdir(STEP_NAME, horizon_months)
    log(f"--- HORIZON: {horizon_months} month(s) ahead -> {out_dir} ---")

    table, feature_cols = prepare_modelling_table(panel, lag_months=horizon_months)
    if len(table) == 0:
        print(f"  SKIPPED horizon={horizon_months}: 0 rows remain after lagging/dropping "
              f"missing values (likely an entirely-NaN feature column, e.g. an NDVI/CHIRPS "
              f"coverage gap -- check outputs column-by-column before re-running).")
        return None

    train, test = temporal_train_test_split(table, cutoff_year=CUTOFF_YEAR)
    X_train, y_train = train[feature_cols], train["conflict_onset"].astype(int)
    X_test, y_test = test[feature_cols], test["conflict_onset"].astype(int)

    print(f"  Train positive rate: {y_train.sum()}/{len(y_train)} "
          f"({y_train.sum() / max(len(y_train), 1):.2%})")
    print(f"  Test positive rate: {y_test.sum()}/{len(y_test)} "
          f"({y_test.sum() / max(len(y_test), 1):.2%})")
    if len(set(y_test)) < 2:
        print(f"  WARNING: test set has only ONE class -- ROC-AUC/PR-AUC/calibration "
              f"are undefined for this horizon. Results below will be reported but are "
              f"not meaningful; adjust CUTOFF_YEAR if this persists.")

    models = {}
    all_metrics = []
    all_cms = []
    all_calibration = []

    print("\n  --- Logistic Regression (baseline) ---")
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train, cv_splits=CV_SPLITS)
    lr_threshold = find_optimal_threshold(lr_model, lr_scaler, X_train, y_train, cv_splits=CV_SPLITS)
    print(f"  Tuned decision threshold (cross-validated on training data only): {lr_threshold:.3f}")
    lr_metrics, lr_cm, lr_proba = evaluate_model(lr_model, lr_scaler, X_test, y_test, "LogisticRegression", threshold=lr_threshold)
    lr_metrics_default, lr_cm_default, _ = evaluate_model(lr_model, lr_scaler, X_test, y_test, "LogisticRegression (default 0.5 threshold)")
    models["LogisticRegression"] = (lr_model, lr_scaler, lr_proba)

    print("\n  --- Random Forest (comparison) ---")
    rf_model, rf_scaler = train_random_forest(X_train, y_train, cv_splits=CV_SPLITS)
    rf_threshold = find_optimal_threshold(rf_model, rf_scaler, X_train, y_train, cv_splits=CV_SPLITS)
    print(f"  Tuned decision threshold (cross-validated on training data only): {rf_threshold:.3f}")
    rf_metrics, rf_cm, rf_proba = evaluate_model(rf_model, rf_scaler, X_test, y_test, "RandomForest", threshold=rf_threshold)
    rf_metrics_default, rf_cm_default, _ = evaluate_model(rf_model, rf_scaler, X_test, y_test, "RandomForest (default 0.5 threshold)")
    models["RandomForest"] = (rf_model, rf_scaler, rf_proba)

    print("\n  --- XGBoost (final) ---")
    xgb_model, xgb_scaler = train_xgboost(X_train, y_train, cv_splits=CV_SPLITS)
    xgb_threshold = find_optimal_threshold(xgb_model, xgb_scaler, X_train, y_train, cv_splits=CV_SPLITS)
    print(f"  Tuned decision threshold (cross-validated on training data only): {xgb_threshold:.3f}")
    xgb_metrics, xgb_cm, xgb_proba = evaluate_model(xgb_model, xgb_scaler, X_test, y_test, "XGBoost", threshold=xgb_threshold)
    xgb_metrics_default, xgb_cm_default, _ = evaluate_model(xgb_model, xgb_scaler, X_test, y_test, "XGBoost (default 0.5 threshold)")
    models["XGBoost"] = (xgb_model, xgb_scaler, xgb_proba)

    # Both the tuned-threshold and default-threshold (0.5) results are
    # saved side by side -- ROC-AUC/PR-AUC are identical between the
    # two (threshold-independent), but accuracy/precision/recall/F1
    # differ, often substantially where the default threshold produced
    # zero recall. Reporting both makes the improvement (or lack of
    # one, for a horizon where it doesn't help) directly visible rather
    # than silently replacing one number with another.
    all_metrics.append(lr_metrics_default)
    all_metrics.append(rf_metrics_default)
    all_metrics.append(xgb_metrics_default)

    tuned_thresholds = {
        "LogisticRegression": lr_threshold,
        "RandomForest": rf_threshold,
        "XGBoost": xgb_threshold,
    }

    for metrics, cm, (model_obj, scaler, proba), name in [
        (lr_metrics, lr_cm, models["LogisticRegression"], "LogisticRegression"),
        (rf_metrics, rf_cm, models["RandomForest"], "RandomForest"),
        (xgb_metrics, xgb_cm, models["XGBoost"], "XGBoost"),
    ]:
        all_metrics.append(metrics)
        all_cms.append(confusion_matrix_to_frame(cm, name))
        cal = compute_calibration(y_test, proba)
        cal["model"] = name
        all_calibration.append(cal)

        # EVERY model saved, not just XGBoost -- baseline and comparison
        # models are as reproducible/inspectable as the final one. The
        # tuned threshold travels with the model so anything loading
        # this file later (12_conflict_risk_mapping.py, the dashboard)
        # applies the same decision rule these results were evaluated
        # with, rather than silently reverting to 0.5.
        joblib.dump(
            {"model": model_obj, "scaler": scaler, "feature_cols": feature_cols,
             "horizon_months": horizon_months, "cutoff_year": CUTOFF_YEAR,
             "threshold": tuned_thresholds[name]},
            out_dir / f"{name.lower()}_model.joblib",
        )

    results = pd.DataFrame(all_metrics)
    print("\n  Model comparison:")
    print(results.to_string(index=False))
    results.to_csv(out_dir / "model_comparison.csv", index=False)

    cm_df = pd.concat(all_cms, ignore_index=True)
    cm_df.to_csv(out_dir / "confusion_matrices.csv", index=False)
    print(f"\n  Confusion matrices:\n{cm_df.to_string(index=False)}")

    calibration_df = pd.concat(all_calibration, ignore_index=True)
    calibration_df.to_csv(out_dir / "calibration.csv", index=False)

    # --- Visualizations for this horizon ---
    has_both_classes = len(set(y_test)) > 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    if has_both_classes:
        for name, (_, _, proba), color in [
            ("LR", models["LogisticRegression"], "#264653"),
            ("RF", models["RandomForest"], "#2a9d8f"),
            ("XGB", models["XGBoost"], "#e76f51"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, proba)
            axes[0, 0].plot(fpr, tpr, label=name, color=color)
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0, 0].set_title("ROC curves"); axes[0, 0].set_xlabel("FPR"); axes[0, 0].set_ylabel("TPR")
        axes[0, 0].legend()

        for name, (_, _, proba), color in [
            ("LR", models["LogisticRegression"], "#264653"),
            ("RF", models["RandomForest"], "#2a9d8f"),
            ("XGB", models["XGBoost"], "#e76f51"),
        ]:
            prec, rec, _ = precision_recall_curve(y_test, proba)
            axes[0, 1].plot(rec, prec, label=name, color=color)
        axes[0, 1].set_title("Precision-Recall curves")
        axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision"); axes[0, 1].legend()

        for name, color in [("LogisticRegression", "#264653"), ("RandomForest", "#2a9d8f"), ("XGBoost", "#e76f51")]:
            cal_subset = calibration_df[calibration_df["model"] == name]
            if len(cal_subset) > 0:
                axes[0, 2].plot(cal_subset["mean_predicted_probability"],
                                 cal_subset["fraction_of_positives"], "o-", label=name, color=color)
        axes[0, 2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        axes[0, 2].set_title("Calibration curves")
        axes[0, 2].set_xlabel("Mean predicted probability"); axes[0, 2].set_ylabel("Observed fraction positive")
        axes[0, 2].legend(fontsize=8)
    else:
        for ax, title in zip(axes[0], ["ROC curves", "Precision-Recall curves", "Calibration curves"]):
            ax.text(0.5, 0.5, "Test set has only one\nclass -- undefined", ha="center", va="center")
            ax.set_title(title); ax.axis("off")

    for ax, (name, cm) in zip(axes[1, :3], [("LR", lr_cm), ("RF", rf_cm), ("XGB", xgb_cm)]):
        ConfusionMatrixDisplay(cm, display_labels=["No onset", "Onset"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} confusion matrix")

    plt.suptitle(f"Model evaluation - {horizon_months}-month horizon")
    plt.tight_layout()
    save_and_display(fig, out_dir / "model_evaluation.png")

    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(importances.index, importances.values, color="#2a9d8f")
    ax2.set_title(f"XGBoost feature importance - {horizon_months}-month horizon")
    plt.tight_layout()
    save_and_display(fig2, out_dir / "feature_importance.png")

    log(f"--- HORIZON {horizon_months} month(s): DONE ---\n")
    combined = {row["model"]: row for row in all_metrics}
    combined["horizon_months"] = horizon_months
    return combined


# %% [markdown]
# ## Run every horizon

# %%
horizon_results = []
for h in HORIZONS_MONTHS:
    result = run_one_horizon(h)
    if result is not None:
        horizon_results.append(result)

# %% [markdown]
# ## Cross-horizon comparison summary

# %%
log("Building cross-horizon comparison summary...")
if horizon_results:
    rows = []
    for result in horizon_results:
        h = result["horizon_months"]
        for model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
            if model_name in result:
                row = dict(result[model_name])
                row["horizon_months"] = h
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = step_dir(STEP_NAME) / "horizon_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary[["horizon_months", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
          .to_string(index=False))

    xgb_rows = summary[summary["model"] == "XGBoost"].sort_values("horizon_months")
    if len(xgb_rows) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xgb_rows["horizon_months"], xgb_rows["roc_auc"], "o-", color="#e76f51")
        ax.set_xlabel("Forecast horizon (months ahead)")
        ax.set_ylabel("XGBoost ROC-AUC")
        ax.set_title("Performance vs. forecast horizon")
        ax.set_xticks(HORIZONS_MONTHS)
        plt.tight_layout()
        save_and_display(fig, step_dir(STEP_NAME) / "horizon_degradation.png")
    print(f"  Saved -> {summary_path}")
else:
    print("  No horizons produced results -- nothing to summarize.")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 6 (MULTI-HORIZON) COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
For each horizon in {HORIZONS_MONTHS}, outputs/{STEP_NAME}/horizon_<N>month/ contains:
  - logisticregression_model.joblib, randomforest_model.joblib, xgboost_model.joblib
    (ALL THREE saved, not just the final model)
  - model_comparison.csv        - accuracy/precision/recall/F1/ROC-AUC/PR-AUC per model
  - confusion_matrices.csv      - TN/FP/FN/TP per model, explicitly labeled
  - calibration.csv             - reliability data per model (is "70% risk" really ~70%?)
  - model_evaluation.png        - ROC, PR, calibration curves + confusion matrix heatmaps
  - feature_importance.png      - XGBoost feature importance

outputs/{STEP_NAME}/horizon_comparison_summary.csv and horizon_degradation.png
compare all three models across all horizons in one place -- the evidence
for "how does performance degrade as the forecast window lengthens."

REVIEW before writing up:
  - CUTOFF_YEAR ({CUTOFF_YEAR}) determines every horizon's train/test split --
    check the printed train/test positive-rate per horizon; if any horizon
    has too few positives on either side, its results aren't meaningful.
  - Calibration curves matter for how you present risk_probability on the
    dashboard/map: if a model is poorly calibrated, "70% risk" doesn't
    literally mean a 70% chance -- state this explicitly if so.
  - REMINDER (still outstanding unless you've run it): dominant_topic_id/
    topic_diversity should come from 07_topic_refit_temporal_safe.py's
    leakage-safe topics (08_nlp_panel_features.py prefers that file
    automatically when present) -- confirm outputs/08_nlp_panel_features/
    was built AFTER running 07, not before, for these numbers to be fully
    leakage-free.
""")
