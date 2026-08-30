"""
ml_models.py

Phase 6 steps 2-3: baseline (Logistic Regression), comparison (Random
Forest), and final (XGBoost) models for conflict onset prediction,
with cross-validation, hyperparameter tuning, and the full metric
suite the project spec calls for.

All three models share the same feature set (from ml_prep.py) and the
same temporal train/test split -- differences in reported performance
reflect genuine model capability differences, not different data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb


def find_optimal_threshold(model, scaler, X_train, y_train, cv_splits: int = 5) -> float:
    """
    Finds a better decision threshold than sklearn's default 0.5,
    which is why several model-horizon combinations in this project's
    results were found to have recall of exactly 0.0 -- at 0.5, a
    severely imbalanced classifier's predicted probabilities can sit
    below threshold for EVERY test case, so the model predicts the
    majority class unconditionally and still scores ~90% accuracy
    while being functionally useless for early warning.

    The threshold is chosen to maximise F1 on cross-validated
    out-of-fold predictions computed on the TRAINING set only, via
    TimeSeriesSplit (the same leakage-safe splitter used for
    hyperparameter search) -- never on the test set. Tuning the
    threshold against the test set would leak test-set information
    into a modelling decision, the same category of mistake as fitting
    hyperparameters against it; cross_val_predict produces predictions
    for each training fold from a model that never saw that fold
    during its own training, which is the correct analogue for a
    threshold search as it is for hyperparameter search.
    """
    X_fit = scaler.transform(X_train) if scaler is not None else X_train
    y_arr = np.asarray(y_train)
    cv = get_cv_splitter(cv_splits)

    # sklearn's cross_val_predict does NOT work with TimeSeriesSplit --
    # confirmed directly (it raises "only works for partitions"),
    # because TimeSeriesSplit's first training chunk is never itself
    # used as a validation fold, so the splits don't cover every sample
    # exactly once the way cross_val_predict requires. The fix is to
    # loop over the same folds manually, refit a fresh clone of the
    # model on each fold's training portion, predict on that fold's
    # validation portion, and accumulate only the predictions that
    # exist -- which is what cross_val_predict does internally for a
    # splitter that IS a full partition, adapted for one that isn't.
    oof_idx, oof_proba = [], []
    try:
        for train_idx, val_idx in cv.split(X_fit):
            if len(set(y_arr[train_idx])) < 2:
                # A fold whose training portion has only one class can't
                # fit most classifiers -- skip it rather than crash the
                # whole threshold search over one bad fold. Realistic at
                # short horizons/small training windows early in the
                # split sequence, not just a small-sample artifact.
                continue
            fold_model = clone(model)
            X_fold_train = X_fit[train_idx] if isinstance(X_fit, np.ndarray) else X_fit.iloc[train_idx]
            X_fold_val = X_fit[val_idx] if isinstance(X_fit, np.ndarray) else X_fit.iloc[val_idx]
            fold_model.fit(X_fold_train, y_arr[train_idx])
            oof_idx.extend(val_idx)
            oof_proba.extend(fold_model.predict_proba(X_fold_val)[:, 1])
    except Exception as e:
        print(f"  WARNING: threshold tuning failed ({e}) -- falling back to 0.5.")
        return 0.5

    if len(oof_proba) == 0:
        print("  WARNING: threshold tuning found no usable CV folds (every fold had only "
              "one class) -- falling back to 0.5.")
        return 0.5

    y_oof = y_arr[np.array(oof_idx)]
    oof_proba = np.array(oof_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_oof, oof_proba)
    # precision_recall_curve returns one more precision/recall pair than
    # thresholds (it appends the (1, 0) endpoint with no corresponding
    # threshold) -- drop that last pair so the three arrays align.
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1_scores = np.where(
        (precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0.0
    )
    if len(f1_scores) == 0 or f1_scores.max() == 0:
        # No threshold does better than predicting nothing -- fall back
        # to 0.5 rather than pick an arbitrary threshold that scored 0
        # on cross-validated data too.
        return 0.5
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def get_cv_splitter(n_splits: int = 5) -> TimeSeriesSplit:
    """
    TimeSeriesSplit, not plain k-fold -- ordinary cross-validation
    shuffles rows randomly across folds, which for panel/time-series
    data means training on some of a later period and validating on an
    earlier one (leakage in the same spirit as the same-month feature
    problem ml_prep.py fixes). TimeSeriesSplit always validates on data
    that comes AFTER what it trained on within each fold.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def train_logistic_regression(X_train, y_train, cv_splits: int = 5) -> tuple:
    """
    Baseline model. Features are standardized first (mean 0, std 1) --
    required for logistic regression's regularization to treat all
    features fairly; tree models below don't need this (they split on
    raw thresholds, unaffected by scale).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, cv=get_cv_splitter(cv_splits), scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_scaled, y_train)
    print(f"  Logistic Regression best params: {grid.best_params_} "
          f"(CV ROC-AUC: {grid.best_score_:.3f})")
    return grid.best_estimator_, scaler


def train_random_forest(X_train, y_train, cv_splits: int = 5) -> tuple:
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5],
        "class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=get_cv_splitter(cv_splits), scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Random Forest best params: {grid.best_params_} "
          f"(CV ROC-AUC: {grid.best_score_:.3f})")
    return grid.best_estimator_, None


def train_xgboost(X_train, y_train, cv_splits: int = 5, n_iter: int = 40) -> tuple:
    # scale_pos_weight addresses class imbalance (conflict onset is
    # necessarily rare relative to non-onset sub-county-months) --
    # computed from the actual training data, not assumed, since the
    # true imbalance ratio should come from what's really observed.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Widened from the original {n_estimators, max_depth, learning_rate}
    # grid -- that grid tuned tree count/depth/pace but left every
    # regularisation and sampling knob at XGBoost's default, which
    # matters more than tree count/depth for imbalanced tabular data
    # like this. subsample/colsample_bytree add randomness that can
    # reduce overfitting to the majority class; min_child_weight and
    # reg_alpha/reg_lambda directly control how aggressively the model
    # is allowed to carve out small, possibly-noisy minority-class
    # regions. The full combinatorial size of this grid is 5 x 6 x 6 x
    # 5 x 5 x 4 x 4 x 4 = 2,880 combinations -- exhaustive GridSearchCV
    # over that, x 5 CV folds x 8 forecast horizons, is not practical to
    # run. RandomizedSearchCV with a fixed n_iter samples n_iter random
    # combinations instead of every combination, trading the guarantee
    # of finding the single best combination for a search that actually
    # completes -- a standard, defensible substitution for a space this
    # size (documented in Chapter Three's methodology).
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.1, 1, 5, 10],
    }
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
        param_distributions, n_iter=n_iter, cv=get_cv_splitter(cv_splits),
        scoring="roc_auc", n_jobs=-1, random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"  XGBoost best params: {search.best_params_} "
          f"(CV ROC-AUC: {search.best_score_:.3f}, scale_pos_weight={scale_pos_weight:.2f}, "
          f"{n_iter} of 2,880 possible combinations sampled)")
    return search.best_estimator_, None


def evaluate_model(model, scaler, X_test, y_test, model_name: str, threshold: float | None = None) -> dict:
    """
    Computes every metric the project spec asks for. Precision/recall/
    F1 use zero_division=0 (a model that never predicts the positive
    class gets 0, not an error) -- meaningful on an imbalanced target
    where that's a real possible (bad) outcome, not an edge case to hide.

    threshold: if given, classification metrics (accuracy, precision,
    recall, F1, confusion matrix) use y_proba >= threshold instead of
    the model's own .predict(), which always uses 0.5 internally.
    ROC-AUC and PR-AUC are threshold-independent (they're computed over
    the whole probability range) and are identical either way -- only
    the fixed-threshold metrics change. Pass the output of
    find_optimal_threshold() here to fix the recall-collapse problem
    documented in that function's docstring; leave as None to reproduce
    the original 0.5-threshold behaviour for comparison.
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold) if threshold is not None else model.predict(X_eval)

    metrics = {
        "model": model_name,
        "threshold": threshold if threshold is not None else 0.5,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_proba


def confusion_matrix_to_frame(cm: np.ndarray, model_name: str) -> pd.DataFrame:
    """
    Turns the raw 2x2 confusion_matrix array into a labeled, exportable
    table (True Negative / False Positive / False Negative / True
    Positive named explicitly) -- the raw array alone is easy to
    transpose or misread when reported in a thesis, so this pins down
    the labeling once, here, rather than leaving it to be reconstructed
    correctly (or not) every time it's written up.
    """
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame([{
        "model": model_name,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "total": int(cm.sum()),
    }])


def compute_calibration(y_test, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability/calibration data: among predictions the model assigned
    roughly probability p, what fraction were ACTUALLY positive? A
    well-calibrated model has these two columns close together across
    the whole range. This matters specifically because scale_pos_weight
    (used in train_xgboost to handle class imbalance) systematically
    shifts predicted probabilities upward to compensate for the rare
    positive class -- useful for ranking/classification, but it means
    the raw probability should NOT be read as a literal "70% chance"
    without checking this first. Returns empty if the test set has
    only one class (calibration is undefined without both).
    """
    if len(set(y_test)) < 2:
        return pd.DataFrame(columns=["mean_predicted_probability", "fraction_of_positives", "bin_count"])
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")
    # calibration_curve doesn't return per-bin counts directly -- derive
    # them separately so the output also shows how much data backs each
    # point (a calibration point from 3 examples is much less trustworthy
    # than one from 300, and that shouldn't be invisible in the export).
    bins = pd.qcut(y_proba, q=min(n_bins, len(set(y_proba))), duplicates="drop")
    counts = pd.Series(y_proba).groupby(bins, observed=True).count().to_numpy()
    # counts may have fewer/more entries than prob_true if bins collapsed
    # differently -- align defensively rather than assume equal length.
    n = min(len(prob_true), len(counts))
    return pd.DataFrame({
        "mean_predicted_probability": prob_pred[:n],
        "fraction_of_positives": prob_true[:n],
        "bin_count": counts[:n],
    })
