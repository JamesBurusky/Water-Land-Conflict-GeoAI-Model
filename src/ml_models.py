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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
import xgboost as xgb


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


def train_xgboost(X_train, y_train, cv_splits: int = 5) -> tuple:
    # scale_pos_weight addresses class imbalance (conflict onset is
    # necessarily rare relative to non-onset sub-county-months) --
    # computed from the actual training data, not assumed, since the
    # true imbalance ratio should come from what's really observed.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(
        xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
        param_grid, cv=get_cv_splitter(cv_splits), scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  XGBoost best params: {grid.best_params_} "
          f"(CV ROC-AUC: {grid.best_score_:.3f}, scale_pos_weight={scale_pos_weight:.2f})")
    return grid.best_estimator_, None


def evaluate_model(model, scaler, X_test, y_test, model_name: str) -> dict:
    """
    Computes every metric the project spec asks for. Precision/recall/
    F1 use zero_division=0 (a model that never predicts the positive
    class gets 0, not an error) -- meaningful on an imbalanced target
    where that's a real possible (bad) outcome, not an edge case to hide.
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_proba
