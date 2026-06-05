# ============================================================
# Train/test split, preprocessing, model building and evaluation
# ============================================================

import subprocess

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import KEYS, RANDOM_STATE, XGBOOST_AVAILABLE

if XGBOOST_AVAILABLE:
    from xgboost import XGBClassifier

DROP_COLS = KEYS + ["target_unsuccessful", "stage"]


def student_train_test_split(base, test_size=0.25):
    """Split by unique id_student to prevent data leakage."""
    unique_students = base.groupby("id_student", as_index=False)["target_unsuccessful"].max()
    train_students, test_students = train_test_split(
        unique_students[["id_student"]],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=unique_students["target_unsuccessful"],
    )
    return train_students, test_students


def apply_split(df, train_students, test_students):
    train = df.merge(train_students, on="id_student", how="inner")
    test = df.merge(test_students, on="id_student", how="inner")
    return train, test


def get_xy(df):
    y = df["target_unsuccessful"].astype(int)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    return X, y


def make_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def build_models(X_train):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBOOST_AVAILABLE:
        try:
            subprocess.check_output("nvidia-smi")
            tree_method = "hist"
            device = "cuda"
            print("GPU detected! XGBoost will use GPU acceleration.")
        except Exception:
            tree_method = "auto"
            device = "cpu"

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method=tree_method,
            device=device,
        )

    pipelines = {
        name: Pipeline(steps=[("prep", make_preprocessor(X_train)[0]), ("model", model)])
        for name, model in models.items()
    }
    return pipelines


def evaluate_predictions(y_true, y_pred, y_prob=None):
    out = {
        "accuracy":            accuracy_score(y_true, y_pred),
        "balanced_accuracy":   balanced_accuracy_score(y_true, y_pred),
        "precision_at_risk":   precision_score(y_true, y_pred, zero_division=0),
        "recall_at_risk":      recall_score(y_true, y_pred, zero_division=0),
        "f1_at_risk":          f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def fit_and_evaluate_stage(stage_name, train_df, test_df):
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    pipelines = build_models(X_train)
    fitted = {}
    rows = []

    for model_name, pipe in pipelines.items():
        print(f"Training {model_name} for {stage_name}...")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["stage"] = stage_name
        metrics["model"] = model_name
        rows.append(metrics)
        fitted[model_name] = pipe

    return pd.DataFrame(rows), fitted
