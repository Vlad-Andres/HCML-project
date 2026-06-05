# ============================================================
# Fairness and subgroup metrics (§8 + §10)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MIN_GROUP_SIZE, SENSITIVE_ATTRS
from src.modeling import get_xy


def safe_confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    selection_rate = float(np.mean(y_pred))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr_recall_at_risk": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "tnr": tnr,
        "selection_rate_pred_at_risk": selection_rate,
    }


def subgroup_metric_table(test_df, fitted_model, sensitive_attr,
                          stage_name, model_name, min_group_size=MIN_GROUP_SIZE):
    X_test, y_true = get_xy(test_df)
    y_pred = fitted_model.predict(X_test)
    y_prob = (
        fitted_model.predict_proba(X_test)[:, 1]
        if hasattr(fitted_model, "predict_proba")
        else np.full(len(y_true), float("nan"))
    )

    tmp = test_df[[sensitive_attr]].copy()
    tmp["y_true"] = np.asarray(y_true)
    tmp["y_pred"] = np.asarray(y_pred)
    tmp["y_prob"] = np.asarray(y_prob)

    rows = []
    for group_value, g in tmp.groupby(sensitive_attr, dropna=False):
        if len(g) < min_group_size:
            continue

        cm = safe_confusion_metrics(g["y_true"], g["y_pred"])
        row = {
            "stage": stage_name,
            "model": model_name,
            "sensitive_attr": sensitive_attr,
            "group": str(group_value),
            "n": len(g),
            "base_rate_true_at_risk": g["y_true"].mean(),
            "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
            "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"]),
            "precision_at_risk": precision_score(g["y_true"], g["y_pred"], zero_division=0),
            "recall_at_risk": recall_score(g["y_true"], g["y_pred"], zero_division=0),
            "f1_at_risk": f1_score(g["y_true"], g["y_pred"], zero_division=0),
        }
        row.update(cm)

        try:
            row["roc_auc"] = roc_auc_score(g["y_true"], g["y_prob"])
        except Exception:
            row["roc_auc"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def fairness_gap_summary(subgroup_df):
    """Summarise group gaps (max - min) for each stage-model-sensitive_attr combo."""
    metrics_for_gap = [
        "selection_rate_pred_at_risk",
        "tpr_recall_at_risk",
        "fpr",
        "fnr",
        "accuracy",
        "balanced_accuracy",
        "f1_at_risk",
        "roc_auc",
    ]

    rows = []
    group_cols = ["stage", "model", "sensitive_attr"]
    for keys_, g in subgroup_df.groupby(group_cols):
        row = dict(zip(group_cols, keys_))
        row["num_groups_included"] = g["group"].nunique()
        row["min_group_size"] = g["n"].min()
        row["max_group_size"] = g["n"].max()

        for m in metrics_for_gap:
            values = g[m].dropna()
            if len(values) >= 2:
                row[f"{m}_gap"] = values.max() - values.min()
                row[f"{m}_min"] = values.min()
                row[f"{m}_max"] = values.max()
            else:
                row[f"{m}_gap"] = float("nan")
                row[f"{m}_min"] = float("nan")
                row[f"{m}_max"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def run_fairness_analysis(stage_name, test_df, fitted_models,
                          sensitive_attrs=None):
    if sensitive_attrs is None:
        sensitive_attrs = SENSITIVE_ATTRS
    all_rows = []
    for model_name, model in fitted_models.items():
        for attr in sensitive_attrs:
            table = subgroup_metric_table(test_df, model, attr, stage_name, model_name)
            all_rows.append(table)
    return pd.concat(all_rows, ignore_index=True)


def inspect_chosen_model(subgroup_metrics, model_name=None):
    """Return the detailed subgroup table for a single model, sorted by FNR."""
    if model_name is None:
        model_name = subgroup_metrics["model"].iloc[0]
    detail = subgroup_metrics[subgroup_metrics["model"] == model_name].copy()
    detail_sorted = detail.sort_values(
        ["sensitive_attr", "stage", "fnr"], ascending=[True, True, False]
    )
    return detail_sorted[[
        "stage", "sensitive_attr", "group", "n", "base_rate_true_at_risk",
        "selection_rate_pred_at_risk", "recall_at_risk", "fnr", "fpr",
        "balanced_accuracy", "roc_auc",
    ]]
