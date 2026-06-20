import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from pathlib import Path
import itertools
from src.modeling import get_xy
from src.config import (
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE_TREE,
    SHAP_SAMPLE_SIZE_SUBGROUP,
    SHAP_TOP_N_FEATURES,
    SHAP_SUMMARY_MAX_DISPLAY,
    PERMUTATION_N_REPEATS,
    PFI_TOP_N,
    PLOT_DPI,
    PLOT_DPI_TEMP,
    HISTOGRAM_BINS,
    HISTOGRAM_ALPHA,
)
import shap as shap_lib

def _positive_class_shap_values(shap_vals):
    if hasattr(shap_vals, "values"):
        shap_vals = shap_vals.values
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        class_axes = [axis for axis, size in enumerate(shap_vals.shape) if size == 2]
        if not class_axes:
            raise ValueError(f"Expected one binary-output axis of size 2, got shape={shap_vals.shape}")
        shap_vals = np.take(shap_vals, indices=1, axis=class_axes[-1])
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape={shap_vals.shape}")
    return shap_vals

def save_figure_to_output_dir(fig, filename, output_dir, dpi=None):
    if output_dir is not None:
        path = Path(output_dir) / filename
        fig.savefig(path, dpi=dpi or PLOT_DPI, bbox_inches="tight")
        print("Saved:", path)


def _sample_data(X, sample_size, random_state=RANDOM_STATE):
    if len(X) > sample_size:
        return X.sample(n=sample_size, random_state=random_state)
    return X


# --- §11: Feature name extraction  ---
# get_feature_names_from_pipeline, get_transformed_matrix, etc. are already there.
def get_feature_names_from_pipeline(pipe):
    prep = pipe.named_steps["prep"]
    feature_names = []

    for name, transformer, cols in prep.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            ohe = transformer.named_steps["onehot"]
            names = ohe.get_feature_names_out(cols)
            feature_names.extend(names.tolist())
        else:
            feature_names.extend(cols)

    return np.array(feature_names)


def model_feature_importance(pipe, top_n=25):
    """Return the top model feature importances for tree or linear models."""
    feature_names = get_feature_names_from_pipeline(pipe)
    model = pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return imp.head(top_n)


# --- §12: SHAP for tree models ---

def run_shap_for_tree_model(
        pipe,
        test_df, 
        stage_name, 
        model_name,
        sample_size= SHAP_SAMPLE_SIZE_TREE, 
        output_dir=None, 
):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping SHAP for {model_name}: model is not tree-based.")
        return None

    X_test, _ = get_xy(test_df)
    X_sample = _sample_data(X_test, sample_size, RANDOM_STATE)

    shap_vals, feature_names = _compute_shap_values_for_pipe(pipe, X_sample)
    prep = pipe.named_steps["prep"]
    X_trans = prep.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    n_features = min(len(feature_names), len(mean_abs_shap))
    # mean_abs_shap = [float(sub_list[0]) for sub_list in mean_abs_shap]
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    importance_df = pd.DataFrame({
        "feature": feature_names[:n_features],
        "mean_abs_shap": mean_abs_shap[:n_features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top = importance_df.head(SHAP_TOP_N_FEATURES).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP feature importance\n{model_name} - {stage_name}")
    plt.tight_layout()
    save_figure_to_output_dir(plt.gcf(), f"shap_{model_name}_{stage_name}.png", output_dir)
    plt.show()

    try:
        shap_lib.summary_plot(
            shap_vals,
            X_trans,
            feature_names=feature_names,
            show=False,
            max_display=SHAP_SUMMARY_MAX_DISPLAY,
        )
        plt.title(f"SHAP summary: {stage_name}, {model_name}")
        plt.tight_layout()
        save_figure_to_output_dir(plt.gcf(), f"shap_summary_{model_name}_{stage_name}.png", output_dir)
        plt.show()
    except Exception as e:
        print(f"Could not create SHAP summary plot: {e}")

    return importance_df


def subgroup_shap_comparison(
        pipe,
        test_df,
        sensitive_attr,
        stage_name, 
        model_name, 
        sample_size=SHAP_SAMPLE_SIZE_SUBGROUP,
        output_dir=None
):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping subgroup SHAP for {model_name}: model is not tree-based.")
        return None

    if sample_size is None:
        sample_size = SHAP_SAMPLE_SIZE_SUBGROUP

    X_test_raw, _ = get_xy(test_df)
    X_sample_raw = _sample_data(X_test_raw, sample_size, RANDOM_STATE)
    sample_idx = X_sample_raw.index
    sensitive_values = test_df.loc[sample_idx, sensitive_attr]

    prep = pipe.named_steps["prep"]
    X_sample_trans = prep.transform(X_sample_raw)
    feature_names = get_feature_names_from_pipeline(pipe)

    if hasattr(X_sample_trans, "toarray"):
        X_sample_trans_dense = X_sample_trans.toarray()
    else:
        X_sample_trans_dense = X_sample_trans

    explainer = shap_lib.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_trans_dense)
    shap_positive = _positive_class_shap_values(shap_values)

    shap_abs_df = pd.DataFrame(np.abs(shap_positive), columns=feature_names)
    shap_abs_df[sensitive_attr] = sensitive_values.values

    grouped_shap = shap_abs_df.groupby(sensitive_attr).mean().T
    overall_mean = shap_abs_df.drop(columns=[sensitive_attr]).mean().sort_values(ascending=False)
    top_features = overall_mean.head(10).index
    plot_data = grouped_shap.loc[top_features]

    ax = plot_data.plot(kind="bar", figsize=(12, 6), width=0.8)
    plt.title(f"Top 10 Feature Importance (Mean |SHAP|) by {sensitive_attr}\nStage: {stage_name} | Model: {model_name}")
    plt.ylabel("Mean |SHAP Value|")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=sensitive_attr, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_figure_to_output_dir(plt.gcf(), f"shap_subgroup_{model_name}_{stage_name}_{sensitive_attr}.png", output_dir)
    plt.show()

    return grouped_shap


def run_permutation_importance(
        pipe,
        test_df, 
        stage_name, 
        model_name,
        output_dir=None, 
        n_repeats=None
):
    if n_repeats is None:
        n_repeats = PERMUTATION_N_REPEATS

    X_test, y_test = get_xy(test_df)
    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    pfi_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    top_pfi = pfi_df.head(PFI_TOP_N)
    plt.figure(figsize=(10, 6))
    plt.barh(top_pfi["feature"][::-1], top_pfi["importance_mean"][::-1])
    plt.xlabel("Mean decrease in balanced accuracy")
    plt.title(f"Permutation Feature Importance (PFI) - Top {PFI_TOP_N}\n{model_name} | Stage: {stage_name}")
    plt.tight_layout()
    save_figure_to_output_dir(plt.gcf(), f"pfi_{model_name}_{stage_name}.png", output_dir)
    plt.show()

    return pfi_df


def run_partial_dependence(
        pipe, 
        test_df, 
        stage_name, 
        model_name,
        features_to_plot, 
        centered=False, 
        output_dir=None
):

    X_test, _ = get_xy(test_df)
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        PartialDependenceDisplay.from_estimator(
            pipe,
            X_test,
            features_to_plot,
            kind="both",
            n_jobs=-1,
            ax=ax,
            grid_resolution=30,
            centered=centered,
        )
        plt.suptitle(f"Partial Dependence Plots: {model_name} ({stage_name})", fontsize=14)
        plt.tight_layout()
        suffix = "_centered" if centered else ""
        save_figure_to_output_dir(fig, f"ice_{model_name}_{stage_name}{suffix}.png", output_dir)
        plt.show()
    except Exception as e:
        print(f"Error generating PDP: {e}")


# --- §13: Feature-family importance ---

def broad_feature_family(feature_name):
    f = str(feature_name)
    if f.startswith("vle_") or f.startswith("clicks_"):
        return "VLE engagement"
    if f.startswith("assess_"):
        return "Assessment"
    if "gender" in f or "age_band" in f or "highest_education" in f or "imd_band" in f:
        return "Demographic"
    if "code_module" in f or "code_presentation" in f:
        return "Course context"
    if "region" in f:
        return "Region"
    return "Registration / other"


def compute_family_importance(shap_results):
    """
    shap_results: dict of {(model_name, stage_name): shap_df}
    Returns a DataFrame with family-level aggregated importance.
    """
    rows = []
    for (model_name, stage_name), shap_df in shap_results.items():
        if shap_df is None:
            continue
        tmp = shap_df.copy()
        if "feature" not in tmp.columns:
            if "mean_abs_shap" in tmp.columns:
                tmp = tmp.reset_index().rename(columns={"index": "feature"})
            else:
                vals = tmp.to_numpy()
                mean_abs = np.abs(vals).mean(axis=0) if vals.size else np.array([])
                tmp = pd.DataFrame({"feature": tmp.columns, "mean_abs_shap": mean_abs})

        tmp["family"] = tmp["feature"].apply(broad_feature_family)
        grp = tmp.groupby("family", dropna=False)["mean_abs_shap"].sum().reset_index()
        grp["model"] = model_name
        grp["stage"] = stage_name
        rows.append(grp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_family_importance(family_importance, output_dir=None):

    for (model_name, stage_name), g in family_importance.groupby(["model", "stage"]):
        g_sorted = g.sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(7, 4))
        plt.barh(g_sorted["family"], g_sorted["mean_abs_shap"])
        plt.xlabel("Total mean |SHAP value|")
        plt.title(f"Feature family importance\n{model_name} - {stage_name}")
        plt.tight_layout()
        save_figure_to_output_dir(plt.gcf(), f"family_importance_{model_name}_{stage_name}.png", output_dir)
        plt.show()


def _compute_shap_values_for_pipe(pipe, X_sample):
    """
    Helper to compute SHAP values for a logistic reg. in a pipeline.
    Return numpy array and feature_names array.
    """
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    
    X_trans = prep.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    
    explainer = shap_lib.TreeExplainer(model)
    shap_vals = _positive_class_shap_values(explainer.shap_values(X_trans))
    
    feature_names = get_feature_names_from_pipeline(pipe)
    return shap_vals, feature_names


def plot_shap_distributions(models_dict, test_df, sample_size=None, output_dir=None):
    """
    For each model pair and feature, plot side-by-side distributions of SHAP values.
    Shows immediately where model A overlays low/high SHAP on model B.
    """
    if sample_size is None:
        sample_size = SHAP_SAMPLE_SIZE_TREE

    X_test, _ = get_xy(test_df)
    X_sample = _sample_data(X_test, sample_size, RANDOM_STATE)
    
    shap_data = {}
    for model_name, pipe in models_dict.items():
        model = pipe.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            continue
        
        shap_vals, feature_names = _compute_shap_values_for_pipe(pipe, X_sample)
        shap_data[model_name] = (shap_vals, feature_names)
    
    # Plot pairwise comparisons
    for model_a, model_b in itertools.combinations(shap_data.keys(), 2):
        shap_vals_a, feature_names_a = shap_data[model_a]
        shap_vals_b, _ = shap_data[model_b]
        
        # Get top 6 features by importance (mean absolute SHAP from model_a)
        mean_abs = np.abs(shap_vals_a).mean(axis=0)
        top_indices = np.argsort(mean_abs)[-6:][::-1]
        
        # 2 plots per row: 6 features = 3 rows, 2 cols
        fig, axes = plt.subplots(6, 1, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            vals_a = shap_vals_a[:, feat_idx]
            vals_b = shap_vals_b[:, feat_idx]
            
            ax.hist(vals_a, alpha=HISTOGRAM_ALPHA, label=model_a, bins=HISTOGRAM_BINS, edgecolor='black')
            ax.hist(vals_b, alpha=HISTOGRAM_ALPHA, label=model_b, bins=HISTOGRAM_BINS, edgecolor='black')
            ax.set_xlabel("SHAP value")
            ax.set_ylabel("Frequency (# samples)")
            ax.set_title(f"{feature_names_a[feat_idx]}")
            ax.legend()
        
        plt.suptitle(f"SHAP Distribution: {model_a} vs {model_b}", fontsize=14)
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/shap_compare_{model_a}_vs_{model_b}.png", dpi=PLOT_DPI_TEMP)
        plt.show()
