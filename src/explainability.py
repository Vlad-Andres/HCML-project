import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# --- §12: SHAP for tree models ---

def run_shap_for_tree_model(pipe, test_df, stage_name, model_name,
                            sample_size=800, output_dir=None, shap_available=True):
    if not shap_available:
        print("SHAP is not available.")
        return None

    try:
        import shap as shap_lib
    except ImportError:
        print("SHAP is not available.")
        return None

    from src.modeling import get_xy
    # from src.explainability import get_feature_names_from_pipeline

    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping SHAP for {model_name}: model is not tree-based.")
        return None

    X_test, y_test = get_xy(test_df)

    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test

    prep = pipe.named_steps["prep"]
    X_trans = prep.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = get_feature_names_from_pipeline(pipe)

    explainer = shap_lib.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values
        if getattr(shap_vals, "ndim", 0) == 3:
            shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    n_features = min(len(feature_names), len(mean_abs_shap))
    importance_df = pd.DataFrame({
        "feature": feature_names[:n_features],
        "mean_abs_shap": mean_abs_shap[:n_features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_n = 20
    top = importance_df.head(top_n).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP feature importance\n{model_name} – {stage_name}")
    plt.tight_layout()
    if output_dir is not None:
        from pathlib import Path
        path = Path(output_dir) / f"shap_{model_name}_{stage_name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print("Saved:", path)
    plt.show()

    return importance_df


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
        plt.title(f"Feature family importance\n{model_name} – {stage_name}")
        plt.tight_layout()
        if output_dir is not None:
            from pathlib import Path
            path = Path(output_dir) / f"family_importance_{model_name}_{stage_name}.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            print("Saved:", path)
        plt.show()
