import numpy as np
import pandas as pd


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
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)

    return np.array(feature_names, dtype=object)


def _to_dense_if_needed(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def get_transformed_matrix(pipe, X_raw, make_dense=True):
    prep = pipe.named_steps["prep"]
    X_trans = prep.transform(X_raw)
    if make_dense:
        X_trans = _to_dense_if_needed(X_trans)
    return X_trans


def get_transformed_matrix_and_feature_names(pipe, X_raw, make_dense=True):
    X_trans = get_transformed_matrix(pipe, X_raw, make_dense=make_dense)
    feature_names = get_feature_names_from_pipeline(pipe)
    return X_trans, feature_names


def get_feature_groups_from_pipeline(pipe):
    prep = pipe.named_steps["prep"]
    groups = {}
    running_idx = 0

    for name, transformer, cols in prep.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            ohe = transformer.named_steps["onehot"]
            names = ohe.get_feature_names_out(cols).tolist()
            for i, full_name in enumerate(names):
                base = full_name.split("_", 1)[0]
                groups.setdefault(base, []).append(running_idx + i)
            running_idx += len(names)
        else:
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                for c in list(cols):
                    groups.setdefault(str(c), []).append(running_idx)
                    running_idx += 1
            else:
                groups.setdefault(str(cols), []).append(running_idx)
                running_idx += 1

    return groups


def grouped_permutation_importance_transformed(
    pipe,
    X_raw,
    y_true,
    groups=None,
    score_fn=None,
    n_repeats=10,
    sample_size=2000,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    if score_fn is None:
        from sklearn.metrics import balanced_accuracy_score

        def score_fn(y_t, y_p):
            return balanced_accuracy_score(y_t, y_p)

    if groups is None:
        groups = get_feature_groups_from_pipeline(pipe)

    if hasattr(X_raw, "sample"):
        n = min(sample_size, len(X_raw))
        X_raw_s = X_raw.sample(n=n, random_state=random_state)
        y_true_s = y_true.loc[X_raw_s.index]
    else:
        n = min(sample_size, len(X_raw))
        idx = rng.choice(len(X_raw), size=n, replace=False)
        X_raw_s = X_raw[idx]
        y_true_s = y_true[idx]

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    X0 = prep.transform(X_raw_s)
    X0 = _to_dense_if_needed(X0)
    y0 = np.asarray(y_true_s)

    base_pred = model.predict(X0)
    base_score = score_fn(y0, base_pred)

    rows = []
    for group_name, col_idxs in groups.items():
        col_idxs = np.array(col_idxs, dtype=int)
        scores = []
        for _ in range(n_repeats):
            Xp = X0.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, col_idxs] = Xp[perm][:, col_idxs]
            yp = model.predict(Xp)
            scores.append(score_fn(y0, yp))

        rows.append(
            {
                "feature_group": group_name,
                "importance_mean": float(base_score - np.mean(scores)),
                "importance_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "base_score": float(base_score),
                "n": int(len(X0)),
                "n_repeats": int(n_repeats),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def try_import_shap():
    try:
        import shap

        return shap
    except Exception:
        return None


def shap_linear_global(
    pipe,
    X_background_raw,
    X_explain_raw,
    max_display=25,
):
    shap = try_import_shap()
    if shap is None:
        raise ImportError("shap is not installed")

    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not look linear (missing coef_)")

    prep = pipe.named_steps["prep"]
    Xb = _to_dense_if_needed(prep.transform(X_background_raw))
    Xe = _to_dense_if_needed(prep.transform(X_explain_raw))
    feature_names = get_feature_names_from_pipeline(pipe)

    try:
        explainer = shap.LinearExplainer(model, Xb, feature_perturbation="interventional")
    except Exception:
        explainer = shap.Explainer(model, Xb)

    sv = explainer(Xe)

    vals = np.array(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    shap_df = pd.DataFrame(vals, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    mean_abs.columns = ["feature", "mean_abs_shap"]

    return mean_abs.head(max_display), sv


def shap_local_explanations(
    pipe,
    X_background_raw,
    X_explain_raw,
    row_indices,
    max_display=15,
):
    shap = try_import_shap()
    if shap is None:
        raise ImportError("shap is not installed")

    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]

    Xb = _to_dense_if_needed(prep.transform(X_background_raw))
    Xe = _to_dense_if_needed(prep.transform(X_explain_raw))
    feature_names = get_feature_names_from_pipeline(pipe)

    try:
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, Xb, feature_perturbation="interventional")
        else:
            explainer = shap.Explainer(model, Xb)
    except Exception:
        explainer = shap.Explainer(model, Xb)

    sv = explainer(Xe)
    base_values = sv.base_values
    values = np.array(sv.values)

    if values.ndim == 3:
        values = values[:, :, 1]
        if isinstance(base_values, np.ndarray) and base_values.ndim == 2:
            base_values = base_values[:, 1]

    out = []
    for idx in row_indices:
        row_vals = values[idx]
        top_idx = np.argsort(np.abs(row_vals))[::-1][:max_display]
        out.append(
            {
                "row_index": int(idx),
                "base_value": float(base_values[idx]) if np.ndim(base_values) else float(base_values),
                "contrib": pd.DataFrame(
                    {
                        "feature": feature_names[top_idx],
                        "shap_value": row_vals[top_idx],
                    }
                ),
            }
        )
    return out, sv


def train_surrogate_tree(
    blackbox_pipe,
    X_train_raw,
    max_depth=3,
    min_samples_leaf=50,
    random_state=42,
):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline

    y_hat = blackbox_pipe.predict(X_train_raw)
    prep = blackbox_pipe.named_steps["prep"]

    surrogate = Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )
    surrogate.fit(X_train_raw, y_hat)
    return surrogate


def surrogate_tree_rules(surrogate_pipe, max_depth=3):
    from sklearn.tree import export_text

    feature_names = get_feature_names_from_pipeline(surrogate_pipe).tolist()
    return export_text(surrogate_pipe.named_steps["model"], feature_names=feature_names, max_depth=max_depth)


def calibration_curve_report(pipe, X_test_raw, y_test, n_bins=10):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    proba = pipe.predict_proba(X_test_raw)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_test, proba)
    return {
        "fraction_of_positives": frac_pos,
        "mean_predicted_value": mean_pred,
        "brier_score": float(brier),
    }


def logistic_regression_odds_ratios(pipe, top_n=30, per_unit_numeric=True):
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not look like LogisticRegression (missing coef_)")

    feature_names = get_feature_names_from_pipeline(pipe)
    coefs = np.asarray(model.coef_[0], dtype=float)

    if per_unit_numeric:
        prep = pipe.named_steps["prep"]
        num_transformer = None
        num_cols = None
        for name, transformer, cols in prep.transformers_:
            if name == "num":
                num_transformer = transformer
                num_cols = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else [cols]
                break

        if num_transformer is not None and hasattr(num_transformer, "named_steps"):
            scaler = num_transformer.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "scale_"):
                scales = dict(zip([str(c) for c in num_cols], scaler.scale_))
                adjusted = []
                for fname, c in zip(feature_names.tolist(), coefs.tolist()):
                    base = fname.split("_", 1)[0]
                    if base in scales and scales[base] not in (0, None):
                        adjusted.append(float(c) / float(scales[base]))
                    else:
                        adjusted.append(float(c))
                coefs = np.asarray(adjusted, dtype=float)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_log_odds": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def what_if_one_feature(pipe, x_row_raw, feature, grid_values):
    if isinstance(x_row_raw, pd.Series):
        row = x_row_raw.to_frame().T.copy()
    else:
        row = x_row_raw.copy()

    probs = []
    for v in grid_values:
        row.loc[row.index[0], feature] = v
        probs.append(float(pipe.predict_proba(row)[0, 1]))

    return pd.DataFrame({"value": list(grid_values), "predicted_risk": probs})
