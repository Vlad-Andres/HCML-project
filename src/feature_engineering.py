# ============================================================
# Feature engineering: target definition + VLE/assessment/registration features
# ============================================================

import numpy as np
import pandas as pd

from src.config import KEYS


def build_base_table(student_info, courses):
    """
    Define binary target (1 = at-risk/unsuccessful, 0 = successful)
    and merge course lengths.
    """
    target_map = {
        "Withdrawn": 1,
        "Fail":      1,
        "Pass":      0,
        "Distinction": 0,
    }

    base = student_info.copy()
    base["target_unsuccessful"] = base["final_result"].map(target_map)
    base = base.dropna(subset=["target_unsuccessful"]).copy()
    base["target_unsuccessful"] = base["target_unsuccessful"].astype(int)

    base = base.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )

    print("\nBase table shape:", base.shape)
    print("Target distribution:")
    print(base["target_unsuccessful"].value_counts(normalize=False))
    print(base["target_unsuccessful"].value_counts(normalize=True))
    return base


def aggregate_vle_features(student_vle_df, vle_df, courses_df, fraction):
    """
    Aggregate VLE activity up to a fraction of the course length.

    fraction = 0.25 for early-course features.
    fraction = 1.00 for full-course features.

    Pre-course clicks (date < 0) are excluded.
    """
    sv = student_vle_df.copy()

    sv = sv.merge(
        courses_df[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )
    sv["cutoff_date"] = np.floor(sv["module_presentation_length"] * fraction)
    sv = sv[(sv["date"] >= 0) & (sv["date"] <= sv["cutoff_date"])].copy()

    if sv.empty:
        return pd.DataFrame(columns=KEYS)

    # Basic aggregate behavior.
    basic = sv.groupby(KEYS).agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_active_days=("date", "nunique"),
        vle_mean_clicks_per_record=("sum_click", "mean"),
        vle_max_clicks_per_record=("sum_click", "max"),
        vle_num_records=("sum_click", "size"),
    ).reset_index()

    basic["vle_clicks_per_active_day"] = (
        basic["vle_total_clicks"] / basic["vle_active_days"].replace(0, np.nan)
    )

    # Activity-type click features.
    sv2 = sv.merge(
        vle_df[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )

    top_types = sv2["activity_type"].value_counts().head(8).index.tolist()
    sv2["activity_type_limited"] = np.where(
        sv2["activity_type"].isin(top_types),
        sv2["activity_type"],
        "other_activity",
    )

    by_type = sv2.pivot_table(
        index=KEYS,
        columns="activity_type_limited",
        values="sum_click",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    by_type.columns = [
        c if c in KEYS else f"clicks_{str(c)}"
        for c in by_type.columns
    ]

    out = basic.merge(by_type, on=KEYS, how="left")
    return out


def aggregate_assessment_features(student_assessment_df, assessments_df, courses_df, fraction):
    """
    Aggregate assessment behavior available up to a fraction of course length.
    Only submissions observed on or before the temporal cutoff are used.
    """
    sa = student_assessment_df.copy()
    ass = assessments_df.copy()

    sa = sa.merge(
        ass[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
        on="id_assessment",
        how="left",
    )

    sa = sa.merge(
        courses_df[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )

    sa["cutoff_date"] = np.floor(sa["module_presentation_length"] * fraction)

    # Prevent data leakage: only use submissions actually observed before the cutoff.
    sa = sa[(sa["date_submitted"].notna()) & (sa["date_submitted"] <= sa["cutoff_date"])].copy()

    if sa.empty:
        return pd.DataFrame(columns=KEYS)

    sa["weighted_score"] = sa["score"] * sa["weight"] / 100.0
    sa["late_submission"] = np.where(
        sa["date"].notna(),
        (sa["date_submitted"] > sa["date"]).astype(float),
        np.nan,
    )
    sa["days_submitted_before_due"] = np.where(
        sa["date"].notna(),
        sa["date"] - sa["date_submitted"],
        np.nan,
    )

    agg = sa.groupby(KEYS).agg(
        assess_num_submitted=("id_assessment", "count"),
        assess_mean_score=("score", "mean"),
        assess_min_score=("score", "min"),
        assess_max_score=("score", "max"),
        assess_weighted_score_sum=("weighted_score", "sum"),
        assess_total_weight_seen=("weight", "sum"),
        assess_late_rate=("late_submission", "mean"),
        assess_mean_days_before_due=("days_submitted_before_due", "mean"),
    ).reset_index()

    return agg


def registration_features(student_reg_df):
    """
    Registration features that should be available before/during the course.
    date_unregistration is deliberately excluded to avoid leakage.
    """
    reg = student_reg_df[KEYS + ["date_registration"]].copy()
    reg["registered_before_start"] = (reg["date_registration"] < 0).astype(float)
    reg["days_registered_before_start"] = -reg["date_registration"]
    return reg


def build_dataset_for_stage(base, student_vle, vle, courses,
                             student_assess, assessments, student_reg,
                             fraction, stage_name):
    """Build modeling table for one temporal stage."""
    print(f"\nBuilding dataset for {stage_name}, fraction={fraction}")

    vle_feats = aggregate_vle_features(student_vle, vle, courses, fraction)
    ass_feats = aggregate_assessment_features(student_assess, assessments, courses, fraction)
    reg_feats = registration_features(student_reg)

    print("VLE features:", vle_feats.shape)
    print("Assessment features:", ass_feats.shape)
    print("Registration features:", reg_feats.shape)

    df = base.copy()
    df = df.merge(reg_feats, on=KEYS, how="left")
    df = df.merge(vle_feats, on=KEYS, how="left")
    df = df.merge(ass_feats, on=KEYS, how="left")

    candidate_cols = KEYS + [
        "target_unsuccessful",
        "module_presentation_length",
        "gender",
        "age_band",
        "highest_education",
        "imd_band",
        "region",
        "num_of_prev_attempts",
        "studied_credits",
        "date_registration",
        "registered_before_start",
        "days_registered_before_start",
    ]

    engineered_cols = [
        c for c in df.columns
        if c.startswith("vle_") or c.startswith("clicks_") or c.startswith("assess_")
    ]

    keep_cols = [c for c in candidate_cols if c in df.columns] + engineered_cols
    keep_cols = list(dict.fromkeys(keep_cols))
    out = df[keep_cols].copy()

    # Students with no observed activity/submission by the cutoff → zero, not imputed median.
    for c in engineered_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    out["stage"] = stage_name
    print("Final stage table:", out.shape)
    return out
