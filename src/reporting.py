# ============================================================
# Export tables to CSV (§14)
# ============================================================

from pathlib import Path


def export_tables(performance_table, subgroup_metrics, fairness_gaps,
                  family_importance=None, output_dir="oulad_outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_table.to_csv(output_dir / "overall_performance.csv", index=False)
    subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    fairness_gaps.to_csv(output_dir / "fairness_gaps.csv", index=False)

    if family_importance is not None and not family_importance.empty:
        family_importance.to_csv(output_dir / "feature_family_importance.csv", index=False)

    print("\nSaved output files to:", output_dir)
    for p in output_dir.glob("*"):
        print(" -", p)
