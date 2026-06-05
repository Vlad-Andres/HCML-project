# ============================================================
# All plotting helpers (§9)
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_bar_plot(df, x, y, hue, title, output_dir, filename, rotation=45):
    plt.figure(figsize=(10, 5))
    groups = df[hue].unique().tolist()
    x_values = df[x].unique().tolist()

    width = 0.8 / max(len(groups), 1)
    positions = np.arange(len(x_values))

    for i, group in enumerate(groups):
        sub = df[df[hue] == group].set_index(x).reindex(x_values)
        plt.bar(positions + i * width, sub[y].values, width=width, label=str(group))

    plt.xticks(positions + width * (len(groups) - 1) / 2, x_values,
               rotation=rotation, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = Path(output_dir) / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", path)


def plot_overall_performance(performance_table, output_dir):
    output_dir = Path(output_dir)
    plt.figure(figsize=(9, 5))
    for model_name, g in performance_table.groupby("model"):
        stage_order = {"early_25pct": 0, "full_100pct": 1}
        g_sorted = g.assign(stage_order=g["stage"].map(stage_order)).sort_values("stage_order")
        plt.plot(g_sorted["stage"], g_sorted["balanced_accuracy"], marker="o", label=model_name)
    plt.title("Overall balanced accuracy: early vs full course")
    plt.ylabel("Balanced accuracy")
    plt.xlabel("Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overall_balanced_accuracy_by_stage.png", dpi=200)
    plt.show()


def plot_fairness_gaps(fairness_gaps, output_dir):
    for model_name in fairness_gaps["model"].unique():
        plot_df = fairness_gaps[fairness_gaps["model"] == model_name].copy()
        save_bar_plot(
            plot_df,
            x="sensitive_attr",
            y="fnr_gap",
            hue="stage",
            title=f"False negative rate gap by sensitive attribute: {model_name}",
            output_dir=output_dir,
            filename=f"fnr_gap_{model_name}.png",
        )
        save_bar_plot(
            plot_df,
            x="sensitive_attr",
            y="selection_rate_pred_at_risk_gap",
            hue="stage",
            title=f"Predicted at-risk selection rate gap: {model_name}",
            output_dir=output_dir,
            filename=f"selection_rate_gap_{model_name}.png",
        )
