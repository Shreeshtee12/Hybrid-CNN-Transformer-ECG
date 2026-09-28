"""
compare_confusion_matrices.py
==============================
Merges the per-class confusion matrix stats (TN/FP/FN/TP, precision,
recall, f1) that eval.py already saved for each model into ONE comparison
table and one grid figure -- no retraining or re-evaluation needed, this
just reads what's already on disk from lightning_logs/metrics/.

Usage (run from src/, after run_all_models.py or individual eval.py runs):
    python compare_confusion_matrices.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def infer_model_name(dir_name: str) -> str:
    return dir_name.split("-epoch")[0]


def main():
    pattern = str(config.LIGHTNING_LOGS_DIR / "metrics" / "*" / "confusion_matrix_summary.csv")
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print("No confusion_matrix_summary.csv files found. Run eval.py for at least one model first.")
        return

    all_rows = []
    for path in summary_files:
        dir_name = os.path.basename(os.path.dirname(path))
        model_name = infer_model_name(dir_name)
        if model_name == "last":
            continue
        df = pd.read_csv(path)
        df.insert(0, "model", model_name)
        all_rows.append(df)
        print(f"[compare] Loaded {model_name} <- {path}")

    combined = pd.concat(all_rows, ignore_index=True)

    # ---- Save the combined table: one row per (model, class) ----
    out_csv = config.RESULTS_DIR / "confusion_matrix_comparison.csv"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"\n[compare] Saved combined table -> {out_csv}")

    # ---- Pivot: one row per class, one column per model, F1 score ----
    pivot_f1 = combined.pivot(index="class", columns="model", values="f1")
    print("\n=== Per-class F1 comparison across models ===")
    print(pivot_f1.to_string())

    pivot_out = config.RESULTS_DIR / "f1_comparison_by_class.csv"
    pivot_f1.to_csv(pivot_out)
    print(f"[compare] Saved -> {pivot_out}")

    # ---- Grid figure: one confusion-style bar chart per model ----
    models = sorted(combined["model"].unique())
    classes = sorted(combined["class"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub = combined[combined["model"] == model_name].set_index("class").reindex(classes)
        ax.barh(classes, sub["f1"], color="gray")
        ax.set_title(model_name, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 score")

    fig.suptitle("Per-class F1 score comparison across architectures", fontsize=12)
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / "f1_comparison_across_models.png"
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Saved comparison figure -> {fig_path}")


if __name__ == "__main__":
    main()