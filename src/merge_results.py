"""
merge_results.py
=================
Rebuilds results/model_comparison.csv from the overall_summary.csv files
that eval.py already saved under lightning_logs/metrics/{checkpoint_name}/.
Use this any time the combined table is missing/deleted but the individual
per-model results still exist -- no retraining needed.

Usage (run from src/):
    python merge_results.py
"""

import glob
import os

import pandas as pd

import config


def infer_model_name(dir_name: str) -> str:
    """Extracts the model name from a folder like 'resnet1d-epoch-epoch-0-...'"""
    return dir_name.split("-epoch")[0]


def main():
    pattern = str(config.LIGHTNING_LOGS_DIR / "metrics" / "*" / "overall_summary.csv")
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print("No overall_summary.csv files found under lightning_logs/metrics/.")
        return

    rows = []
    for path in summary_files:
        dir_name = os.path.basename(os.path.dirname(path))
        model_name = infer_model_name(dir_name)
        # skip the generic "last" checkpoint folder -- it's a duplicate of
        # whichever model was trained most recently, not a distinct architecture
        if model_name == "last":
            print(f"[merge_results] Skipping {path} (generic 'last' checkpoint, not a named model)")
            continue
        row = pd.read_csv(path)
        row.insert(0, "model", model_name)
        rows.append(row)
        print(f"[merge_results] Loaded {model_name} <- {path}")

    combined = pd.concat(rows, ignore_index=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "model_comparison.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n=== Rebuilt {out_path} ===")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()