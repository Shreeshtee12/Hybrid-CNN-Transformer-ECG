"""
run_all_models.py
==================
Calls your EXISTING train_lightning.py and eval.py once per architecture,
with identical arguments every time. This is what turns 5 separate,
manually-run notebooks into one combined, reproducible run.

It does NOT reimplement training -- it just runs your real scripts,
back to back, for every model in MODEL_NAMES, so nothing can drift
between architectures.

Usage (run from the repo root, after `python data_split.py` has been run once):
    python run_all_models.py \
        --signal-path /path/to/records500 \
        --csv-path /path/to/ptbxl_database.csv \
        --max-epochs 120

Requires config.py (for MODEL_NAMES) to be importable from this location.
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

import config


def run_one_model(model_name: str, args) -> str:
    """Trains one architecture, then evaluates its best checkpoint. Returns the eval output dir."""
    print(f"\n{'='*70}\nTRAINING: {model_name}\n{'='*70}")
    train_cmd = [
        sys.executable, "train_lightning.py",
        "--model", model_name,
        "--signal-path", args.signal_path,
        "--csv-path", args.csv_path,
        "--sr", str(args.sr),
        "--seed", str(args.seed),
        "--max-epochs", str(args.max_epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.augment:
        train_cmd.append("--augment")
    subprocess.run(train_cmd, check=True)

    # Find the best val_loss checkpoint for this model (train_lightning.py now
    # saves to a model-specific folder after the checkpoint-collision fix).
    ckpt_dir = str(config.LIGHTNING_LOGS_DIR / "checkpoints" / model_name)
    candidates = sorted(
        f for f in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if "val_loss" in f
    )
    if not candidates:
        raise FileNotFoundError(f"No val_loss checkpoint found for {model_name} in {ckpt_dir}")
    best_ckpt = candidates[-1]
    print(f"[run_all] Using checkpoint: {best_ckpt}")

    print(f"\n{'='*70}\nEVALUATING: {model_name}\n{'='*70}")
    eval_cmd = [
        sys.executable, "eval.py",
        "--model", model_name,
        "--checkpoint", best_ckpt,
        "--signal-path", args.signal_path,
        "--csv-path", args.csv_path,
        "--sr", str(args.sr),
        "--seed", str(args.seed),
        "--threshold", str(args.threshold),
    ]
    subprocess.run(eval_cmd, check=True)

    # eval.py's sanitize_name() replaces any character outside [A-Za-z0-9_.-]
    # with "-" when building its output folder name (so "=" in checkpoint
    # filenames becomes "-"). Replicate that exact transformation here so we
    # look in the same place eval.py actually saved to.
    import re
    ckpt_basename = os.path.splitext(os.path.basename(best_ckpt))[0]
    ckpt_basename = re.sub(r"[^A-Za-z0-9_.-]", "-", ckpt_basename)
    eval_out_dir = str(config.LIGHTNING_LOGS_DIR / "metrics" / ckpt_basename)
    return eval_out_dir


def build_comparison_table(eval_dirs: dict):
    """Merges each model's overall_summary.csv (produced by eval.py) into one table."""
    rows = []
    for model_name, out_dir in eval_dirs.items():
        summary_path = os.path.join(out_dir, "overall_summary.csv")
        if not os.path.exists(summary_path):
            print(f"[run_all] WARNING: missing {summary_path}, skipping {model_name}")
            continue
        row = pd.read_csv(summary_path)
        row.insert(0, "model", model_name)
        rows.append(row)

    if not rows:
        print("[run_all] No results to combine.")
        return

    combined = pd.concat(rows, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "model_comparison.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n\n=== FINAL MODEL COMPARISON -> {out_path} ===")
    print(combined.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--sr", type=int, default=100, choices=[100, 500],
                         help="100Hz matches the notebook that produced the thesis's reported "
                              "results (filename_lr, confidence>=50 labels). Only use 500 if "
                              "intentionally testing the older, different data pipeline.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES,
                         help="Subset of models to run (default: all in config.MODEL_NAMES)")
    args = parser.parse_args()

    eval_dirs = {}
    for model_name in args.models:
        eval_dirs[model_name] = run_one_model(model_name, args)

    build_comparison_table(eval_dirs)


if __name__ == "__main__":
    main()