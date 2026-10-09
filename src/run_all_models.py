"""
run_all_models.py
==================
Calls your EXISTING train_lightning.py and eval.py once per architecture,
with identical arguments every time. Optionally also calls make_figures.py
after each model's eval.
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

import config


def run_one_model(model_name: str, args) -> str:
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
    if args.wandb:
        train_cmd += ["--wandb", "--project", args.project,
                      "--run-name", f"{model_name}-seed{args.seed}-ep{args.max_epochs}"]
        if args.entity:
            train_cmd += ["--entity", args.entity]
        if args.log_artifact:
            train_cmd.append("--log-artifact")
    subprocess.run(train_cmd, check=True)

    ckpt_dir = str(config.LIGHTNING_LOGS_DIR / "checkpoints" / model_name)
    from eval import pick_checkpoint
    class _A: checkpoint = None; checkpoint_dir = ckpt_dir
    best_ckpt = pick_checkpoint(_A())
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

    if args.figures:
        print(f"\n{'='*70}\nFIGURES: {model_name}\n{'='*70}")
        fig_cmd = [
            sys.executable, "make_figures.py",
            "--model", model_name,
            "--checkpoint", best_ckpt,
            "--signal-path", args.signal_path,
            "--csv-path", args.csv_path,
            "--sr", str(args.sr),
        ]
        if args.figure_kinds:
            fig_cmd += ["--classes"] + args.figure_kinds
        rc = subprocess.run(fig_cmd).returncode
        if rc != 0:
            print(f"[run_all] WARNING: make_figures.py failed for {model_name} (exit {rc}). "
                  f"Training and evaluation are unaffected; re-run make_figures.py by hand.")

    import re
    ckpt_basename = os.path.splitext(os.path.basename(best_ckpt))[0]
    ckpt_basename = re.sub(r"[^A-Za-z0-9_.-]", "-", ckpt_basename)
    eval_out_dir = str(config.LIGHTNING_LOGS_DIR / "metrics" / ckpt_basename)
    return eval_out_dir


def build_comparison_table(eval_dirs: dict):
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

    # Second table: TEST-fold metrics under the three threshold rules (fit on validation)
    rule_rows = []
    for model_name, out_dir in eval_dirs.items():
        rp = os.path.join(out_dir, "threshold_rules_test.csv")
        if os.path.exists(rp):
            r = pd.read_csv(rp)
            r.insert(0, "model", model_name)
            rule_rows.append(r)
    if rule_rows:
        pd.concat(rule_rows, ignore_index=True).to_csv(
            os.path.join("results", "model_comparison_threshold_rules.csv"), index=False)
    out_path = os.path.join("results", "model_comparison.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n\n=== FINAL MODEL COMPARISON -> {out_path} ===")
    print(combined.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--sr", type=int, default=100, choices=[100, 500])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--figure-kinds", nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="ptbxl-ecg")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--log-artifact", action="store_true")
    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES)
    args = parser.parse_args()

    eval_dirs = {}
    for model_name in args.models:
        eval_dirs[model_name] = run_one_model(model_name, args)

    build_comparison_table(eval_dirs)


if __name__ == "__main__":
    main()