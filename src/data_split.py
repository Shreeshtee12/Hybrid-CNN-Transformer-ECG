"""
data_split.py
=============
ONE shared train/val/test split for the whole project, using PTB-XL's OFFICIAL
`strat_fold` column (Wagner et al. 2020; same protocol as Strodthoff et al. 2021):

    folds 1-8 -> train      fold 9 -> validation      fold 10 -> test

strat_fold is patient-grouped (no patient appears in two folds), label-stratified,
and fixed in the dataset itself -- there is no random seed involved, so the split
is identical for everyone and directly comparable with published PTB-XL results.

Usage (run once, from src/, on the machine where the dataset lives):
    python data_split.py
"""

import ast
import json

import numpy as np
import pandas as pd

import config

TRAIN_FOLDS = list(range(1, 9))   # 1..8
VAL_FOLD = 9
TEST_FOLD = 10


def build_and_save_split():
    csv_path = config.DATASET_PATH / "ptbxl_database.csv"
    print(f"[data_split] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ("strat_fold", "patient_id", "filename_lr"):
        if col not in df.columns:
            raise RuntimeError(f"'{col}' column not found in ptbxl_database.csv")

    # ---- same label filtering as train_lightning.py / eval.py (confidence >= 50) ----
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["scp_filtered"] = df["scp_codes"].apply(config.filter_codes)
    n_all = len(df)
    df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)
    print(f"[data_split] {len(df)} / {n_all} records have >=1 of the 8 target classes (rhythm classes any conf, diagnostic >= 50)")

    records = df["filename_lr"].str.replace(".hea", "", regex=False).values
    patients = df["patient_id"].values
    folds = df["strat_fold"].values

    masks = {
        "train": np.isin(folds, TRAIN_FOLDS),
        "val": folds == VAL_FOLD,
        "test": folds == TEST_FOLD,
    }

    # ---- sanity checks ----
    pats = {k: set(patients[m].tolist()) for k, m in masks.items()}
    overlaps = {
        "train&val": len(pats["train"] & pats["val"]),
        "train&test": len(pats["train"] & pats["test"]),
        "val&test": len(pats["val"] & pats["test"]),
    }
    assert all(v == 0 for v in overlaps.values()), f"Patient leakage across splits: {overlaps}"

    # ---- per-class positive counts per split (the AFIB diagnostic) ----
    print("\n[data_split] Positives per class (records):")
    print(f"  {'class':<7}{'train':>8}{'val':>8}{'test':>8}")
    for c in config.CLASSES:
        has = df["scp_filtered"].apply(lambda L: c in L).values
        print(f"  {c:<7}{int(has[masks['train']].sum()):>8}"
              f"{int(has[masks['val']].sum()):>8}{int(has[masks['test']].sum()):>8}")

    split = {
        "method": "strat_fold",
        "train_folds": TRAIN_FOLDS,
        "val_fold": VAL_FOLD,
        "test_fold": TEST_FOLD,
        "classes": config.CLASSES,
        "train_records": records[masks["train"]].tolist(),
        "val_records": records[masks["val"]].tolist(),
        "test_records": records[masks["test"]].tolist(),
        "num_train_patients": len(pats["train"]),
        "num_val_patients": len(pats["val"]),
        "num_test_patients": len(pats["test"]),
    }

    config.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\n[data_split] Saved -> {config.SPLIT_FILE}")
    for k in ("train", "val", "test"):
        print(f"[data_split] {k:<5}: {int(masks[k].sum())} records, {len(pats[k])} patients")
    print(f"[data_split] Patient overlap: {overlaps} (all must be 0)")


def load_split():
    """Every training/eval script calls THIS instead of re-splitting."""
    with open(config.SPLIT_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    build_and_save_split()