"""
data_split.py
=============
Generates the ONE patient-grouped train/val split for the project and
saves it to disk. Every training/eval script should read this file
instead of recomputing its own split.

This matches EXACTLY the filtering + split logic already used in
src/train_lightning.py (and now src/eval.py after the leakage fix):
  - load ptbxl_database.csv
  - parse scp_codes, filter to the 8 TARGET_CLASSES
  - drop records with none of the target classes
  - GroupShuffleSplit on patient_id, test_size=0.2, seed=42

NOTE: the real codebase only ever used a single 80/20 train/val split
(no separate held-out test set). This script reproduces that reality
rather than inventing a 3-way split that doesn't match your other files.

Usage (run once, from the repo root, on the machine where the dataset lives):
    python data_split.py
"""

import ast
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

import config


def build_and_save_split():
    csv_path = config.DATASET_PATH / "ptbxl_database.csv"
    print(f"[data_split] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # ---- identical filtering to train_lightning.py / eval.py ----
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    # Confidence >= 50 rule, matching the notebook that actually produced the
    # thesis's reported results.
    df["scp_filtered"] = df["scp_codes"].apply(
        lambda codes: [k for k, conf in codes.items() if k in config.CLASSES and conf >= 50]
    )
    df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=config.CLASSES)
    y = mlb.fit_transform(df["scp_filtered"])
    # filename_lr (100Hz) matches the notebook's actual pipeline, NOT filename_hr (500Hz).
    records = df["filename_lr"].str.replace(".hea", "", regex=False).values

    if "patient_id" not in df.columns:
        raise RuntimeError(
            "patient_id column not found in ptbxl_database.csv — "
            "cannot build a patient-grouped split."
        )

    groups = df["patient_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    train_idx, val_idx = next(gss.split(records, y, groups=groups))

    train_records = records[train_idx].tolist()
    val_records = records[val_idx].tolist()

    train_patients = set(df.iloc[train_idx]["patient_id"].tolist())
    val_patients = set(df.iloc[val_idx]["patient_id"].tolist())

    overlap = train_patients & val_patients
    assert not overlap, (
        f"Patient leakage detected: {len(overlap)} patients appear in both splits."
    )

    split = {
        "seed": config.SEED,
        "classes": config.CLASSES,
        "train_records": train_records,
        "val_records": val_records,
        "num_train_patients": len(train_patients),
        "num_val_patients": len(val_patients),
    }

    config.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    print(f"[data_split] Saved -> {config.SPLIT_FILE}")
    print(f"[data_split] train: {len(train_records)} records, {len(train_patients)} patients")
    print(f"[data_split] val:   {len(val_records)} records, {len(val_patients)} patients")
    print(f"[data_split] Patient overlap: {len(overlap)} (must be 0)")


def load_split():
    """Every training/eval script should call THIS instead of re-splitting."""
    with open(config.SPLIT_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    build_and_save_split()