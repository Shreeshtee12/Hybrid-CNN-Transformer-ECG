# Filename: src/eval.py
#
# Evaluate a saved checkpoint on PTB-XL.
# - Auto-detects num_classes from the checkpoint head (prefers model.head.4.*).
# - If the checkpoint is 8-class, evaluates on the fixed TARGET_CLASSES.
# - If the checkpoint is larger, projects to TARGET_CLASSES (when present).
# - Outputs a classification report and train micro-accuracy @ configurable threshold.
# - Saves per-class confusion matrices, a grid figure, ROC-per-label, macro ROC,
#   and CSV summaries (confusion stats + per-class AUCs) into a unique subfolder
#   named after the checkpoint.
# - ALSO saves an **overall (micro-aggregated) confusion matrix** and a one-row CSV
#   with micro precision/recall/F1/accuracy + subset accuracy + Jaccard.

import argparse
import glob
import os
import re
import ast
import math
import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
import wfdb
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from typing import Optional

from torchmetrics.functional.classification import multilabel_accuracy as tm_multilabel_accuracy

from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model

import config
import thresholds

# Allowlist optimizer classes that may appear inside Lightning checkpoints when loading with weights_only
add_safe_globals([torch.optim.AdamW])

# Fixed 8-class target set used in your training pipeline
TARGET_CLASSES = ['NORM', 'AFIB', 'PVC', 'LVH', 'IMI', 'ASMI', 'LAFB', 'IRBBB']


# ----------------------------- Dataset -----------------------------

class PTBXL_Dataset(torch.utils.data.Dataset):
    """Returns (T, C) tensors with per-lead z-score normalization."""
    def __init__(self, records, labels, signal_path, sr=500):
        self.records = records
        self.labels = labels
        self.signal_path = signal_path
        self.sr = sr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = self.labels[idx]
        signal, _ = wfdb.rdsamp(f"{self.signal_path}/{rec}")  # (samples, channels)
        seq_len = 5000 if self.sr == 500 else 1000
        signal = signal[:seq_len]  # (T, C)

        # Per-lead z-score (record-wise)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label


# ----------------------------- Utils -----------------------------

def pick_checkpoint(args) -> str:
    """
    Return a checkpoint path: args.checkpoint if given, else the LOWEST
    val_loss checkpoint in args.checkpoint_dir, else last.ckpt.
    This is the ONE place checkpoint selection happens -- run_all_models.py
    and make_figures.py both call this instead of each re-implementing
    their own picking logic, so a standalone `python eval.py` always
    scores the same checkpoint run_all_models.py already picked.
    """
    if args.checkpoint:
        return args.checkpoint
    ckpt_dir = args.checkpoint_dir or str(config.LIGHTNING_LOGS_DIR / "checkpoints")
    files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    scored = []
    for f in files:
        m = re.findall(r"val_loss=(?:val_loss=)?([0-9]+\.[0-9]+)", os.path.basename(f))
        if m:
            scored.append((float(m[-1]), f))
    if scored:
        return min(scored)[1]
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")


def infer_num_classes_from_state(sd: dict) -> Optional[int]:
    """Infer classifier head out_features from Lightning state_dict."""
    # 1) Prefer explicit head/classifier keys used by your models
    preferred_bias_keys = [
        "model.head.4.bias",
        "model.head.bias",
        "model.classifier.bias",
        "model.fc.bias",
    ]
    for k in preferred_bias_keys:
        if k in sd and sd[k].ndim == 1:
            return int(sd[k].shape[0])

    preferred_weight_keys = [
        "model.head.4.weight",
        "model.head.weight",
        "model.classifier.weight",
        "model.fc.weight",
    ]
    for k in preferred_weight_keys:
        if k in sd and sd[k].ndim == 2:
            return int(sd[k].shape[0])

    # 2) Generic fallback: take the *last* Linear-like pair under "model.*"
    candidates = []
    for k, v in sd.items():
        if k.startswith("model.") and k.endswith(".weight") and v.ndim == 2:
            bname = k[:-7] + ".bias"
            if bname in sd and sd[bname].ndim == 1:
                candidates.append((k, bname, int(v.shape[0])))
    if candidates:
        return candidates[-1][2]
    return None


def infer_model_name_from_state(sd: dict, default_model: str) -> str:
    """Best-effort auto-detect model family if args.model == 'auto'."""
    if default_model != "auto":
        return default_model
    has_tx = any(("transformer" in k.lower() or "encoder" in k.lower()) for k in sd.keys())
    has_cnn = any(("cnn." in k.lower() or "conv" in k.lower()) for k in sd.keys())
    if has_tx and has_cnn:
        return "cnn_transformer"
    if has_tx:
        return "transformer"
    return "resnet1d"  # safe fallback


def sanitize_name(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on PTB-XL with confusion matrices and ROC curves.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (cnn_transformer, resnet1d, transformer, xlstm, xresnet1d, or 'auto')")
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--signal-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--sr", type=int, default=500, choices=[100, 500])
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the --seed used in train_lightning.py so the same "
                             "patient-grouped split is reproduced here.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binarizing probabilities.")
    parser.add_argument("--output-dir", type=str, default=str(config.LIGHTNING_LOGS_DIR / "metrics"), help="Base directory for eval artifacts.")
    args = parser.parse_args()

    # seq-len is fully determined by sampling rate -- derive it automatically,
    # matching train_lightning.py, instead of relying on --seq-len being passed
    # correctly by hand every time.
    args.seq_len = 5000 if args.sr == 500 else 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load CSV & labels ----------
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["diagnostic_superclass"] = df["scp_codes"].apply(lambda x: list(x.keys()))
    all_keys = df["diagnostic_superclass"]

    # ---------- Load checkpoint & detect head size ----------
    ckpt_path = pick_checkpoint(args)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state["state_dict"] if "state_dict" in state else state

    ckpt_num_classes = infer_num_classes_from_state(sd)
    print(f"[eval] Detected {ckpt_num_classes} output classes from checkpoint head.")

    # ---------- Label setup ----------
    if ckpt_num_classes == len(TARGET_CLASSES):
        # 8-class evaluation. Confidence >= 50 rule, matching train_lightning.py
        # and the notebook that actually produced the thesis's reported results.
        df["scp_filtered"] = df["scp_codes"].apply(config.filter_codes)
        df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)
        mlb = MultiLabelBinarizer(classes=TARGET_CLASSES)
        y = mlb.fit_transform(df["scp_filtered"])
        use_full_space = False
        target_names = TARGET_CLASSES
        target_idxs = None  # not needed
    else:
        # Full label space (project to TARGET_CLASSES if present).
        # NOTE: this legacy fallback path does not apply the confidence>=50 rule
        # (it's not used by the current 8-class hybrid model, only by older/
        # differently-sized checkpoints). Flagging so it isn't assumed consistent
        # with the 8-class branch above if this path is ever relied on again.
        mlb = MultiLabelBinarizer()
        y_full = mlb.fit_transform(all_keys)
        present_targets = [c for c in TARGET_CLASSES if c in mlb.classes_]
        name_to_idx = {n: i for i, n in enumerate(mlb.classes_)}
        target_idxs = [name_to_idx[c] for c in present_targets]
        y = y_full
        use_full_space = True
        target_names = present_targets
        print(f"[eval] Using full label space with projection to present targets: {target_names}")

    # ---------- Split (loaded from the ONE shared split, matching train_lightning.py) ----------
    # Run `python data_split.py` once before evaluating any checkpoint.
    # filename_lr (100Hz) matches the notebook's actual pipeline, NOT filename_hr (500Hz).
    records = df["filename_lr"].str.replace(".hea", "", regex=False).values

    try:
        from data_split import load_split
        split = load_split()
        train_mask = np.isin(records, split["train_records"])
        val_mask = np.isin(records, split["val_records"])
        test_mask = np.isin(records, split["test_records"])
        train_rec, y_train = records[train_mask], y[train_mask]
        val_rec, y_val = records[val_mask], y[val_mask]
        test_rec, y_test = records[test_mask], y[test_mask]
        print(f"[eval] Loaded shared split ({split['method']}): {len(train_rec)} train / "
              f"{len(val_rec)} val (fold {split['val_fold']}) / {len(test_rec)} test (fold {split['test_fold']}) records")
    except FileNotFoundError:
        raise SystemExit(
            "No shared split found. Run `python data_split.py` once before evaluating "
            "any checkpoint, so evaluation uses the exact same split training used."
        )

    train_ds = PTBXL_Dataset(train_rec, y_train, args.signal_path, sr=args.sr)
    val_ds   = PTBXL_Dataset(val_rec,  y_val,   args.signal_path, sr=args.sr)
    test_ds  = PTBXL_Dataset(test_rec, y_test,  args.signal_path, sr=args.sr)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ---------- Build model ----------
    detected_model = infer_model_name_from_state(sd, args.model)
    num_classes_for_model = ckpt_num_classes if ckpt_num_classes is not None else len(TARGET_CLASSES)
    model = get_model(
        detected_model,
        input_shape=(args.input_channels, args.seq_len),
        num_classes=num_classes_for_model,
    )
    lit_model = LitGenericModel(model)
    # Allow partial load if some keys differ (e.g., optimizer states or buffers)
    missing, unexpected = lit_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[eval] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing keys (truncated):", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("  unexpected keys (truncated):", unexpected[:8], "..." if len(unexpected) > 8 else "")

    lit_model.eval().to(device)

    # ---------- Inference helpers ----------
    @torch.no_grad()
    def infer(loader):
        ys, ps = [], []
        for x, yb in loader:
            x = x.to(device)
            logits = lit_model(x)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.numpy())
        return np.vstack(ps), np.vstack(ys)

    # ---------- Predictions: validation (fit thresholds) and TEST (report) ----------
    def _project(p_all, t_all):
        if not use_full_space and num_classes_for_model == len(TARGET_CLASSES):
            return p_all, t_all
        if target_idxs is None or len(target_idxs) == 0:
            raise ValueError("None of TARGET_CLASSES were present in the dataset when using full label space.")
        return p_all[:, target_idxs], t_all[:, target_idxs]

    yv_pred, yv_true = _project(*infer(val_loader))     # fold 9: used ONLY to fit thresholds
    y_pred, y_true = _project(*infer(test_loader))      # fold 10: everything reported below
    print(f"[eval] Reporting on the held-out TEST fold: {len(y_true)} records")

    # ----- Create unique output directory per checkpoint -----
    base_out = args.output_dir
    ckpt_name = sanitize_name(ckpt_path)
    out_dir = os.path.join(base_out, ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    # Console report @threshold
    thr = float(args.threshold)
    y_pred_bin = (y_pred >= thr).astype(int)
    print("\nClassification Report (TEST fold, fixed threshold):")
    print(classification_report(y_true, y_pred_bin, target_names=target_names,
                                zero_division=0, digits=2))

    # ---------- Train micro-accuracy @threshold ----------
    y_pred_tr, y_true_tr = infer(train_loader)
    if not use_full_space and num_classes_for_model == len(TARGET_CLASSES):
        t_pred, t_true = y_pred_tr, y_true_tr
        n_labels = len(TARGET_CLASSES)
    else:
        t_pred, t_true = y_pred_tr[:, target_idxs], y_true_tr[:, target_idxs]
        n_labels = len(target_names)

    tm_acc = tm_multilabel_accuracy(
        torch.tensor(t_pred, dtype=torch.float32),
        torch.tensor(t_true, dtype=torch.int32),
        num_labels=n_labels,
        threshold=thr,
        average="micro",
    ).item()
    print(f"\nTrain accuracy (micro @ {thr:.2f}): {tm_acc:.3f}")

    # ---------- Confusion Matrices (per-class) ----------
    mcm = multilabel_confusion_matrix(y_true, y_pred_bin)

    # Save a per-class 2x2 confusion matrix image
    for i, cls in enumerate(target_names):
        cm = mcm[i]
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"¬{cls}", cls])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"CM: {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{cls}.png"), dpi=160)
        plt.close(fig)

    # Save an aggregated grid of confusion matrices
    k = len(target_names)
    cols = min(4, k)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.4*rows))
    axes = np.array(axes).reshape(rows, cols)
    for idx, cls in enumerate(target_names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cm = mcm[idx]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"¬{cls}", cls])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(cls)
    # hide empty subplots
    for j in range(k, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')
    plt.suptitle(f"Multilabel Confusion Matrices @ threshold={thr:.2f}")
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(out_dir, "cm_grid.png"), dpi=160)
    plt.close(fig)

    # Save numeric summary as CSV (per-class)
    per_class_rows = []
    for i, cls in enumerate(target_names):
        tn, fp, fn, tp = mcm[i].ravel()
        support_pos = tp + fn
        support_neg = tn + fp
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = (tp / support_pos) if support_pos > 0 else 0.0  # recall
            fpr = (fp / support_neg) if support_neg > 0 else 0.0
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
            acc = (tp + tn) / max(1, tp + tn + fp + fn)
        per_class_rows.append({
            "class": cls,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "support_pos": int(support_pos), "support_neg": int(support_neg),
            "recall": float(tpr), "precision": float(prec), "f1": float(f1), "fpr": float(fpr), "accuracy": float(acc)
        })
    df_cm = pd.DataFrame.from_records(per_class_rows)
    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix_summary.csv"), index=False)

    # ---------- Overall (micro-aggregated) Confusion Matrix & Metrics ----------
    overall_tn = int(mcm[:, 0, 0].sum())
    overall_fp = int(mcm[:, 0, 1].sum())
    overall_fn = int(mcm[:, 1, 0].sum())
    overall_tp = int(mcm[:, 1, 1].sum())

    overall_cm = np.array([[overall_tn, overall_fp], [overall_fn, overall_tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Overall (micro) CM @ {thr:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cm_overall_micro.png"), dpi=160)
    plt.close(fig)

    # overall precision/recall/F1/accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        micro_prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        micro_rec  = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
        micro_acc  = (overall_tp + overall_tn) / max(1, overall_tp + overall_tn + overall_fp + overall_fn)

    # subset accuracy (exact match) & sample-wise Jaccard index
    subset_acc = float((y_true == y_pred_bin).all(axis=1).mean())
    inter = np.logical_and(y_true == 1, y_pred_bin == 1).sum(axis=1).astype(float)
    union = np.logical_or(y_true == 1, y_pred_bin == 1).sum(axis=1).astype(float)
    jaccard_per_sample = np.where(union > 0, inter / union, 1.0)
    jaccard_mean = float(jaccard_per_sample.mean())

    pd.DataFrame([
        {
            "split": "test",
            "macro_auroc": thresholds.macro_auroc(y_true, y_pred),
            "threshold": thr,
            "TN": overall_tn, "FP": overall_fp, "FN": overall_fn, "TP": overall_tp,
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "micro_accuracy": micro_acc,
            "subset_accuracy": subset_acc,
            "jaccard_mean": jaccard_mean,
        }
    ]).to_csv(os.path.join(out_dir, "overall_summary.csv"), index=False)

    # ---------- Three threshold rules: FIT on validation (fold 9), REPORT on test (fold 10) ----------
    rule_rows, thr_rows = [], []
    for rule in thresholds.RULES:
        thr_vec = thresholds.get_thresholds(rule, yv_true, yv_pred)
        per, macro, micro = thresholds.score_at(y_true, y_pred, thr_vec)
        rule_rows.append({
            "rule": rule, "split": "test",
            **{f"macro_{k}": v for k, v in macro.items()},
            **{f"micro_{k}": v for k, v in micro.items()},
        })
        for i, cls in enumerate(target_names):
            thr_rows.append({"rule": rule, "class": cls, "threshold": float(thr_vec[i]),
                             "n_pos_val": int(yv_true[:, i].sum()), "n_pos_test": int(y_true[:, i].sum()),
                             **per[i]})
    pd.DataFrame(rule_rows).to_csv(os.path.join(out_dir, "threshold_rules_test.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(out_dir, "per_class_thresholds_test.csv"), index=False)
    print("\nTEST-fold macro metrics by threshold rule (thresholds fit on validation fold):")
    print(pd.DataFrame(rule_rows)[["rule", "macro_precision", "macro_recall", "macro_f1", "macro_f2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------- ROC Plots ----------
    # ROC per label + per-class AUC table
    auc_rows = []
    plt.figure()
    for k_idx in range(len(target_names)):
        # skip degenerate labels
        if y_true[:, k_idx].sum() in [0, len(y_true)]:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, k_idx], y_pred[:, k_idx])
        roc_auc = auc(fpr, tpr)
        auc_rows.append({"class": target_names[k_idx], "AUC": float(roc_auc)})
        plt.plot(fpr, tpr, label=f"{target_names[k_idx]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC per Label")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_per_label.png"), dpi=160)
    plt.close()

    if auc_rows:
        pd.DataFrame(auc_rows).sort_values("AUC", ascending=False).to_csv(
            os.path.join(out_dir, "per_class_auc.csv"), index=False
        )

    # Macro ROC (simple macro of interpolated TPRs)
    grid = np.linspace(0, 1, 101)
    tprs_interp = []
    for k_idx in range(len(target_names)):
        if y_true[:, k_idx].sum() in [0, len(y_true)]:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, k_idx], y_pred[:, k_idx])
        fpr = np.clip(fpr, 0, 1)
        tpr = np.clip(tpr, 0, 1)
        t_interp = np.interp(grid, fpr, tpr)
        t_interp[0] = 0.0
        tprs_interp.append(t_interp)
    if tprs_interp:
        macro_tpr = np.mean(np.vstack(tprs_interp), axis=0)
        macro_auc = auc(grid, macro_tpr)
        plt.figure()
        plt.plot(grid, macro_tpr, label=f"Macro ROC (AUC={macro_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Macro-average ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_macro.png"), dpi=160)
        plt.close()

    print(f"\n[eval] Saved artifacts to: {out_dir}")
    print(f"[eval] Evaluated checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()