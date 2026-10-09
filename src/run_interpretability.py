"""
run_interpretability.py
========================
Runs Grad-CAM + SHAP across ALL 8 target classes for a trained checkpoint,
and computes a quantitative agreement score for each -- this is what turns
"extend interpretability to all classes" into a real experiment instead of
just more pictures (which is what the notebook did, one class at a time,
manually, for only AFIB and LVH).

Also computes TRUE NEGATIVE attention: for each class, does the model's
Grad-CAM attention pattern actually change when the class is absent, or does
it attend to the same regions regardless? A model whose positive and
negative attention look identical isn't really discriminating on that class,
whatever its accuracy says. This generalizes the notebook's one-off
true-negative check (which it only ever did for a couple of classes by hand).

Grad-CAM's hook target is architecture-specific (see get_target_layer below).
It's supported for cnn_transformer, resnet1d, xresnet1d, and cnn_bilstm (the last
with a caveat: it only explains the CNN front-end, not the LSTM's own
temporal reasoning). It does NOT apply to the plain transformer model, which
has no conv layer at all -- SHAP still runs for it, just without a Grad-CAM
comparison or agreement score.

Reuses the exact same checkpoint-loading and dataset code as eval.py, so
results are guaranteed consistent with the rest of the pipeline.

Usage (run from src/, after a model has been trained):
    python run_interpretability.py \
        --model cnn_transformer \
        --checkpoint-dir lightning_logs/checkpoints/cnn_transformer \
        --signal-path /path/to/ptb-xl/1.0.3 \
        --csv-path /path/to/ptb-xl/1.0.3/ptbxl_database.csv
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "interpretability"))

import config
from data_split import load_split
from eval import (
    PTBXL_Dataset, TARGET_CLASSES, pick_checkpoint,
    infer_num_classes_from_state, infer_model_name_from_state,
)
from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model

from gradcam import compute_mean_cam
from shap_utils import compute_shap_for_class
from metrics import gradcam_lead_importance, agreement_score


def get_target_layer(model, model_name):
    """
    Returns the Grad-CAM hook target for a given architecture, or None if
    Grad-CAM doesn't apply to this model's structure.

    Caveats:
    - xresnet1d's index (-4) assumes the default layers=[2,2,2,2] config
      (stem=2 items, head=3 items -- the last ResBlock sits 4 from the end).
      If that config ever changes, this index needs updating too.
    - cnn_bilstm only explains the CNN front-end (conv2), not what the LSTM
      itself attends to over time -- a genuine limitation, not a bug.
    - transformer has no conv layer at all; Grad-CAM as implemented here
      cannot apply. Would need attention-weight visualization instead,
      which is a different technique, not a parameter change.
    """
    if model_name == "cnn_transformer":
        return model.cnn4
    elif model_name == "cnn1d":
        return model.blocks[-1]
    elif model_name == "resnet1d":
        return model.layer4
    elif model_name == "xresnet1d":
        return model[-4]
    elif model_name == "cnn_bilstm":
        return model.conv2
    elif model_name == "transformer":
        return None
    else:
        return None


def collect_class_samples(dataset, class_idx, n_samples, positive=True):
    """Pulls up to n_samples signals where the given class is present (or absent)."""
    signals = []
    for i in range(len(dataset)):
        signal, label = dataset[i]
        val = label[class_idx].item() if torch.is_tensor(label) else label[class_idx]
        if (positive and val == 1) or (not positive and val == 0):
            signals.append(signal.numpy() if torch.is_tensor(signal) else signal)
        if len(signals) >= n_samples:
            break
    return np.array(signals)


def to_chw(signals, seq_len):
    """Converts (N, T, C) -> (N, C, T) if needed. PTBXL_Dataset returns (T, C)."""
    return signals.transpose(0, 2, 1) if signals.shape[1] == seq_len else signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--signal-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--sr", type=int, default=100, choices=[100, 500])
    parser.add_argument("--n-gradcam-samples", type=int, default=50,
                         help="Samples per class for the Grad-CAM average (notebook used 50 for AFIB).")
    parser.add_argument("--n-negative-samples", type=int, default=50,
                         help="Negative (class-absent) samples per class for the true-negative check.")
    parser.add_argument("--n-shap-background", type=int, default=5)
    parser.add_argument("--n-shap-explain", type=int, default=2,
                         help="SHAP KernelExplainer is slow -- notebook only explained 2 samples per class.")
    parser.add_argument("--shap-nsamples", type=int, default=50)
    parser.add_argument("--skip-shap", action="store_true",
                         help="Skip SHAP entirely (useful for a quick Grad-CAM-only + true-negative pass).")
    args = parser.parse_args()

    args.seq_len = 5000 if args.sr == 500 else 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load data (identical to eval.py) ----------
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["scp_filtered"] = df["scp_codes"].apply(config.filter_codes)
    df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=TARGET_CLASSES)
    y = mlb.fit_transform(df["scp_filtered"])
    records = df["filename_lr"].str.replace(".hea", "", regex=False).values

    split = load_split()
    # Interpretability is computed on the held-out TEST fold (10).
    test_mask = np.isin(records, split["test_records"])
    test_rec, y_test = records[test_mask], y[test_mask]
    val_dataset = PTBXL_Dataset(test_rec, y_test, args.signal_path, sr=args.sr)
    print(f"[interp] Loaded {len(val_dataset)} TEST records (strat_fold 10)")

    # ---------- Load checkpoint (identical approach to eval.py) ----------
    ckpt_path = pick_checkpoint(args)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state["state_dict"] if "state_dict" in state else state
    ckpt_num_classes = infer_num_classes_from_state(sd)
    detected_model = infer_model_name_from_state(sd, args.model)

    model_obj = get_model(detected_model, input_shape=(args.input_channels, args.seq_len),
                           num_classes=ckpt_num_classes)
    lit_model = LitGenericModel(model_obj)
    lit_model.load_state_dict(sd, strict=False)
    lit_model.eval().to(device)
    model = lit_model.model  # the underlying nn.Module

    print(f"[interp] Loaded checkpoint: {ckpt_path}")

    target_layer = get_target_layer(model, detected_model)
    gradcam_supported = target_layer is not None
    if not gradcam_supported:
        print(f"[interp] NOTE: Grad-CAM is not supported for '{detected_model}' "
              f"(no conv layer to hook). Running SHAP only, no agreement score.")

    # ---------- Run for every class ----------
    rows = []
    for class_idx, class_name in enumerate(TARGET_CLASSES):
        print(f"\n[interp] === {class_name} ===")
        pos_signals = collect_class_samples(val_dataset, class_idx, args.n_gradcam_samples, positive=True)
        if len(pos_signals) == 0:
            print(f"[interp] No positive test samples for {class_name}, skipping.")
            continue
        print(f"[interp] {len(pos_signals)} positive samples found")
        pos_chw = to_chw(pos_signals, args.seq_len)

        row = {"class": class_name, "n_positive_samples": len(pos_signals), "model": detected_model}

        # ---- Grad-CAM: positive attention, and true-negative comparison ----
        if gradcam_supported:
            mean_cam_pos, _ = compute_mean_cam(
                model, pos_chw, class_idx, device, signal_length=args.seq_len, target_layer=target_layer
            )

            neg_signals = collect_class_samples(val_dataset, class_idx, args.n_negative_samples, positive=False)
            if len(neg_signals) > 0:
                neg_chw = to_chw(neg_signals, args.seq_len)
                mean_cam_neg, _ = compute_mean_cam(
                    model, neg_chw, class_idx, device, signal_length=args.seq_len, target_layer=target_layer
                )
                # How similar is the model's attention when the class is present
                # vs absent? Low/negative correlation = attention genuinely shifts
                # for this class. High correlation = the model attends to the same
                # regions regardless -- a sign this class isn't really driving the
                # model's own internal attention, whatever its accuracy looks like.
                pos_neg_corr, _ = pearsonr(mean_cam_pos, mean_cam_neg)
                print(f"[interp] Positive vs negative attention correlation: {pos_neg_corr:.3f} "
                      f"(lower = more discriminative)")
                row["n_negative_samples"] = len(neg_signals)
                row["gradcam_pos_vs_neg_attention_corr"] = float(pos_neg_corr)
            else:
                print(f"[interp] No negative test samples for {class_name} (unusual).")
                row["n_negative_samples"] = 0
                row["gradcam_pos_vs_neg_attention_corr"] = None

        # ---- SHAP + agreement (only if Grad-CAM is supported, for the bridge metric) ----
        if not args.skip_shap:
            shap_signals = pos_chw[: max(args.n_shap_background, args.n_shap_explain)]
            lead_importance, _ = compute_shap_for_class(
                model, device, shap_signals, class_idx,
                n_background=args.n_shap_background, n_explain=args.n_shap_explain,
                nsamples=args.shap_nsamples, signal_length=args.seq_len,
            )
            shap_leads_mean = lead_importance.mean(axis=0)
            row["n_shap_explained"] = args.n_shap_explain

            if gradcam_supported:
                gradcam_leads = np.array([
                    gradcam_lead_importance(mean_cam_pos, pos_chw[i])
                    for i in range(min(len(pos_chw), args.n_shap_explain))
                ]).mean(axis=0)
                corr, p_value = agreement_score(gradcam_leads, shap_leads_mean)
                print(f"[interp] SHAP-GradCAM lead-agreement (Spearman): {corr:.3f} (p={p_value:.3f})")
                row["gradcam_shap_agreement_spearman"] = corr
                row["agreement_p_value"] = p_value

        rows.append(row)

    # ---------- Save the one comparison table across all classes ----------
    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"interpretability_scores_{detected_model}.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"\n[interp] Saved -> {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()