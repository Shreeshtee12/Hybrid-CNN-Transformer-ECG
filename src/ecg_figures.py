"""
ecg_figures.py
===============
Plotting code for every figure type in CNN_Transformer_Hybrid.ipynb, moved
into reusable functions so the SAME figures can be produced for every model
and every class (the notebook only made them for the hybrid model, and the
interpretability ones only for AFIB and LVH).

Styling (colors, fonts, layouts, panel structure) is copied from the notebook
cells. Deliberate differences from the notebook, all listed here so nothing
is silently changed:

  1. Class / model names are parameters, not hardcoded to AFIB / "Hybrid".
  2. The "Clinical interpretation" text boxes in the SHAP and attention
     figures are replaced by a caption computed from the data. The notebook
     wrote fixed sentences (e.g. "Lead V1 shows the highest SHAP
     contribution") regardless of what the values were; a fixed sentence
     cannot be correct for every class and every model.
  3. The SHAP heatmap (panel C) shows mean |SHAP|. The notebook passed a
     signed mean while labelling the colorbar "|SHAP|".
  4. The negative-example figure uses the tuned per-class thresholds for
     every class. The notebook hardcoded AFIB=0.50 and NORM=0.40 and skipped
     AFIB entirely.
  5. Macro-ROC comparison AUCs are printed to 3 decimals (notebook: 2) so
     close models are distinguishable.

No torch import here on purpose: this file is pure numpy/matplotlib/scipy/
sklearn, so the layout code can be tested without a GPU.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from sklearn.metrics import (
    auc, confusion_matrix, precision_score, recall_score, roc_curve,
)

ACCENT = "#C0392B"
ECGCOLOR = "#1A1A2E"
GRIDCOL = "#E8E8E8"
BEST_COL = "#1A6B3C"
TN_COL = "#2E4057"
BW_GRIDCOL = "#F0F0F0"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

MODEL_DISPLAY = {
    "cnn_transformer": "Hybrid CNN-Transformer",
    "resnet1d": "ResNet1D",
    "xresnet1d": "xResNet1D",
    "transformer": "Transformer",
    "cnn1d": "1D CNN",
    "cnn_bilstm": "CNN-BiLSTM",
}


def model_label(model_name):
    return MODEL_DISPLAY.get(model_name, model_name)


def _minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (np.ptp(x) + 1e-8)


def _resample(x, n):
    x = np.asarray(x, dtype=float)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def lead_importance_from_cam(cam, signal):
    """Notebook's per-lead score: CAM-weighted mean |amplitude|, min-max normalised.
    signal: (12, T). cam: (T,)."""
    imp = np.array([
        np.sum(cam * np.abs(signal[l])) / (np.sum(np.abs(signal[l])) + 1e-8)
        for l in range(signal.shape[0])
    ])
    return _minmax(imp)


# =====================================================================
# Thresholds (notebook cell 12)
# =====================================================================
def optimal_f2_thresholds(y_true, y_pred, beta=2, grid=None):
    """Per-class threshold maximising F-beta (default F2), grid 0.05..0.89 step 0.01,
    exactly as the notebook. Returns list of floats."""
    if grid is None:
        grid = np.arange(0.05, 0.90, 0.01)
    out = []
    for i in range(y_true.shape[1]):
        best_t, best_s = 0.5, 0.0
        for th in grid:
            yb = (y_pred[:, i] >= th).astype(int)
            r = recall_score(y_true[:, i], yb, zero_division=0)
            p = precision_score(y_true[:, i], yb, zero_division=0)
            s = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if (beta ** 2 * p + r) > 0 else 0.0
            if s > best_s:
                best_s, best_t = s, th
        out.append(float(best_t))
    return out


# =====================================================================
# ROC (notebook cell 10)
# =====================================================================
def compute_macro_roc(y_true, y_pred):
    """Notebook's macro-ROC (interpolated mean TPR). Classes with no positives are skipped."""
    valid = [i for i in range(y_true.shape[1]) if 0 < y_true[:, i].sum() < len(y_true)]
    fpr, tpr = {}, {}
    for i in valid:
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def plot_roc(y_true, y_pred, class_names, model_name, save_path):
    style = {
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "black", "axes.grid": True, "grid.color": "lightgray",
        "grid.alpha": 0.3, "grid.linestyle": "--", "font.size": 12,
    }
    assert y_true.shape == y_pred.shape, "Mismatch between labels and predictions!"
    valid = [i for i in range(len(class_names)) if 0 < y_true[:, i].sum() < len(y_true)]
    with plt.rc_context(style):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        for i in valid:
            fpr_i, tpr_i, _ = roc_curve(y_true[:, i], y_pred[:, i])
            plt.plot(fpr_i, tpr_i, lw=2, label=f"{class_names[i]} (AUC = {auc(fpr_i, tpr_i):.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Per-Class ROC Curves ({model_label(model_name)})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        all_fpr, mean_tpr, macro_auc = compute_macro_roc(y_true, y_pred)
        plt.plot(all_fpr, mean_tpr, lw=3, label=f"{model_label(model_name)} (Macro-AUC = {macro_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Macro-Average ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return macro_auc


def plot_macro_roc_comparison(preds, save_path):
    """preds: dict model_name -> (y_true, y_pred). Notebook MacroROC_Comparison cell 4."""
    plt.figure(figsize=(8, 6))
    aucs = {}
    for name, (yt, yp) in preds.items():
        fpr, tpr, a = compute_macro_roc(yt, yp)
        aucs[name] = a
        plt.plot(fpr, tpr, label=f"{model_label(name)} (AUC = {a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Average ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return aucs


# =====================================================================
# Confusion matrices (notebook cell 12)
# =====================================================================
def plot_confusion_grid(y_true, y_pred, thresholds, class_names, model_name, save_path):
    n = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        y_true_bin = y_true[:, i].astype(int)
        y_pred_bin = (y_pred[:, i] >= thresholds[i]).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        total = cm.sum()
        cm_prop = cm / total
        im = ax.imshow(cm_prop, cmap="OrRd", vmin=0, vmax=1)

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        for r in range(2):
            for c in range(2):
                count = cm[r, c]
                percent = cm_prop[r, c] * 100
                color = "white" if cm_prop[r, c] > 0.5 else "black"
                ax.text(c, r, f"{percent:.2f}%\n(n={count})", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted label", fontsize=8)
        ax.set_ylabel("True label", fontsize=8)
        ax.set_title(f"{class_names[i]} (t={thresholds[i]:.2f})", fontsize=11, fontweight="bold")

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        if sens >= 0.85:
            sens_color = "green"
        elif sens >= 0.70:
            sens_color = "darkorange"
        else:
            sens_color = "red"
        ax.text(0.5, -0.35, f"Sens: {sens:.3f} | Spec: {spec:.3f}",
                transform=ax.transAxes, ha="center", fontsize=8, color=sens_color,
                bbox=dict(facecolor="white", edgecolor=sens_color, boxstyle="round,pad=0.3"))
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Confusion Matrices \u2014 {model_label(model_name)}  (threshold optimised for F2-score)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Grad-CAM (notebook cell 20)
# =====================================================================
_GC_RC = {
    "font.family": "DejaVu Serif", "font.size": 12, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
}


def plot_gradcam_panel_a(signal, mean_cam, class_name, save_path, fs=100):
    """Panel A: lead-I ECG with mean Grad-CAM underlay."""
    lead = np.asarray(signal[0], dtype=float)
    n = len(lead)
    t = np.arange(n) / fs
    cam_norm = _minmax(_resample(mean_cam, n))
    with plt.rc_context(_GC_RC):
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
        ax.plot(t, lead, color=ECGCOLOR, linewidth=1.0, zorder=3, alpha=0.9)
        ax.fill_between(t, lead.min() - 0.3, lead.min() - 0.3 + cam_norm * 0.5,
                        alpha=0.35, color=ACCENT, zorder=2, label="Grad-CAM activation")
        peaks, _ = find_peaks(cam_norm, height=0.4, distance=int(0.5 * fs))
        for pk in peaks[:3]:
            ax.axvline(t[pk], color=ACCENT, linewidth=1.0, alpha=0.6, linestyle="--", zorder=4)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(lead.min() - 0.5, lead.max() + 0.3)
        ax.set_xlabel("Time (s)", fontsize=14, labelpad=8)
        ax.set_ylabel("Amplitude (mV)", fontsize=14, labelpad=8)
        ax.tick_params(labelsize=13)
        ax.set_title(f"Grad-CAM Explanation \u2014 {class_name} (Lead I)",
                     fontsize=14, fontweight="bold", loc="left", pad=10)
        ax.legend(fontsize=12, frameon=False, loc="upper right")
        ax.grid(color=GRIDCOL, linewidth=0.5)
        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_gradcam_panels_bc(all_cams, signal, mean_cam, class_name, save_path, fs=100):
    """Panel B: per-lead importance. Panel C: cross-sample activation heatmap."""
    n_lead_samples = signal.shape[1]
    lead_imp = lead_importance_from_cam(_resample(mean_cam, n_lead_samples), signal)
    order = np.argsort(lead_imp)
    n_samples = len(all_cams)
    cam_rows = np.array([
        [seg.mean() for seg in np.array_split(_resample(c, n_lead_samples), 50)] for c in all_cams
    ])
    with plt.rc_context(_GC_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white",
                                 gridspec_kw={"width_ratios": [2.0, 2.0], "wspace": 0.20})
        ax_bar = axes[0]
        colors_bar = [ACCENT if lead_imp[i] > 0.6 else "#AAAAAA" for i in order]
        bars = ax_bar.barh(range(12), lead_imp[order], color=colors_bar, height=0.55, edgecolor="none")
        for bar, val in zip(bars, lead_imp[order]):
            if val > 0.08:
                ax_bar.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                            va="center", fontsize=10, color="#444")
        ax_bar.set_yticks(range(12))
        ax_bar.set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=11)
        ax_bar.set_xlabel("Normalised Grad-CAM score", fontsize=11, labelpad=6)
        ax_bar.tick_params(labelsize=11, left=False)
        ax_bar.set_title(f"Per-lead Grad-CAM importance \u2014 {class_name}",
                         fontsize=11, fontweight="bold", loc="left", pad=8)
        ax_bar.grid(axis="x", color=GRIDCOL, linewidth=0.5)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.set_xlim(0, 1.45)
        ax_bar.text(1.14, 11, f"\u2190 {LEAD_NAMES[order[-1]]}", fontsize=10,
                    color=ACCENT, va="center", fontweight="bold")

        ax_heat = axes[1]
        im = ax_heat.imshow(cam_rows, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax_heat.set_yticks(range(0, n_samples, 5))
        ax_heat.set_yticklabels([f"S{i + 1}" for i in range(0, n_samples, 5)], fontsize=10)
        ax_heat.set_xlabel("Time segment", fontsize=11, labelpad=6)
        ax_heat.tick_params(labelsize=10)
        ax_heat.set_title(f"Cross-sample activation \u2014 {class_name}",
                          fontsize=11, fontweight="bold", loc="left", pad=8)
        cb = plt.colorbar(im, ax=ax_heat, fraction=0.06, pad=0.05)
        cb.set_label("Activation", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return lead_imp


def plot_gradcam_combined(all_cams, signal, mean_cam, class_name, save_path, fs=100):
    """
    ONE figure instead of two files: panel A (ECG + Grad-CAM overlay) on top,
    spanning the full width, with panel B (per-lead importance) and panel C
    (cross-sample heatmap) side by side underneath. Same drawing code as
    plot_gradcam_panel_a / plot_gradcam_panels_bc, just laid out on one canvas
    -- easier to scan when you have 5 models x 8 classes to look through.
    Returns lead_imp, same as plot_gradcam_panels_bc.
    """
    lead = np.asarray(signal[0], dtype=float)
    n = len(lead)
    t = np.arange(n) / fs
    cam_norm = _minmax(_resample(mean_cam, n))

    n_lead_samples = signal.shape[1]
    lead_imp = lead_importance_from_cam(_resample(mean_cam, n_lead_samples), signal)
    order = np.argsort(lead_imp)
    n_samples = len(all_cams)
    cam_rows = np.array([
        [seg.mean() for seg in np.array_split(_resample(c, n_lead_samples), 50)] for c in all_cams
    ])

    with plt.rc_context(_GC_RC):
        fig = plt.figure(figsize=(14, 9.5), facecolor="white")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.4, wspace=0.20)

        # ---- Panel A: ECG + Grad-CAM overlay, full width ----
        ax_ecg = fig.add_subplot(gs[0, :])
        ax_ecg.plot(t, lead, color=ECGCOLOR, linewidth=1.0, zorder=3, alpha=0.9)
        ax_ecg.fill_between(t, lead.min() - 0.3, lead.min() - 0.3 + cam_norm * 0.5,
                            alpha=0.35, color=ACCENT, zorder=2, label="Grad-CAM activation")
        peaks, _ = find_peaks(cam_norm, height=0.4, distance=int(0.5 * fs))
        for pk in peaks[:3]:
            ax_ecg.axvline(t[pk], color=ACCENT, linewidth=1.0, alpha=0.6, linestyle="--", zorder=4)
        ax_ecg.set_xlim(t[0], t[-1])
        ax_ecg.set_ylim(lead.min() - 0.5, lead.max() + 0.3)
        ax_ecg.set_xlabel("Time (s)", fontsize=13, labelpad=6)
        ax_ecg.set_ylabel("Amplitude (mV)", fontsize=13, labelpad=6)
        ax_ecg.tick_params(labelsize=12)
        ax_ecg.set_title(f"A   Grad-CAM Explanation \u2014 {class_name} (Lead I)",
                         fontsize=13, fontweight="bold", loc="left", pad=8)
        ax_ecg.legend(fontsize=11, frameon=False, loc="upper right")
        ax_ecg.grid(color=GRIDCOL, linewidth=0.5)

        # ---- Panel B: per-lead importance ----
        ax_bar = fig.add_subplot(gs[1, 0])
        colors_bar = [ACCENT if lead_imp[i] > 0.6 else "#AAAAAA" for i in order]
        bars = ax_bar.barh(range(12), lead_imp[order], color=colors_bar, height=0.55, edgecolor="none")
        for bar, val in zip(bars, lead_imp[order]):
            if val > 0.08:
                ax_bar.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                            va="center", fontsize=9, color="#444")
        ax_bar.set_yticks(range(12))
        ax_bar.set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=10)
        ax_bar.set_xlabel("Normalised Grad-CAM score", fontsize=10, labelpad=5)
        ax_bar.tick_params(labelsize=10, left=False)
        ax_bar.set_title(f"B   Per-lead importance \u2014 {class_name}",
                         fontsize=11, fontweight="bold", loc="left", pad=6)
        ax_bar.grid(axis="x", color=GRIDCOL, linewidth=0.5)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.set_xlim(0, 1.45)
        ax_bar.text(1.14, 11, f"\u2190 {LEAD_NAMES[order[-1]]}", fontsize=9,
                    color=ACCENT, va="center", fontweight="bold")

        # ---- Panel C: cross-sample activation heatmap ----
        ax_heat = fig.add_subplot(gs[1, 1])
        im = ax_heat.imshow(cam_rows, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        step = max(1, n_samples // 10)
        ax_heat.set_yticks(range(0, n_samples, step))
        ax_heat.set_yticklabels([f"S{i + 1}" for i in range(0, n_samples, step)], fontsize=9)
        ax_heat.set_xlabel("Time segment", fontsize=10, labelpad=5)
        ax_heat.tick_params(labelsize=9)
        ax_heat.set_title(f"C   Cross-sample activation \u2014 {class_name}",
                          fontsize=11, fontweight="bold", loc="left", pad=6)
        cb = plt.colorbar(im, ax=ax_heat, fraction=0.06, pad=0.05)
        cb.set_label("Activation", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        fig.suptitle(f"Grad-CAM Interpretability \u2014 {class_name}", fontsize=14,
                    fontweight="bold", y=0.98, color=ECGCOLOR)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return lead_imp


# =====================================================================
# SHAP (notebook cell 36)
# =====================================================================
_SANS_RC = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
}


def plot_shap_figure(class_name, signal, shap3d, model_name, save_path, fs=100, nsamples=None):
    """shap3d: (n_explained, 12, n_bins) raw SHAP values for this class."""
    n_explained, _, n_bins = shap3d.shape
    lead = np.asarray(signal[0], dtype=float)
    n = len(lead)
    t = np.arange(n) / fs

    abs_shap = np.abs(shap3d)
    mean_lead_imp = abs_shap.mean(axis=(0, 2))
    mean_lead_imp_n = _minmax(mean_lead_imp)
    order = np.argsort(mean_lead_imp_n)
    mean_abs_time = abs_shap.mean(axis=0)                      # (12, n_bins)
    shap_up = _resample(_minmax(abs_shap.mean(axis=(0, 1))), n)
    top_lead = LEAD_NAMES[order[-1]]

    with plt.rc_context(_SANS_RC):
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1.5, 0.8], width_ratios=[2.5, 1],
                               hspace=0.45, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.06)

        ax_ecg = fig.add_subplot(gs[0, :])
        ax_ecg.plot(t, lead, color=ECGCOLOR, linewidth=0.9, zorder=3, alpha=0.9)
        ax_ecg.fill_between(t, lead.min() - 0.3, lead.min() - 0.3 + shap_up * 0.5,
                            alpha=0.35, color=ACCENT, zorder=2, label="SHAP magnitude")
        peaks, _ = find_peaks(shap_up, height=0.4, distance=int(0.5 * fs))
        for pk in peaks[:3]:
            ax_ecg.axvline(t[pk], color=ACCENT, linewidth=0.8, alpha=0.6, linestyle="--", zorder=4)
        ax_ecg.set_xlim(t[0], t[-1])
        ax_ecg.set_ylim(lead.min() - 0.5, lead.max() + 0.3)
        ax_ecg.set_xlabel("Time (s)", fontsize=10)
        ax_ecg.set_ylabel("Amplitude (mV)", fontsize=10)
        ax_ecg.set_title(f"A    SHAP explanation \u2014 {class_name} (Lead I)",
                         fontsize=11, fontweight="bold", loc="left", pad=8)
        ax_ecg.legend(fontsize=9, frameon=False, loc="upper right")
        ax_ecg.grid(color=GRIDCOL, linewidth=0.5)
        ax_ecg.tick_params(labelsize=9)

        ax_bar = fig.add_subplot(gs[1, 0])
        colors_bar = [ACCENT if mean_lead_imp_n[i] > 0.6 else "#AAAAAA" for i in order]
        ax_bar.barh(range(12), mean_lead_imp_n[order], color=colors_bar, height=0.65, edgecolor="none")
        ax_bar.set_yticks(range(12))
        ax_bar.set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=9)
        ax_bar.set_xlabel("Normalised |SHAP|", fontsize=9)
        ax_bar.set_title("B    Lead importance", fontsize=11, fontweight="bold", loc="left")
        ax_bar.grid(axis="x", color=GRIDCOL, linewidth=0.5)
        ax_bar.tick_params(labelsize=9)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.tick_params(left=False)
        ax_bar.text(mean_lead_imp_n[order[-1]] + 0.01, 11, f"\u2190 {top_lead}",
                    fontsize=8, color=ACCENT, va="center")

        ax_heat = fig.add_subplot(gs[1, 1])
        im = ax_heat.imshow(mean_abs_time, aspect="auto", cmap="Reds", vmin=0,
                            vmax=mean_abs_time.max(), interpolation="nearest")
        ax_heat.set_yticks(range(12))
        ax_heat.set_yticklabels(LEAD_NAMES, fontsize=7)
        ax_heat.set_xlabel("Time segment", fontsize=9)
        ax_heat.set_title("C    SHAP heatmap  lead \u00d7 time", fontsize=10, fontweight="bold", loc="left")
        plt.colorbar(im, ax=ax_heat, fraction=0.06, pad=0.04, label="|SHAP|")

        ax_note = fig.add_subplot(gs[2, :])
        ax_note.axis("off")
        ns = f", KernelExplainer nsamples={nsamples}" if nsamples else ""
        ax_note.text(0.0, 0.85, "Computed summary:", fontsize=10, fontweight="bold",
                     transform=ax_note.transAxes, color=ECGCOLOR)
        ax_note.text(0.0, 0.15,
                     f"Top-ranked lead by mean |SHAP|: {top_lead}.  Averaged over {n_explained} "
                     f"explained {class_name}-positive recordings{ns}.",
                     fontsize=9, transform=ax_note.transAxes, color="#444444", style="italic")
        fig.suptitle(f"SHAP Interpretability \u2014 {class_name} | {model_label(model_name)}",
                     fontsize=13, fontweight="bold", y=0.97, color=ECGCOLOR)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return mean_lead_imp_n


# =====================================================================
# Transformer attention (notebook cell 22) -- models with transformer blocks only
# =====================================================================
_ATTN_RC = dict(_SANS_RC, **{"xtick.major.size": 3, "ytick.major.size": 3})


def plot_attention_figure(class_name, signal, attn_stack, model_name, save_path, fs=100):
    """attn_stack: (n_layers, n_heads, T', T')."""
    lead = np.asarray(signal[0], dtype=float)
    n = len(lead)
    t = np.arange(n) / fs
    n_layers = attn_stack.shape[0]

    mean_attn = attn_stack.mean(axis=(0, 1))
    attn_ts = mean_attn.mean(axis=0)
    attn_norm = _minmax(attn_ts)
    attn_up = _resample(attn_norm, n)
    layer_ts = attn_stack.mean(axis=1).mean(axis=1)          # (layers, T')
    layer_ts = (layer_ts - layer_ts.min(axis=1, keepdims=True)) / \
               (np.ptp(layer_ts, axis=1, keepdims=True) + 1e-8)

    lead_attn = _minmax(np.array([
        np.sum(attn_up * np.abs(signal[l])) / (np.sum(np.abs(signal[l])) + 1e-8) for l in range(12)
    ]))
    order = np.argsort(lead_attn)
    peak_t = t[int(np.argmax(attn_up))]

    with plt.rc_context(_ATTN_RC):
        fig = plt.figure(figsize=(14, 10), facecolor="white")
        gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1.5, 1.2], width_ratios=[2.5, 1],
                               hspace=0.45, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.08)

        ax_ecg = fig.add_subplot(gs[0, :])
        ax_ecg.plot(t, lead, color=ECGCOLOR, linewidth=0.9, zorder=3, alpha=0.9)
        ax_ecg.fill_between(t, lead.min() - 0.15, lead.min() - 0.15 + attn_up * 0.6,
                            alpha=0.35, color=ACCENT, zorder=2, label="Attention weight")
        peaks, _ = find_peaks(attn_up, height=0.4, distance=int(0.5 * fs))
        for pk in peaks[:3]:
            ax_ecg.axvline(t[pk], color=ACCENT, linewidth=0.8, alpha=0.6, linestyle="--", zorder=4)
        ax_ecg.set_xlim(t[0], t[-1])
        ax_ecg.set_xlabel("Time (s)", fontsize=10)
        ax_ecg.set_ylabel("Amplitude (mV)", fontsize=10)
        ax_ecg.set_title(f"A    Transformer attention overlay \u2014 {class_name} (Lead I)",
                         fontsize=11, fontweight="bold", loc="left", pad=8)
        ax_ecg.legend(fontsize=9, frameon=False, loc="upper right")
        ax_ecg.grid(color=GRIDCOL, linewidth=0.5)
        ax_ecg.tick_params(labelsize=9)

        ax_bar = fig.add_subplot(gs[1, 0])
        colors_bar = [ACCENT if lead_attn[i] > 0.6 else "#AAAAAA" for i in order]
        ax_bar.barh(range(12), lead_attn[order], color=colors_bar, height=0.65, edgecolor="none")
        ax_bar.set_yticks(range(12))
        ax_bar.set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=9)
        ax_bar.set_xlabel("Normalised attention score", fontsize=9)
        ax_bar.set_title("B    Attention per lead", fontsize=11, fontweight="bold", loc="left")
        ax_bar.grid(axis="x", color=GRIDCOL, linewidth=0.5)
        ax_bar.tick_params(labelsize=9)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.tick_params(left=False)

        ax_heat = fig.add_subplot(gs[1, 1])
        im = ax_heat.imshow(layer_ts, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax_heat.set_yticks(range(n_layers))
        ax_heat.set_yticklabels([f"L{i + 1}" for i in range(n_layers)], fontsize=8)
        ax_heat.set_xlabel("Time step", fontsize=9)
        ax_heat.set_title("C    Layer attention", fontsize=11, fontweight="bold", loc="left")
        plt.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.04, label="Attention")

        ax_note = fig.add_subplot(gs[2, :])
        ax_note.axis("off")
        ax_note.text(0.0, 0.75, "Computed summary:", fontsize=10, fontweight="bold",
                     transform=ax_note.transAxes, color=ECGCOLOR)
        ax_note.text(0.0, 0.15,
                     f"Peak attention at t = {peak_t:.1f} s; top-ranked lead by attention-weighted "
                     f"amplitude: {LEAD_NAMES[order[-1]]}. Averaged over {n_layers} layers and all heads.",
                     fontsize=9, transform=ax_note.transAxes, color="#444444", style="italic")
        fig.suptitle(f"Interpretability Analysis \u2014 {class_name} | {model_label(model_name)}",
                     fontsize=13, fontweight="bold", y=0.97, color=ECGCOLOR)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# =====================================================================
# Best vs worst case (notebook cell 46)
# =====================================================================
_BW_RC = {"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 150}


def pick_best_worst(y_true, y_pred, class_idx, threshold):
    """Notebook's selection. Returns (best_idx, best_conf, worst_idx, worst_conf, worst_label)
    or None if the class has no true positives at this threshold."""
    tp_idx = np.where((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] >= threshold))[0]
    if len(tp_idx) == 0:
        return None
    best_idx = tp_idx[np.argmax(y_pred[tp_idx, class_idx])]
    fn_idx = np.where((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] < threshold))[0]
    if len(fn_idx) > 0:
        worst_idx = fn_idx[np.argmin(y_pred[fn_idx, class_idx])]
        worst_label = "False Negative"
    else:
        worst_idx = tp_idx[np.argmin(y_pred[tp_idx, class_idx])]
        worst_label = "Least Certain TP"
    return (best_idx, float(y_pred[best_idx, class_idx]),
            worst_idx, float(y_pred[worst_idx, class_idx]), worst_label)


def plot_best_worst(class_name, model_name, best, worst, worst_label, save_path, fs=100):
    """best / worst: dicts with keys signal (12,T), cam (T,) normalised, limp (12,) normalised, conf."""
    with plt.rc_context(_BW_RC):
        n = best["signal"].shape[1]
        t = np.arange(n) / fs
        fig = plt.figure(figsize=(22, 12), facecolor="white")
        ax_eb = fig.add_axes([0.05, 0.40, 0.43, 0.48])
        ax_ew = fig.add_axes([0.53, 0.40, 0.43, 0.48])
        ax_bb = fig.add_axes([0.05, 0.06, 0.43, 0.30])
        ax_bw = fig.add_axes([0.53, 0.06, 0.43, 0.30])

        fig.text(0.5, 0.975, f"Best vs Worst Case \u2014 {class_name}  |  {model_label(model_name)}",
                 ha="center", fontsize=14, fontweight="bold", color=ECGCOLOR)
        fig.text(0.5, 0.948,
                 "Left: highest confidence correct prediction  |  Right: lowest confidence / missed case",
                 ha="center", fontsize=9, color="#6C757D", style="italic")
        fig.add_artist(plt.Line2D([0.505, 0.505], [0.04, 0.94], transform=fig.transFigure,
                                  color="#DDDDDD", linewidth=1.2, linestyle="--"))

        panels = [
            (ax_eb, best, BEST_COL, "A   Best Case", f"True Positive  \u00b7  confidence = {best['conf']:.3f}"),
            (ax_ew, worst, ACCENT, "B   " + worst_label,
             f"{worst_label}  \u00b7  confidence = {worst['conf']:.3f}"),
        ]
        for ax, d, color, title, subtitle in panels:
            sig, cam, conf = d["signal"], d["cam"], d["conf"]
            ax.set_facecolor("#FAFAFA")
            for sp in ax.spines.values():
                sp.set_color("#E0E0E0")
                sp.set_linewidth(0.8)
            ax.plot(t, sig[0], color=ECGCOLOR, linewidth=0.85, zorder=3, alpha=0.92)
            base = sig[0].min() - 0.4
            ax.fill_between(t, base, base + cam * 0.55, alpha=0.38, color=color, zorder=2)
            peaks, _ = find_peaks(cam, height=0.4, distance=int(0.5 * fs))
            for pk in peaks[:3]:
                ax.axvline(t[pk], color=color, linewidth=0.9, alpha=0.5, linestyle="--", zorder=4)
            ax.set_xlim(0, t[-1])
            ax.set_ylim(sig[0].min() - 0.6, sig[0].max() + 0.25)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Amplitude (mV)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(color=BW_GRIDCOL, linewidth=0.5, zorder=0)
            x_pos = ax.get_position().x0
            y_pos = ax.get_position().y1 + 0.005
            fig.text(x_pos, y_pos + 0.018, title, fontsize=11, fontweight="bold", color=color)
            fig.text(x_pos, y_pos + 0.002, subtitle, fontsize=9, color=color, style="italic")
            ax.text(0.985, 0.965, f"conf = {conf:.3f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, linewidth=1.2))
            ax.text(0.015, 0.04, "Grad-CAM \u2191", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color=color, style="italic")

        for ax, d, color, label in [
            (ax_bb, best, BEST_COL, "Lead importance \u2014 Best case"),
            (ax_bw, worst, ACCENT, "Lead importance \u2014 " + worst_label),
        ]:
            _lead_bars(ax, d["limp"], color, label)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _lead_bars(ax, limp, color, label):
    order = np.argsort(limp)
    colors_bar = [color if limp[i] > 0.6 else "#CCCCCC" for i in order]
    bars = ax.barh(range(12), limp[order], color=colors_bar, height=0.62, edgecolor="none")
    for bar, val in zip(bars, limp[order]):
        if val > 0.08:
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                    va="center", fontsize=7.5, color="#555")
    ax.set_yticks(range(12))
    ax.set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=9)
    ax.set_xlabel("Normalised Grad-CAM score", fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_title(label, fontsize=9, fontweight="bold", loc="left", color=color, pad=6)
    ax.grid(axis="x", color=BW_GRIDCOL, linewidth=0.5)
    ax.tick_params(labelsize=8, left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#E0E0E0")


# =====================================================================
# Negative example (notebook cell 48)
# =====================================================================
def pick_true_negative(y_true, y_pred, class_idx, threshold):
    """Most confidently-correct negative (lowest predicted probability). None if no TN."""
    tn_idx = np.where((y_true[:, class_idx] == 0) & (y_pred[:, class_idx] < threshold))[0]
    if len(tn_idx) == 0:
        return None
    idx = tn_idx[np.argmin(y_pred[tn_idx, class_idx])]
    return idx, float(y_pred[idx, class_idx])


def plot_negative_example(class_name, model_name, d, threshold, save_path, fs=100):
    """d: dict with signal (12,T), cam (T,) normalised, limp (12,) normalised, conf."""
    with plt.rc_context(_BW_RC):
        sig, cam, limp, conf = d["signal"], d["cam"], d["limp"], d["conf"]
        t = np.arange(sig.shape[1]) / fs
        fig = plt.figure(figsize=(14, 10), facecolor="white")
        ax_ecg = fig.add_axes([0.07, 0.42, 0.88, 0.46])
        ax_bar = fig.add_axes([0.07, 0.07, 0.88, 0.28])
        color = TN_COL

        fig.text(0.5, 0.975, f"Negative Example \u2014 {class_name}  |  {model_label(model_name)}",
                 ha="center", fontsize=13, fontweight="bold", color=ECGCOLOR)
        fig.text(0.5, 0.948,
                 f"True Negative  \u00b7  Model correctly predicted NO {class_name}  \u00b7  confidence = {conf:.3f}",
                 ha="center", fontsize=9, color="#6C757D", style="italic")

        ax_ecg.set_facecolor("#FAFAFA")
        for sp in ax_ecg.spines.values():
            sp.set_color("#E0E0E0")
            sp.set_linewidth(0.8)
        ax_ecg.plot(t, sig[0], color=ECGCOLOR, linewidth=0.85, zorder=3, alpha=0.92)
        base = sig[0].min() - 0.4
        ax_ecg.fill_between(t, base, base + cam * 0.55, alpha=0.3, color=color, zorder=2)
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylim(sig[0].min() - 0.6, sig[0].max() + 0.25)
        ax_ecg.set_xlabel("Time (s)", fontsize=9)
        ax_ecg.set_ylabel("Amplitude (mV)", fontsize=9)
        ax_ecg.tick_params(labelsize=8)
        ax_ecg.grid(color=BW_GRIDCOL, linewidth=0.5, zorder=0)
        ax_ecg.set_title("ECG Signal \u2014 No activation for this class (Lead I)", fontsize=10,
                         fontweight="bold", color=color, loc="left", pad=8)
        ax_ecg.text(0.985, 0.965, f"pred = {conf:.3f}  (< threshold {threshold:.2f})",
                    transform=ax_ecg.transAxes, ha="right", va="top", fontsize=9, fontweight="bold",
                    color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                           edgecolor=color, linewidth=1.2))
        ax_ecg.text(0.015, 0.04, "Grad-CAM \u2191", transform=ax_ecg.transAxes, ha="left", va="bottom",
                    fontsize=8, color=color, style="italic")
        _lead_bars(ax_bar, limp, color, "Lead importance \u2014 True Negative")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)