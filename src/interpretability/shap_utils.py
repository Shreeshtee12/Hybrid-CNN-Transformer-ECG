"""
shap_utils.py
=============
SHAP (KernelExplainer) for 1D ECG signals, extracted from
CNN_Transformer_Hybrid.ipynb and generalized to work with any target class.

The notebook downsampled each 12-lead signal to 50 bins per lead before
running KernelExplainer (SHAP is too slow to run on raw 1000-length signals
directly) then expanded predictions back to full length inside the model
wrapper. Same approach here, just not hardcoded to one class at a time.
"""

import numpy as np
import shap
import torch


def downsample_signals(signals, n_bins=50, signal_length=1000):
    """(N, 12, signal_length) -> (N, 12, n_bins), each bin = mean of its segment."""
    segment_length = signal_length // n_bins
    n = signals.shape[0]
    reduced = signals[:, :, : n_bins * segment_length].reshape(
        n, 12, n_bins, segment_length
    ).mean(axis=3)
    return reduced


def make_model_predict_fn(model, device, n_bins=50, signal_length=1000):
    """
    Builds the closure SHAP calls repeatedly. Bound to a specific model/device
    instead of relying on a global `model` variable like the notebook did.
    """
    expand_factor = signal_length // n_bins

    def model_predict(x, _batch_size=32):
        x = np.array(x)
        x = x.reshape(x.shape[0], 12, n_bins)
        x = np.repeat(x, expand_factor, axis=2)
        if x.shape[2] != signal_length:
            # handle signal_length not evenly divisible by n_bins
            pad = signal_length - x.shape[2]
            x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="edge")
        # Chunk into small batches: SHAP's KernelExplainer can pass hundreds of
        # perturbed rows in one call, and an LSTM's memory use per sample (scanned
        # over the full sequence) is far steeper than a CNN's -- passing them all
        # at once is what caused a 21+ GiB allocation for xlstm even though the
        # same code was fine for the CNN-based models.
        all_preds = []
        with torch.no_grad():
            for start in range(0, x.shape[0], _batch_size):
                chunk = x[start:start + _batch_size]
                x_t = torch.tensor(chunk, dtype=torch.float32, device=device)
                all_preds.append(model(x_t).cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    return model_predict


def compute_shap_for_class(model, device, signals, target_class,
                            n_background=5, n_explain=2, nsamples=50,
                            n_bins=50, signal_length=1000):
    """
    Runs SHAP KernelExplainer for one class, returns per-lead importance.

    Args:
        signals: numpy array (N, 12, signal_length) of samples for this class
        target_class: integer class index

    Returns:
        lead_importance: (n_explain, 12) mean absolute SHAP value per lead
        shap_values_3d: (n_explain, 12, n_bins) raw SHAP values, for plotting
    """
    reduced = downsample_signals(signals, n_bins=n_bins, signal_length=signal_length)
    flat = reduced.reshape(reduced.shape[0], -1)

    background = flat[:n_background]
    model_predict = make_model_predict_fn(model, device, n_bins, signal_length)

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(flat[:n_explain], nsamples=nsamples)

    # SHAP's return format for multi-output models differs across versions:
    #   older shap: a list of length n_classes, each (n_samples, n_features)
    #   newer shap: a single ndarray (n_samples, n_features, n_classes)
    # Handle both so this doesn't silently misindex depending on installed version.
    if isinstance(shap_values, list):
        class_shap = shap_values[target_class]              # (n_explain, 12*n_bins)
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            class_shap = shap_values[:, :, target_class]     # (n_explain, 12*n_bins)
        else:
            # Single-output model (shouldn't happen for our 8-class model, but
            # fail loudly rather than silently returning the wrong thing).
            raise ValueError(
                f"Unexpected SHAP output shape {shap_values.shape} for a "
                f"multi-class model -- check installed shap version."
            )

    class_shap_3d = class_shap.reshape(-1, 12, n_bins)
    lead_importance = np.mean(np.abs(class_shap_3d), axis=2)  # (n_explain, 12)

    return lead_importance, class_shap_3d