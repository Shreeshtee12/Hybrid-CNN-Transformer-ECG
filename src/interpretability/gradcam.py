"""
gradcam.py
==========
Grad-CAM for 1D ECG signals, extracted from CNN_Transformer_Hybrid.ipynb and
generalized to work with any target class (the notebook only ever ran this
for AFIB and LVH, hardcoded each time).

Default hook target is model.cnn4, the last residual CNN block of
OptimizedHybrid (src/models/torch_models/cnn_transformer.py). Pass a
different target_layer to use this on another architecture.
"""

import numpy as np
import torch


def gradcam_1d(model, input_signal, target_class, device, target_layer=None):
    """
    Computes a Grad-CAM heatmap for one ECG signal and one target class.

    Args:
        model: a trained model in eval mode (or will be set to eval here)
        input_signal: tensor of shape (C, T) or (1, C, T)
        target_class: integer class index (0-7 for the 8 TARGET_CLASSES)
        device: torch.device
        target_layer: the layer to hook (defaults to model.cnn4)

    Returns:
        cam: numpy array of shape (T',) -- normalized to [0, 1], where T' is
             the hooked layer's output length (shorter than the input length;
             callers typically resample this back to the original length).
    """
    model.eval()
    input_signal = input_signal.to(device)

    if input_signal.dim() == 2:  # (C, T) -> (1, C, T)
        input_signal = input_signal.unsqueeze(0)
    assert input_signal.dim() == 3, f"Expected (B, C, T), got {input_signal.shape}"

    if target_layer is None:
        target_layer = model.cnn4

    gradients = []
    activations = []

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # cuDNN's RNN kernel cannot run backward() while the module is in eval()
    # mode ("cudnn RNN backward can only be called in training mode").
    # Disabling cuDNN for just this forward+backward falls back to a generic
    # implementation that supports it. No-op for CNN-only models.
    with torch.backends.cudnn.flags(enabled=False):
        output = model(input_signal)
        score = output[:, target_class]

        model.zero_grad()
        score.backward(torch.ones_like(score))

    grads = gradients[0].detach().cpu().numpy()[0]   # (C, T')
    acts = activations[0].detach().cpu().numpy()[0]  # (C, T')

    handle_fwd.remove()
    handle_bwd.remove()

    weights = np.mean(grads, axis=1)               # global average pooling
    cam = np.zeros(acts.shape[1])
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)                        # ReLU
    cam = cam / (np.max(cam) + 1e-8)                # normalize
    return cam


def resample_cam(cam, target_length):
    """Resamples a CAM (shorter, due to CNN downsampling) back to signal length."""
    x_original = np.linspace(0, len(cam) - 1, len(cam))
    x_target = np.linspace(0, len(cam) - 1, target_length)
    return np.interp(x_target, x_original, cam)


def compute_mean_cam(model, signals, target_class, device, signal_length, target_layer=None):
    """
    Runs Grad-CAM over a list/array of signals for one class and returns the
    MEAN cam, resampled to signal_length. This is what the notebook did
    manually once per class (AFIB, then LVH) -- generalized here to work for
    any class so it can be looped over all 8.
    """
    cams = []
    for signal in signals:
        if not torch.is_tensor(signal):
            signal = torch.tensor(signal, dtype=torch.float32)
        cam = gradcam_1d(model, signal, target_class, device, target_layer)
        cams.append(resample_cam(cam, signal_length))
    return np.array(cams).mean(axis=0), np.array(cams)