"""
metrics.py
==========
Quantitative comparison between Grad-CAM and SHAP -- this is the actual
upgrade over the notebook, which only ever showed the two methods as
separate pictures with no numeric agreement score.

Design note: Grad-CAM (as implemented in gradcam.py) produces a TEMPORAL
attention curve in the model's internal feature space -- which moments in
time mattered. SHAP (as implemented in shap_utils.py) produces a PER-LEAD
importance score -- which of the 12 ECG leads mattered. These are not
directly comparable as-is.

To make a fair comparison, gradcam_lead_importance() converts Grad-CAM's
temporal curve into a per-lead score by weighting each lead's raw signal
amplitude by the temporal attention, then summing over time. This answers
"which leads does Grad-CAM's attention actually fall on?" in the same
units as SHAP's per-lead importance, so the two can be rank-correlated.
"""

import numpy as np
from scipy.stats import spearmanr


def gradcam_lead_importance(cam, input_signal):
    """
    Converts a temporal Grad-CAM curve into a per-lead importance score.

    Args:
        cam: (T,) Grad-CAM curve, already resampled to signal length
        input_signal: (12, T) the raw ECG signal it was computed on

    Returns:
        (12,) normalized per-lead importance
    """
    weighted = np.abs(input_signal) * cam[None, :]
    lead_scores = weighted.sum(axis=1)
    total = lead_scores.sum()
    return lead_scores / total if total > 0 else lead_scores


def agreement_score(gradcam_leads, shap_leads):
    """
    Spearman rank correlation between Grad-CAM's and SHAP's per-lead
    importance rankings. +1 = perfect agreement on which leads matter most,
    0 = no relationship, -1 = they disagree completely.
    """
    corr, p_value = spearmanr(gradcam_leads, shap_leads)
    return float(corr), float(p_value)