"""
thresholds.py
=============
Decision-threshold rules, shared by eval.py and make_figures.py.

Rule of use (this is what makes the reported numbers honest):
    FIT thresholds on the VALIDATION fold (9), APPLY + REPORT on the TEST fold (10).

Three rules, per class:
    fixed_0.5 : 0.5 for every class
    f1_opt    : threshold maximizing F1 on validation
    f2_opt    : threshold maximizing F2 on validation (recall-weighted; clinical screening)
"""

import numpy as np
from sklearn.metrics import roc_auc_score

GRID = np.round(np.arange(0.05, 0.90, 0.01), 2)
RULES = ("fixed_0.5", "f1_opt", "f2_opt")


def _counts(y, p, t):
    pred = p >= t
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float(((~pred) & (y == 1)).sum())
    return tp, fp, fn


def _fbeta(tp, fp, fn, beta):
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom > 0 else 0.0


def fit_thresholds(y_true, y_prob, beta):
    """Per-class threshold maximizing F-beta on the given (validation) data."""
    thr = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue  # no positives: keep 0.5
        scores = [_fbeta(*_counts(y_true[:, i], y_prob[:, i], t), beta) for t in GRID]
        thr[i] = GRID[int(np.argmax(scores))]
    return thr


def get_thresholds(rule, y_val_true, y_val_prob):
    if rule == "fixed_0.5":
        return np.full(y_val_true.shape[1], 0.5)
    if rule == "f1_opt":
        return fit_thresholds(y_val_true, y_val_prob, beta=1)
    if rule == "f2_opt":
        return fit_thresholds(y_val_true, y_val_prob, beta=2)
    raise ValueError(rule)


def score_at(y_true, y_prob, thr):
    """Per-class P/R/F1/F2 plus macro and micro averages, at the given thresholds."""
    per = []
    TP = FP = FN = 0.0
    for i in range(y_true.shape[1]):
        tp, fp, fn = _counts(y_true[:, i], y_prob[:, i], thr[i])
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        per.append(dict(precision=p, recall=r, f1=_fbeta(tp, fp, fn, 1), f2=_fbeta(tp, fp, fn, 2)))
        TP += tp; FP += fp; FN += fn
    macro = {k: float(np.mean([d[k] for d in per])) for k in ("precision", "recall", "f1", "f2")}
    micro = dict(precision=TP / (TP + FP) if TP + FP > 0 else 0.0,
                 recall=TP / (TP + FN) if TP + FN > 0 else 0.0,
                 f1=_fbeta(TP, FP, FN, 1), f2=_fbeta(TP, FP, FN, 2))
    return per, macro, micro


def macro_auroc(y_true, y_prob):
    vals = []
    for i in range(y_true.shape[1]):
        if 0 < y_true[:, i].sum() < len(y_true):
            vals.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(vals)) if vals else float("nan")