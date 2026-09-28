"""
make_macro_roc_comparison.py
=============================
Overlays the macro-average ROC curve of every model that make_figures.py has
been run for (reads results/preds_<model>.npz). Reproduces the notebook's
macro_roc_comparison.pdf, but with all models found (the notebook's version
had ResNet1D and Transformer commented out).

Usage (from src/, after make_figures.py for each model):
    python make_macro_roc_comparison.py
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import config
import ecg_figures as F

ORDER = ["cnn_transformer", "resnet1d", "xresnet1d", "transformer", "xlstm"]


def main():
    found = {}
    for path in glob.glob(str(config.RESULTS_DIR / "preds_*.npz")):
        name = os.path.basename(path)[len("preds_"):-len(".npz")]
        d = np.load(path)
        found[name] = (d["y_true"], d["y_pred"])
    if not found:
        print("No results/preds_*.npz found. Run make_figures.py for at least one model first.")
        return

    preds = {m: found[m] for m in ORDER if m in found}
    preds.update({m: v for m, v in found.items() if m not in preds})

    # Overlaying curves is only meaningful if every model was scored on the same records.
    ref = next(iter(preds.values()))[0]
    for m, (yt, _) in preds.items():
        if yt.shape != ref.shape or not np.array_equal(yt, ref):
            print(f"[warn] {m} was evaluated on different labels/records than the others; "
                  f"the comparison may not be like-for-like.")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "macro_roc_comparison.pdf"
    aucs = F.plot_macro_roc_comparison(preds, out)
    table = pd.DataFrame({"model": list(aucs), "macro_auc": list(aucs.values())})
    table.to_csv(config.RESULTS_DIR / "macro_auc_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"[compare] Saved -> {out}")


if __name__ == "__main__":
    main()