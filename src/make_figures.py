"""
make_figures.py
================
Generates, for ONE trained model, the same figure set your notebook made for
the hybrid model -- but for every class and every supported architecture:

    ROC curves (per-class + macro)            roc_curves.png
    Confusion matrices (F2-tuned thresholds)  confusion.pdf
    Grad-CAM panel A / panels B+C, per class  gradcam_<cls>_panel_A.pdf, gradcam_<cls>_panels_BC.pdf
    SHAP figure, per class                    shap_<cls>.pdf
    Transformer attention, per class          attention_<cls>.pdf   (models with transformer blocks only)
    Best vs worst case, per class             best_worst_<cls>.pdf
    Negative (true-negative) example          negative_<cls>.pdf

Everything lands in  <repo>/figures/<model>/ . Predictions are also saved to
results/preds_<model>.npz so make_macro_roc_comparison.py can overlay models.

Checkpoint choice: by default this picks the checkpoint with the LOWEST
val_loss in the model's checkpoint folder. (eval.py's pick_checkpoint takes
the alphabetically-last file, which is last.ckpt for some model names and a
val_loss checkpoint for others -- pass --checkpoint explicitly to override.)

IMPORTANT honesty notes (also printed at run time):
  * The repo currently has only a train/val split. The val set was used for
    early stopping / checkpoint selection, AND here it is used to tune the
    per-class thresholds AND to report the confusion matrices. That is
    optimistic. A held-out test split is the proper fix.
  * Grad-CAM figures need a conv layer to hook: supported for cnn_transformer,
    resnet1d, xresnet1d, xlstm (xlstm/xresnet1d hooks are untested). The plain
    transformer gets ROC / confusion / SHAP only.

Usage (from src/):
    python make_figures.py --model cnn_transformer \
        --signal-path /path/to/ptb-xl/1.0.3 \
        --csv-path   /path/to/ptb-xl/1.0.3/ptbxl_database.csv
"""

import argparse
import ast
import glob
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "interpretability"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

import config
import ecg_figures as F
from data_split import load_split
from eval import (
    PTBXL_Dataset, TARGET_CLASSES, pick_checkpoint,
    infer_num_classes_from_state, infer_model_name_from_state,
)
from gradcam import gradcam_1d, compute_mean_cam, resample_cam
from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model
from run_interpretability import get_target_layer, collect_class_samples, to_chw
from shap_utils import compute_shap_for_class


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
class OrientedForSHAP:
    """shap_utils feeds (B, C, T). PTBXL_Dataset / eval.py feed (B, T, C) into the
    Lightning wrapper. This bridges the two so models whose first layer is Linear
    (the plain transformer) also work, using the exact input path eval.py uses."""
    def __init__(self, lit_model):
        self.m = lit_model

    def __call__(self, x):
        return self.m(x.transpose(1, 2))


def gradcam_abs(model, signal, target_class, device, target_layer):
    """Notebook's gradcam_1d_tn: ABSOLUTE gradients and activations, no ReLU.
    This is a DIFFERENT quantity from standard Grad-CAM (it also highlights
    suppression); the notebook used it for negative examples, so we do too."""
    model.eval()
    x = signal.to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    grads, acts = [], []
    hf = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    hb = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        with torch.backends.cudnn.flags(enabled=False):
            out = model(x)
            score = out[:, target_class]
            model.zero_grad()
            score.backward(torch.ones_like(score))
    finally:
        hf.remove()
        hb.remove()
    g = grads[0].detach().cpu().numpy()[0]
    a = acts[0].detach().cpu().numpy()[0]
    w = np.mean(np.abs(g), axis=1)
    cam = np.zeros(a.shape[1])
    for i, wi in enumerate(w):
        cam += wi * np.abs(a[i])
    return cam / (np.max(cam) + 1e-8)


def cam_bundle(model, sig_ct, cls, device, target_layer, absolute=False):
    x = torch.tensor(sig_ct, dtype=torch.float32)
    cam = (gradcam_abs(model, x, cls, device, target_layer) if absolute
           else gradcam_1d(model, x, cls, device, target_layer))
    cam = resample_cam(cam, sig_ct.shape[1])
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    return cam, F.lead_importance_from_cam(cam, sig_ct)


def extract_attention_weights(model, sig_ct, device):
    """Notebook cell 22, unchanged logic. Only for models with transformer_blocks."""
    x = torch.tensor(sig_ct, dtype=torch.float32, device=device).unsqueeze(0)
    weights = []
    with torch.no_grad():
        x = model.cnn1(x)
        x = model.cnn2(x)
        x = model.cnn3(x)
        x = model.cnn4(x)
        x = x.transpose(1, 2)
        x = model.pos_encoding(x)
        for block in model.transformer_blocks:
            xn = block.norm1(x)
            attn_out, attn_w = block.attn(xn, xn, xn, need_weights=True, average_attn_weights=False)
            weights.append(attn_w.squeeze(0).detach().cpu().numpy())
            x = x + attn_out
            x = x + block.mlp(block.norm2(x))
    return np.stack(weights)   # (layers, heads, T', T')


def predict_all(lit_model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.sigmoid(lit_model(x.to(device))).cpu().numpy())
    return np.concatenate(probs)


def sample_ct(dataset, idx):
    sig_tc, _ = dataset[int(idx)]
    arr = sig_tc.numpy() if torch.is_tensor(sig_tc) else np.asarray(sig_tc)
    return arr.T   # (T, C) -> (C, T)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--signal-path", required=True)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--sr", type=int, default=100, choices=[100, 500])
    ap.add_argument("--input-channels", type=int, default=12)
    ap.add_argument("--classes", nargs="+", default=None, help="Subset of classes (default: all 8)")
    ap.add_argument("--n-gradcam-samples", type=int, default=50)
    ap.add_argument("--n-shap-background", type=int, default=5)
    ap.add_argument("--n-shap-explain", type=int, default=8,
                    help="Notebook explained 8 samples for its SHAP figures.")
    ap.add_argument("--shap-nsamples", type=int, default=300,
                    help="Notebook used 50, which is very few for 600 features (SHAP warned the "
                         "regression was singular). Raise further for the journal (time cost).")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-gradcam", action="store_true")
    ap.add_argument("--split-gradcam-panels", action="store_true",
                    help="Also save the separate panel_A / panels_BC files alongside the combined one.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb", action="store_true",
                    help="Upload every generated figure to Weights & Biases as it's saved.")
    ap.add_argument("--project", type=str, default="ptbxl-ecg", help="W&B project name")
    ap.add_argument("--entity", type=str, default=None, help="W&B entity (user or team)")
    ap.add_argument("--run-name", type=str, default=None,
                    help="W&B run name (default: <model>-figures)")
    args = ap.parse_args()

    seq_len = 5000 if args.sr == 500 else 1000
    fs = args.sr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[figs] NOTE: thresholds are tuned AND reported on the validation set (no held-out "
          "test split exists yet) -- optimistic; see docstring.")

    # ---- data (identical to eval.py / run_interpretability.py) ----
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["scp_filtered"] = df["scp_codes"].apply(
        lambda codes: [k for k, c in codes.items() if k in TARGET_CLASSES and c >= 50])
    df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)
    y = MultiLabelBinarizer(classes=TARGET_CLASSES).fit_transform(df["scp_filtered"])
    records = df["filename_lr"].str.replace(".hea", "", regex=False).values
    split = load_split()
    mask = np.isin(records, split["val_records"])
    val_ds = PTBXL_Dataset(records[mask], y[mask], args.signal_path, sr=args.sr)
    y_true = y[mask].astype(int)
    print(f"[figs] {len(val_ds)} validation records")

    # ---- checkpoint + model ----
    ckpt_dir = args.checkpoint_dir or str(config.LIGHTNING_LOGS_DIR / "checkpoints" / args.model)
    class _A: checkpoint = args.checkpoint; checkpoint_dir = ckpt_dir
    ckpt = pick_checkpoint(_A())  # the one checkpoint-selection function, from eval.py
    print(f"[figs] Checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["state_dict"] if "state_dict" in state else state
    detected = infer_model_name_from_state(sd, args.model)
    model_obj = get_model(detected, input_shape=(args.input_channels, seq_len),
                          num_classes=infer_num_classes_from_state(sd))
    lit_model = LitGenericModel(model_obj)
    lit_model.load_state_dict(sd, strict=False)
    lit_model.eval().to(device)
    model = lit_model.model

    out_dir = config.FIGURES_DIR / detected
    out_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.project, entity=args.entity,
            name=args.run_name or f"{detected}-figures",
            job_type="figures", config={"model": detected, "checkpoint": str(ckpt)},
        )

    def _to_png_bytes(path):
        """PDF -> PNG bytes (first page) for W&B display. PNG files pass through untouched."""
        path = str(path)
        if path.endswith(".png"):
            with open(path, "rb") as f:
                return f.read()
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print(f"[figs]   [wandb] skipped {path}: PyMuPDF not installed "
                  f"(pip install pymupdf) -- figure saved locally, just not uploaded")
            return None
        doc = fitz.open(path)
        pix = doc[0].get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    def log_figure(key, path):
        """Uploads one saved figure to the current W&B run. No-op if --wandb wasn't passed."""
        if wandb_run is None:
            return
        png_bytes = _to_png_bytes(path)
        if png_bytes is None:
            return
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        wandb_run.log({key: wandb.Image(img, caption=str(path))})

    def attempt(label, fn, wandb_key=None):
        """
        Runs fn(). If fn() returns a path (or list of paths) and --wandb is on,
        each returned path gets uploaded under wandb_key (default: label, with
        spaces -> "/" so e.g. "gradcam AFIB" logs as "gradcam/AFIB").
        """
        try:
            result = fn()
            print(f"[figs]   ok: {label}")
            if wandb_run is not None and result:
                key = wandb_key or label.replace(" ", "/")
                paths = result if isinstance(result, (list, tuple)) else [result]
                for p in paths:
                    log_figure(key, p)
        except Exception as e:
            failures.append((label, repr(e)))
            print(f"[figs]   FAILED: {label}: {e!r}")
            traceback.print_exc()

    # ---- predictions, thresholds, ROC, confusion ----
    y_pred = predict_all(lit_model, val_ds, device, args.batch_size, args.num_workers)
    np.savez(config.RESULTS_DIR / f"preds_{detected}.npz", y_true=y_true, y_pred=y_pred)
    thr = F.optimal_f2_thresholds(y_true, y_pred)
    rows = []
    for i, c in enumerate(TARGET_CLASSES):
        yb = (y_pred[:, i] >= thr[i]).astype(int)
        p = precision_score(y_true[:, i], yb, zero_division=0)
        r = recall_score(y_true[:, i], yb, zero_division=0)
        f2 = 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0
        rows.append(dict(model=detected, cls=c, n_pos_val=int(y_true[:, i].sum()),
                         threshold=thr[i], precision=p, recall=r, f2=f2))
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / f"f2_thresholds_{detected}.csv", index=False)

    attempt("roc_curves", lambda: (F.plot_roc(y_true, y_pred, TARGET_CLASSES, detected, out_dir / "roc_curves.png"), out_dir / "roc_curves.png")[-1])
    attempt("confusion", lambda: (F.plot_confusion_grid(y_true, y_pred, thr, TARGET_CLASSES, detected, out_dir / "confusion.pdf"), out_dir / "confusion.pdf")[-1])

    # ---- Native W&B charts (built from the raw numbers, not uploaded images) ----
    # ROC and confusion matrix are the two figure types W&B can draw itself, as
    # interactive charts, given the raw data. Everything else below this point
    # (Grad-CAM, SHAP, attention, best/worst, negative) is a custom visualization
    # with no native W&B chart type -- those stay as uploaded images, which is
    # the correct and only way to log a custom matplotlib figure to W&B.
    if wandb_run is not None:
        try:
            wandb_run.log({
                "native/roc_curves": wandb.plot.roc_curve(
                    y_true, y_pred, labels=TARGET_CLASSES,
                    classes_to_plot=list(range(len(TARGET_CLASSES))),
                )
            })
            print("[figs]   ok: native roc_curves (wandb)")
        except Exception as e:
            print(f"[figs]   FAILED: native roc_curves (wandb): {e!r}")

        # wandb's confusion_matrix expects single-label classification (one true
        # class per sample). This is multi-label (8 independent yes/no decisions),
        # so we log ONE native confusion matrix per class -- each is a valid,
        # correct 2-class (absent/present) confusion matrix on its own.
        for i, cname in enumerate(TARGET_CLASSES):
            try:
                yt_i = y_true[:, i]
                yp_i = (y_pred[:, i] >= thr[i]).astype(int)
                wandb_run.log({
                    f"native/confusion_matrix/{cname}": wandb.plot.confusion_matrix(
                        y_true=yt_i.tolist(), preds=yp_i.tolist(),
                        class_names=[f"No {cname}", cname],
                    )
                })
            except Exception as e:
                print(f"[figs]   FAILED: native confusion_matrix {cname} (wandb): {e!r}")
        print("[figs]   ok: native confusion matrices (wandb), one per class")

    # ---- capabilities ----
    target_layer = None if args.skip_gradcam else get_target_layer(model, detected)
    gradcam_ok = target_layer is not None
    attention_ok = hasattr(model, "transformer_blocks") and hasattr(model, "cnn4")
    if not gradcam_ok:
        print(f"[figs] Grad-CAM figures skipped for '{detected}' (unsupported or --skip-gradcam).")
    shap_model = OrientedForSHAP(lit_model)

    # ---- per-class figures ----
    classes = args.classes or TARGET_CLASSES
    for cname in classes:
        ci = TARGET_CLASSES.index(cname)
        lc = cname.lower()
        print(f"\n[figs] === {cname} ===")
        pos = collect_class_samples(val_ds, ci, args.n_gradcam_samples, positive=True)
        if len(pos) == 0:
            print(f"[figs]   no positive validation samples, skipping {cname}")
            continue
        pos_ct = to_chw(pos, seq_len)
        rep = pos_ct[0]

        if gradcam_ok:
            def _gc():
                mean_cam, all_cams = compute_mean_cam(model, pos_ct, ci, device, seq_len, target_layer)
                gc_path = out_dir / f"gradcam_{lc}.pdf"
                F.plot_gradcam_combined(all_cams, rep, mean_cam, cname, gc_path, fs)
                if args.split_gradcam_panels:
                    F.plot_gradcam_panel_a(rep, mean_cam, cname, out_dir / f"gradcam_{lc}_panel_A.pdf", fs)
                    F.plot_gradcam_panels_bc(all_cams, rep, mean_cam, cname, out_dir / f"gradcam_{lc}_panels_BC.pdf", fs)
                return gc_path
            attempt(f"gradcam {cname}", _gc)

        if not args.skip_shap:
            def _shap():
                sig_n = pos_ct[: max(args.n_shap_background, args.n_shap_explain)]
                _, shap3d = compute_shap_for_class(
                    shap_model, device, sig_n, ci, n_background=args.n_shap_background,
                    n_explain=args.n_shap_explain, nsamples=args.shap_nsamples, signal_length=seq_len)
                shap_path = out_dir / f"shap_{lc}.pdf"
                F.plot_shap_figure(cname, rep, shap3d, detected, shap_path, fs, args.shap_nsamples)
                return shap_path
            attempt(f"shap {cname}", _shap)

        if attention_ok:
            attempt(f"attention {cname}", lambda: (F.plot_attention_figure(
                cname, rep, extract_attention_weights(model, rep, device), detected,
                out_dir / f"attention_{lc}.pdf", fs), out_dir / f"attention_{lc}.pdf")[-1])

        if gradcam_ok:
            def _bw():
                pick = F.pick_best_worst(y_true, y_pred, ci, thr[ci])
                if pick is None:
                    print(f"[figs]   no true positives at threshold {thr[ci]:.2f} for {cname}; "
                          f"best/worst skipped")
                    return
                b_i, b_conf, w_i, w_conf, w_label = pick
                b_cam, b_limp = cam_bundle(model, sample_ct(val_ds, b_i), ci, device, target_layer)
                w_cam, w_limp = cam_bundle(model, sample_ct(val_ds, w_i), ci, device, target_layer)
                bw_path = out_dir / f"best_worst_{lc}.pdf"
                F.plot_best_worst(
                    cname, detected,
                    dict(signal=sample_ct(val_ds, b_i), cam=b_cam, limp=b_limp, conf=b_conf),
                    dict(signal=sample_ct(val_ds, w_i), cam=w_cam, limp=w_limp, conf=w_conf),
                    w_label, bw_path, fs)
                return bw_path
            attempt(f"best_worst {cname}", _bw)

            def _neg():
                tn = F.pick_true_negative(y_true, y_pred, ci, thr[ci])
                if tn is None:
                    print(f"[figs]   no true negatives for {cname}; negative example skipped")
                    return
                t_i, t_conf = tn
                sig = sample_ct(val_ds, t_i)
                cam, limp = cam_bundle(model, sig, ci, device, target_layer, absolute=True)
                neg_path = out_dir / f"negative_{lc}.pdf"
                F.plot_negative_example(cname, detected, dict(signal=sig, cam=cam, limp=limp, conf=t_conf),
                                        thr[ci], neg_path, fs)
                return neg_path
            attempt(f"negative {cname}", _neg)

    print(f"\n[figs] Done. Figures in {out_dir}")
    print(f"[figs] Thresholds  -> {config.RESULTS_DIR / f'f2_thresholds_{detected}.csv'}")
    print(f"[figs] Predictions -> {config.RESULTS_DIR / f'preds_{detected}.npz'}")
    if failures:
        print(f"\n[figs] {len(failures)} figure(s) FAILED:")
        for label, err in failures:
            print(f"   - {label}: {err}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()