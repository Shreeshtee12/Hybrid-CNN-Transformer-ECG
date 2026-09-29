# Hybrid CNN–Transformer Models for Multi-Label ECG Classification with Interpretability

Multi-label classification of 12-lead ECG signals on the **PTB-XL** dataset, comparing a hybrid
CNN–Transformer architecture against four baselines (ResNet1D, xResNet1D, xLSTM, and a plain
Transformer), with Grad-CAM, SHAP, and attention-based interpretability.

## Branches

- **`main`** — the thesis-submitted code. Kept untouched; do not merge new work into it.
- **`new-experiments`** — the active, corrected pipeline (this branch). All 30%-new-experiment
  work for the journal submission happens here.

## Dataset

Experiments use **PTB-XL** ([physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/)).
Due to licensing, the dataset itself is not included in this repository — set `DATASET_PATH`
in `src/config.py` (or pass `--signal-path`/`--csv-path`) to point at your local copy.

The pipeline uses the **100Hz** records (`filename_lr`, 1000 samples/record) and only counts an
SCP diagnostic code as a positive label when its **confidence ≥ 50**, matching the settings used
to produce the results reported in the thesis.

## The 8 target classes

`NORM, AFIB, PVC, LVH, IMI, ASMI, LAFB, IRBBB`

**Known caveat:** AFIB has very few positive records under the confidence≥50 rule (single
digits in the validation split). Treat any AFIB-specific metric or figure as exploratory,
not a reliable result, until this is investigated further.

## Repository structure

```
src/
├── config.py                    # paths, seed, class list -- single source of truth
├── data_split.py                # generates the ONE shared patient-grouped train/val split
├── train_lightning.py           # trains one architecture (PyTorch Lightning)
├── eval.py                      # evaluates one checkpoint; also exposes pick_checkpoint()
├── run_all_models.py            # trains + evaluates + figures for all 5 architectures
├── merge_results.py             # rebuilds results/model_comparison.csv if needed
├── ecg_figures.py                # all figure-drawing functions (ROC, confusion, Grad-CAM, SHAP, ...)
├── make_figures.py              # generates the full figure set for one checkpoint
├── make_macro_roc_comparison.py # overlays all models' macro ROC curves
├── confusion_matrices.py        # merges per-model confusion-matrix stats into one table
├── run_interpretability.py      # SHAP-vs-Grad-CAM agreement scores, all 8 classes
├── interpretability/
│   ├── gradcam.py                # Grad-CAM (per-model target-layer lookup lives here)
│   ├── shap_utils.py             # SHAP KernelExplainer wrapper
│   └── metrics.py                # SHAP/Grad-CAM agreement metric
└── models/
    ├── torch_models/              # the 5 architecture definitions + model_selector.py
    └── components/lit_generic.py  # shared PyTorch Lightning wrapper

archive/            # superseded code, kept for reference (not imported by anything)
notebooks/archive/  # original exploratory notebooks (source of the thesis's real model + labels)
scripts/archive/    # old, pre-fix training/inference scripts
run_full_experiment.sbatch   # submits the full 100-epoch run on OSC (Slurm)
```

## Setup

```bash
git clone https://github.com/Shreeshtee12/Hybrid-CNN-Transformer-ECG.git
cd Hybrid-CNN-Transformer-ECG
git checkout new-experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytorch_lightning torchmetrics shap pymupdf wandb   # not yet in requirements.txt
```

## Running it

**1. Build the shared split (run once, or whenever the label rule/seed changes):**
```bash
cd src
python data_split.py
```
Confirms `Patient overlap: 0` — if it's not 0, stop before training anything.

**2. Train + evaluate + generate figures for all 5 architectures:**
```bash
python run_all_models.py \
  --signal-path /path/to/ptb-xl/1.0.3 \
  --csv-path /path/to/ptb-xl/1.0.3/ptbxl_database.csv \
  --max-epochs 100 \
  --figures \
  --wandb --project ptbxl-ecg
```
Or submit as a batch job on OSC: `sbatch run_full_experiment.sbatch`.

**3. One model at a time**, if you just want to retrain or re-figure a single architecture:
```bash
python train_lightning.py --model resnet1d --signal-path ... --csv-path ...
python eval.py --model resnet1d --checkpoint-dir lightning_logs/checkpoints/resnet1d --signal-path ... --csv-path ...
python make_figures.py --model resnet1d --checkpoint-dir lightning_logs/checkpoints/resnet1d --signal-path ... --csv-path ...
```

**4. After training, cross-model comparisons:**
```bash
python make_macro_roc_comparison.py   # figures/macro_roc_comparison.pdf
python confusion_matrices.py          # results/confusion_matrix_comparison.csv
```

**5. Interpretability (SHAP vs Grad-CAM agreement), per model:**
```bash
python run_interpretability.py --model cnn_transformer --checkpoint-dir ... --signal-path ... --csv-path ...
```

## Outputs

- `splits/split_seed<N>.json` — the one shared train/val split
- `lightning_logs/checkpoints/<model>/` — checkpoints, tagged with model name (no collisions)
- `lightning_logs/metrics/<checkpoint>/` — per-checkpoint eval artifacts (CSVs, confusion PNGs)
- `figures/<model>/` — ROC, confusion grid, Grad-CAM (combined panel), SHAP, attention
  (hybrid only), best/worst case, negative example — for every class
- `results/model_comparison.csv` — one row per architecture, identical metrics
- Weights & Biases (`--wandb`): training curves auto-logged by Lightning; `make_figures.py
  --wandb` additionally uploads every figure as an image, plus native interactive ROC/confusion
  charts under keys `native/roc_curves` and `native/confusion_matrix/<class>`

## Interpretability methods

Three methods are used, and they don't all apply to every architecture — for different reasons:

**SHAP** works on every model, unconditionally. It only ever calls `model(x)` and looks at how
the output changes as inputs are perturbed — it has no idea what's inside the model, so it
doesn't care whether that's a CNN, an LSTM, or a Transformer. This is why SHAP is the one method
in the table below that's never marked "unsupported."

**Grad-CAM** needs a convolutional layer to hook into — it works by capturing that layer's
activations and gradients during a backward pass. That means it can only run on architectures
that actually have a conv layer somewhere, and only tells you something meaningful about *that*
layer's spatial/temporal activations. No conv layer, no Grad-CAM — there's no substitute hook to
fall back on.

**Attention** visualization reads the Transformer's own self-attention weights directly. It only
exists for architectures that actually contain Transformer blocks — right now, only the hybrid
model.

**Best/worst case** and **negative example** figures are built on top of Grad-CAM (they show the
same kind of activation overlay, just on hand-picked high/low-confidence samples), so they
inherit Grad-CAM's same support restrictions.

| Model | SHAP | Grad-CAM | Attention | Grad-CAM hook / notes |
|---|:---:|:---:|:---:|---|
| `cnn_transformer` (Hybrid) | Yes | Yes | Yes | `model.cnn4` — the proposed architecture; matches thesis Methodology |
| `resnet1d` | Yes | Yes | No | `model.layer4` |
| `xresnet1d` | Yes | Yes | No | last residual block — structurally near-identical to `resnet1d` in this implementation; does not yet include the deep-stem / avgpool-shortcut improvements from the original xResNet paper |
| `xlstm` | Yes | Yes (partial) | No | `model.conv2` — explains the CNN front-end only, not what the LSTM itself attends to over time. Also needed two fixes: cuDNN refuses to run backward() on an RNN while the model is in eval() mode (worked around by disabling cuDNN for just that computation), and SHAP's default batch size caused a 20+ GiB allocation on this model specifically (fixed by chunking predictions into batches of 32) |
| `transformer` | Yes | No | No (see note) | No conv layer at all — Grad-CAM cannot apply here regardless of hook choice. Attention could in principle be extracted from this model's own self-attention layers, but `run_interpretability.py`/`make_figures.py` currently only wire up attention extraction for the hybrid model's specific block structure — extending it to the plain transformer is a small, not-yet-done addition |

**The SHAP-vs-Grad-CAM agreement score** (`run_interpretability.py`) answers a different question
than either method alone: do the two methods agree on *which ECG lead* matters most? Since
Grad-CAM's raw output is a *temporal* curve (which time points mattered, in the model's internal
feature space) and SHAP's is a *per-lead* score, the two aren't directly comparable — the script
bridges them by weighting each lead's raw signal amplitude by Grad-CAM's temporal attention,
producing a per-lead score in the same units as SHAP, then computes a Spearman rank correlation
between the two. This is run once per class per model; a low or negative correlation means the
two methods are pointing at different leads, which is worth knowing regardless of which one (if
either) is "more correct."

## Known limitations (not yet resolved)

- **No held-out test split.** Only train/val exists. Per-class decision thresholds are tuned
  and reported on the same validation data used for checkpoint selection — optimistic.
  `make_figures.py` prints this warning on every run.
- **AFIB label scarcity** under the confidence≥50 rule — see above.
- Confidence≥50 filtering, 100Hz resolution, and the real `OptimizedHybrid` architecture were
  only discovered/restored mid-project by tracing the original notebook; verify against the
  thesis text if anything here looks inconsistent with what's written there.

## Status

Active development for the 30%-new-experiment journal submission. Thesis work (on `main`) is
complete and unaffected by anything on this branch.