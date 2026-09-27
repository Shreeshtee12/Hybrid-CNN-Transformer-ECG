"""
config.py
=========
SINGLE SOURCE OF TRUTH for the whole pipeline.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root, no matter where script runs from
LIGHTNING_LOGS_DIR = BASE_DIR / "lightning_logs"

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------
# Class definition + FIXED numbering
# ---------------------------------------------------------------------
CLASSES = ["NORM", "AFIB", "PVC", "LVH", "IMI", "ASMI", "LAFB", "IRBBB"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATASET_PATH = Path(
    "/users/PLS0150/shreeshtee/PTBXL-Dataset-Thesiswork/physionet.org/files/ptb-xl/1.0.3"
)
SPLIT_DIR = BASE_DIR / "splits"
SPLIT_FILE = SPLIT_DIR / f"split_seed{SEED}.json"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

for d in [SPLIT_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Split ratio (patient-grouped) -- matches train_lightning.py / eval.py exactly
# ---------------------------------------------------------------------
VAL_FRAC = 0.20   # single 80/20 train/val split -- this repo has no separate test set

# ---------------------------------------------------------------------
# Evaluation (identical for every model)
# ---------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64

# ---------------------------------------------------------------------
# The model registry -- add/remove architectures here ONLY.
# ---------------------------------------------------------------------
MODEL_NAMES = [
    "resnet1d",
    "xresnet1d",
    "transformer",
    "xlstm",
    "cnn_transformer",
]