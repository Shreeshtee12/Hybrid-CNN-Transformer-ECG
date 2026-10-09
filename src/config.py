"""
config.py
=========
SINGLE SOURCE OF TRUTH for the whole pipeline.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root, no matter where script runs from
LIGHTNING_LOGS_DIR = BASE_DIR / "lightning_logs"

SEED = 42

CLASSES = ["NORM", "AFIB", "PVC", "LVH", "IMI", "ASMI", "LAFB", "IRBBB"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Label rule (ONE place, used by data_split / train / eval / figures / interpretability).
# PTB-XL stores RHYTHM statements (AFIB, PVC) with confidence 0, because they are not
# diagnostic statements -- a >=50 cutoff silently drops ~97% of AFIB (1514 -> 48 records).
# Rhythm classes therefore count at ANY confidence (as in the Strodthoff benchmark code);
# diagnostic classes keep the >=50 cutoff.
RHYTHM_CLASSES = ["AFIB", "PVC"]
DIAGNOSTIC_CONF_THRESHOLD = 50


def filter_codes(codes):
    """scp_codes dict -> list of target classes present under the rule above."""
    return [k for k, conf in codes.items()
            if k in CLASSES and (k in RHYTHM_CLASSES or conf >= DIAGNOSTIC_CONF_THRESHOLD)]


DATASET_PATH = Path(
    "/users/PLS0150/shreeshtee/PTBXL-Dataset-Thesiswork/physionet.org/files/ptb-xl/1.0.3"
)
SPLIT_DIR = BASE_DIR / "splits"
SPLIT_FILE = SPLIT_DIR / "split_strat_fold.json"   # official PTB-XL folds: 1-8 train / 9 val / 10 test

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

for d in [SPLIT_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64

MODEL_NAMES = [
    "cnn1d",
    "resnet1d",
    "xresnet1d",
    "transformer",
    "xlstm",
    "cnn_transformer",
]
