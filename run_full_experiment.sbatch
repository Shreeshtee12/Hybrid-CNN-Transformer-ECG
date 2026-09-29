#!/bin/bash
#SBATCH --job-name=ecg_full_run
#SBATCH --account=PLS0150
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

REPO_DIR=/users/PLS0150/shreeshtee/Hybrid-CNN-Transformer-ECG
DATA_DIR=/users/PLS0150/shreeshtee/PTBXL-Dataset-Thesiswork/physionet.org/files/ptb-xl/1.0.3

cd "$REPO_DIR/src"
source "$REPO_DIR/.venv/bin/activate"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python run_all_models.py \
  --signal-path "$DATA_DIR" \
  --csv-path "$DATA_DIR/ptbxl_database.csv" \
  --max-epochs 100 \
  --figures \
  --wandb --project ptbxl-ecg

echo "Job finished: $(date)"
