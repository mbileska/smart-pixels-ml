#!/bin/bash
#SBATCH --job-name=conv1d-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
module load anaconda3/2026.7
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
conda activate "$DATA/baseline_env"
cd "$SLURM_SUBMIT_DIR"
export TF_USE_LEGACY_KERAS=1
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
VARIANT=${VARIANT:-qkeras}
LABEL_NORM=${LABEL_NORM:-centered}
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-5000}
RUN="$DATA/runs_centered/reference_conv1d_slim_${VARIANT}_${LABEL_NORM}"
python -m lnn_regression.reference_baseline \
  --data "$DATA/prepared/2frame_2bit" --out "$RUN" \
  --variant "$VARIANT" --normalization "$LABEL_NORM" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --require-gpu
