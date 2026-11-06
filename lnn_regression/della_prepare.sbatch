#!/bin/bash
#SBATCH --job-name=lnn-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
module load anaconda3/2026.7
conda activate /scratch/gpfs/IOJALVO/mb7126/lnn_regression/env
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR=/scratch/gpfs/IOJALVO/mb7126/lnn_regression/mplconfig
mkdir -p "$MPLCONFIGDIR"
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
mkdir -p "$DATA/raw/dataset_3sExtended_16x16_50x12P5x100" "$DATA/raw/dataset_2s_16x16_50x12P5x100"
unzip -n -q "$DATA/archives/dataset_3sExtended_16x16_50x12P5x100.zip" -d "$DATA/raw/dataset_3sExtended_16x16_50x12P5x100"
unzip -n -q "$DATA/archives/dataset_2s_16x16_50x12P5x100.zip" -d "$DATA/raw/dataset_2s_16x16_50x12P5x100"
python -m lnn_regression.data \
  --train-source "$DATA/raw/dataset_3sExtended_16x16_50x12P5x100" \
  --test-source "$DATA/raw/dataset_2s_16x16_50x12P5x100" \
  --out "$DATA/prepared/2frame_2bit"
python -m lnn_regression.calibrate --data "$DATA/prepared/2frame_2bit" \
  --target-span 1 --margin 0.1 --readout-bits 64
