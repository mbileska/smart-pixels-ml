#!/bin/bash
#SBATCH --job-name=lnn-diagnose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
module load anaconda3/2026.7
conda activate /scratch/gpfs/IOJALVO/mb7126/lnn_regression/env
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR=/scratch/gpfs/IOJALVO/mb7126/lnn_regression/mplconfig
mkdir -p "$MPLCONFIGDIR"
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
python -m lnn_regression.diagnose \
  --data "$DATA/prepared/2frame_2bit" \
  --run "$DATA/runs_centered/mlp_slim"
