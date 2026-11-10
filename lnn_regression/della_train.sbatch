#!/bin/bash
#SBATCH --job-name=lnn-regress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
module load anaconda3/2026.7
conda activate /scratch/gpfs/IOJALVO/mb7126/lnn_regression/env
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR=/scratch/gpfs/IOJALVO/mb7126/lnn_regression/mplconfig
mkdir -p "$MPLCONFIGDIR"
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
ARCH=${ARCH:-mlp}
TASK=${TASK:-slim}
PROJECTION=${PROJECTION:-or}
READOUT_BITS=${READOUT_BITS:-64}
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-5000}
RUN="$DATA/runs_centered/${ARCH}_${TASK}"
if [[ "$PROJECTION" != or ]]; then RUN="${RUN}_${PROJECTION}"; fi
if [[ "$READOUT_BITS" != 64 ]]; then RUN="${RUN}_g${READOUT_BITS}"; fi
RESUME=()
if [[ -f "$RUN/latest.pt" ]]; then RESUME=(--resume); fi
python -m lnn_regression.run train --data "$DATA/prepared/2frame_2bit" \
  --out "$RUN" --architecture "$ARCH" --task "$TASK" --projection "$PROJECTION" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --readout-bits "$READOUT_BITS" "${RESUME[@]}"
python -m lnn_regression.run evaluate --data "$DATA/prepared/2frame_2bit" \
  --checkpoint "$RUN/best.pt" --out "$RUN/test" --split test
python -m lnn_regression.export --checkpoint "$RUN/best.pt" --out "$RUN/export"
