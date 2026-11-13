#!/bin/bash
#SBATCH --job-name=conv1d-gpu-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
set -euo pipefail
module load anaconda3/2026.7
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
conda activate "$DATA/baseline_env"
cd "$SLURM_SUBMIT_DIR"
source lnn_regression/reference_gpu_env.sh
nvidia-smi -L
python - <<'PY'
import tensorflow as tf
import tf_keras
import qkeras
print('TensorFlow:', tf.__version__)
print('tf-keras:', tf_keras.__version__)
print('QKeras:', qkeras.__version__)
devices = tf.config.list_physical_devices('GPU')
print('TensorFlow GPUs:', devices)
if not devices:
    raise SystemExit('TensorFlow GPU preflight failed')
PY
