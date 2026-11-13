#!/bin/bash
# TensorFlow 2.16 pip CUDA wheels place libraries under site-packages/nvidia.
# Della may also provide CUDA through modules; put the environment's matching
# pip libraries first so TensorFlow can resolve the versions it was built for.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the baseline Conda environment first" >&2
  return 1
fi
export TF_USE_LEGACY_KERAS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/gpfs/IOJALVO/mb7126/lnn_regression/mplconfig}"
mkdir -p "$MPLCONFIGDIR"
cuda_lib_dirs=0
for lib_dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cuda_lib_dirs=$((cuda_lib_dirs + 1))
  fi
done
for ptxas in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cuda_nvcc/bin/ptxas; do
  if [[ -x "$ptxas" ]]; then
    export PATH="${ptxas%/*}:$PATH"
    break
  fi
done
echo "CUDA library directories from baseline environment: $cuda_lib_dirs"
if [[ "$cuda_lib_dirs" == 0 ]]; then
  echo "No pip NVIDIA libraries found; check tensorflow[and-cuda] installation" >&2
fi
