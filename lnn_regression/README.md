# Smartpixels logic-gate regression on Della

This workflow trains **pure TorchLogix logic-gate models** for the regression task
in [Dickinson et al., *On-chip probabilistic inference for charged-particle tracking
at the sensor edge*](https://arxiv.org/abs/2602.15946v4). The public input is
[Zenodo record 18472791](https://zenodo.org/records/18472791). It uses the
`2bit_optimization` checkout of this repository as the preprocessing reference.

## What is matched

- Contained 16 × 16 clusters: boundary charge at the final frame must be below
  50 electrons. For the paper's combined Parquet files, their
  `chargeOriginal_atEdge` column is used when present.
- Two input frames, indices 0 and 19, separated by 3.8 ns; 2-bit charge levels
  0, 1, 2, 3. Default thresholds are the transformer's published means:
  247.8, 668.4, 1662.9 electrons. Change with `--thresholds` if the exact
  model-specific thresholds become available.
- Inputs remain in charge-bin units; labels are x-midplane, y-midplane,
  cotAlpha, cotBeta divided by the maximum per-file 99th percentile absolute
  values from training data. No input standardization or log transform.
- 80/20 train/validation split from the extended angular sample and a separate
  physical-pT `2s` test sample. Adam with Nesterov momentum (`torch.optim.NAdam`),
  learning rate 0.001, batch size 5000, 1000 epochs, and best hard-LUT validation
  checkpoint.
- Slim predicts x, y, cotBeta with MSE. Full predicts four means and diagonal
  standard deviations with Gaussian NLL. Max predicts four means and a full
  lower-triangular Gaussian scale with multivariate NLL.

The public Zenodo archive calls the extended sample `3sExtended`; the paper
checkout calls its prepared training sample `3sr`/`3src`. The README in the
paper checkout does not document the exact transformation between these names.
The public workflow uses `3sExtended` and records that choice. If the paper's
combined `part.*.parquet` files are supplied, the converter accepts them too.
The seeded event split may differ from the paper's published split, so exact
numerical reproduction of its figure is not claimed.

## Why the model differs from Keras

The three network layouts follow the paper's MLP, projected Conv1D, and spatial
Conv2D variants. The learned layers are `LogicDense` and `LogicConv2d` gates
from TorchLogix. A 2-bit ADC value is encoded as three cumulative comparator
bits. Projection and spatial pooling use Boolean OR, the logic equivalent of
max pooling; the paper uses charge averages. Binary fan-in requires extra gate
reduction stages before the 16-node blocks. The output uses binary gate counts
(`GroupSum`) and a fixed numeric mapping for means, widths, and covariance.
The loss and reporting arithmetic are used during training and evaluation; there
are no learned floating-point layers in the deployed model. Hard LUTs are used
for validation and test predictions. This is an LNN comparison with the paper,
not a reproduction of its Keras weights, QKeras quantization, or HLS area/latency.

## Start to finish on Della

Run from a shell on Della. The paths below keep downloads, extracted files,
prepared shards, environments, and outputs in your requested scratch directory.
The two required archives are about 15 GB compressed; leave room for extraction.

```bash
ssh mb7126@della-gpu.princeton.edu
git clone git@github.com:mbileska/smart-pixels-ml.git ~/smart-pixels-ml
cd ~/smart-pixels-ml
git switch wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
mkdir -p "$DATA/archives" "$DATA/runs" "$DATA/mplconfig"
cd "$DATA/archives"
wget -c 'https://zenodo.org/records/18472791/files/dataset_3sExtended_16x16_50x12P5x100.zip?download=1' -O dataset_3sExtended_16x16_50x12P5x100.zip
wget -c 'https://zenodo.org/records/18472791/files/dataset_2s_16x16_50x12P5x100.zip?download=1' -O dataset_2s_16x16_50x12P5x100.zip
echo 'b36afb2b4122b1367f2df19d3f0038e9  dataset_3sExtended_16x16_50x12P5x100.zip' | md5sum -c -
echo 'e94011c8322e8ee6444c06a9f8ef20e3  dataset_2s_16x16_50x12P5x100.zip' | md5sum -c -
cd ~/smart-pixels-ml
module load anaconda3/2026.7
conda create --prefix "$DATA/env" python=3.12 pip -y
conda activate "$DATA/env"
python -m pip install torch numpy pandas pyarrow matplotlib
git clone git@github.com:ligerlac/torchlogix.git "$DATA/torchlogix"
git -C "$DATA/torchlogix" checkout abdf71681a3d2b91ea05c145aa8a0056f68d87f5
python -m pip install --no-deps -e "$DATA/torchlogix"
python -c 'import torch, torchlogix; print(torch.__version__, torch.cuda.is_available())'
PREP_JOB=$(sbatch --parsable lnn_regression/della_prepare.sbatch)
echo "Preparation job: $PREP_JOB"
```

If `~/smart-pixels-ml` is already cloned, use `cd ~/smart-pixels-ml`,
`git fetch origin`, and `git switch wip-mila` in place of the clone command.

The current TorchLogix `main` checkout uses Python 3.12 syntax. Submit the
preparation job from the repository root after the checksums pass. Wait for it
to finish (`squeue -u "$USER"`). It creates
`$DATA/prepared/2frame_2bit/metadata.json` and `train`, `val`, `test` shards.
Review the event counts and label scale in that metadata before training.

Run an initial MLP Slim model. If you started a new shell, set `PREP_JOB` to the
preparation job ID printed above. The dependency makes training wait for
successful preparation.

```bash
cd ~/smart-pixels-ml
sbatch --dependency=afterok:"$PREP_JOB" --export=ALL,ARCH=mlp,TASK=slim lnn_regression/della_train.sbatch
```

The training job saves `latest.pt` every epoch, selects `best.pt` by validation
loss, then writes `test/metrics.json`, `test/predictions.parquet`,
`test/residuals.png`, `test/history.png`, and for Full/Max `test/pulls.png`.
The job requests one GPU with `--gres=gpu:1`; Della assigns the public GPU
partition, so the script does not name a partition.
If the 72-hour job ends before 1000 epochs, submit the same command again; it
automatically resumes. Once the job finishes successfully, train the paper's
seven architecture/output combinations:

```bash
cd ~/smart-pixels-ml
for spec in mlp:slim mlp:full conv1d:slim conv1d:full conv2d:slim conv2d:full conv2d:max; do
  ARCH=${spec%:*}
  TASK=${spec#*:}
  sbatch --dependency=afterok:"$PREP_JOB" --export=ALL,ARCH="$ARCH",TASK="$TASK" lnn_regression/della_train.sbatch
done
```

That loop includes MLP Slim again; its existing completed run resumes without
retraining and evaluates again. GPU limits and queue policy may favor submitting
fewer jobs at a time. For an optional 20-frame, 2-bit study, run the converter
with `--frames 20 --out "$DATA/prepared/20frame_2bit"`, then point `--data` at
that directory in a direct `python -m lnn_regression.run train ...` invocation.
This is *not* the paper's 20-frame electron-precision study.

After the runs finish, make a paper-style residual summary:

```bash
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
python -m lnn_regression.compare \
  --metrics "$DATA"/runs/*_*/test/metrics.json \
  --out "$DATA/runs/comparison.png"
```

Each `metrics.json` reports mean, standard deviation, mean absolute residual,
and minimum-width 68% residual interval for x and y (microns), and alpha and
beta (degrees). Full/Max also report mean predicted one-sigma uncertainty,
pull mean/std, and one-sigma coverage. Residuals use the paper convention
`prediction - truth`. Compare these test results with the paper's Figure 4,
allowing for the model and split differences above.

## Local smoke check

For a small prepared dataset, use `python -m lnn_regression.run train --data
PATH --out PATH --architecture mlp --task slim --epochs 2 --batch-size 16`.
This repo does not contain the 15 GB of public input archives, so full physical
metrics are produced only after the Della jobs run.
