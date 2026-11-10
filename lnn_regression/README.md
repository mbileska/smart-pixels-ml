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
- Inputs remain in charge-bin units. A separate calibration pass subtracts the
  **training mean** from each label and divides by its maximum absolute
  training deviation, putting every training label within [-1, 1]. The
  converter also records the reference checkout's per-file 99th-percentile
  scale in `metadata.json`; this LNN training uses `calibration.json` instead.
  No input standardization or log transform is applied.
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
bits. The default projection and spatial pooling use Boolean OR, the logic
equivalent of max pooling; the paper uses charge averages. Optional MLP
projections preserve charge sums or combine OR with charge thresholds. Binary
fan-in requires extra gate reduction stages before the 16-node blocks. Each
output feature has an independent `GroupSum(k=1)` over its binary gates. Each
mean head receives its own `tau` and `beta`, fitted from the training feature's
centered range with
10% headroom on both ends and at least 10% beyond [-1, 1]. The output is
`(count + beta) / tau`, with `tau = readout_bits / (upper - lower)` and
`beta = lower * tau`. Separate positive intervals are used for uncertainty
heads and signed intervals for Max covariance heads. These values are stored
directly in the TorchLogix circuit representation.
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
to finish and confirm `COMPLETED` with exit code `0:0` using
`sacct -j "$PREP_JOB" --format=JobID,State,ExitCode`. It creates
`$DATA/prepared/2frame_2bit/metadata.json` and `train`, `val`, `test` shards.
Review event counts in that metadata and the newly produced
`calibration.json` and `label_normalization.png`. If preparation was already
completed before this update, run calibration directly on the existing shards;
there is no need to download or convert them again:

```bash
cd ~/smart-pixels-ml
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
python -m lnn_regression.calibrate --data "$DATA/prepared/2frame_2bit" \
  --target-span 1 --margin 0.1 --readout-bits 64
```

The command prints the fitted range, `tau`, `beta`, and original-unit step for
each mean output, plus counts of validation/test labels outside the available
range. Training ranges use only training labels; if either held-out split has
out-of-range values, increase `--margin`, inspect the new plot and calibration
report, and retrain from a fresh output directory. A finite training sample
cannot prove a bound for every possible future event.

Run an initial MLP Slim model after preparation succeeds:

```bash
cd ~/smart-pixels-ml
sbatch --export=ALL,ARCH=mlp,TASK=slim lnn_regression/della_train.sbatch
```

For the seven **pending** jobs submitted before this calibration change, cancel
only those job IDs, then pull and calibrate before submitting replacements:

```bash
scancel 16910982 16910983 16910984 16910985 16910986 16910987 16910988
squeue -u "$USER" -n lnn-regress
cd ~/smart-pixels-ml
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
python -m lnn_regression.calibrate --data "$DATA/prepared/2frame_2bit" \
  --target-span 1 --margin 0.1 --readout-bits 64
```

The training job writes to `$DATA/runs_centered/`, keeping these calibrated
runs separate from the earlier uncentered jobs. It saves `latest.pt` every
epoch, selects `best.pt` by validation
loss, then writes `test/metrics.json`, `test/predictions.parquet`,
`test/residuals.png`, `test/history.png`, and for Full/Max `test/pulls.png`. It
also exports the binary-input gate core to `export/circuit.json` and
`export/circuit.c`, with per-feature readout settings in `export/readout.json`.
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
  sbatch --export=ALL,ARCH="$ARCH",TASK="$TASK" lnn_regression/della_train.sbatch
done
```

That loop includes MLP Slim again; its existing completed run resumes without
retraining and evaluates again. GPU limits and queue policy may favor submitting
fewer jobs at a time. For an optional 20-frame, 2-bit study, run the converter
with `--frames 20 --out "$DATA/prepared/20frame_2bit"`, then point `--data` at
that directory in a direct `python -m lnn_regression.run train ...` invocation.
This is *not* the paper's 20-frame electron-precision study.

### Earlier checkpoints

Earlier checkpoints can still be evaluated and exported using their original
zero-offset scale and readout. They cannot be resumed into the new centered
calibration, so new jobs use `runs_centered`. To export earlier completed runs:

```bash
cd ~/smart-pixels-ml
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
for run in "$DATA"/runs/*_*; do
  [[ -f "$run/best.pt" ]] || continue
  python -m lnn_regression.export --checkpoint "$run/best.pt" --out "$run/export"
done
```

The C and JSON circuit exports retain `tau` and `beta`. The pinned TorchLogix
Verilog writer currently emits raw reduction counts for nonconstant sums, so
use the C/JSON artifacts for calibrated regression outputs until that writer
supports scaled reductions.

After the runs finish, make a paper-style residual summary:

```bash
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
python -m lnn_regression.compare \
  --metrics "$DATA"/runs_centered/*_*/test/metrics.json \
  --out "$DATA/runs_centered/comparison.png"
```

Each `metrics.json` reports mean, standard deviation, mean absolute residual,
and minimum-width 68% residual interval for x and y (microns), and alpha and
beta (degrees). Full/Max also report mean predicted one-sigma uncertainty,
pull mean/std, and one-sigma coverage. Residuals use the paper convention
`prediction - truth`. Compare these test results with the paper's Figure 4,
allowing for the model and split differences above.

### Diagnose a completed Slim run

The diagnostic evaluates `best.pt` on the validation split, then compares
validation and test truth distributions, predictions, residuals, and GroupSum
gate counts. This helps distinguish a train-to-test distribution shift from
readout saturation or predictions collapsing to a few count values. It does
not retrain the network. Submit from the `smart-pixels-ml` repository root:

```bash
sbatch lnn_regression/della_diagnose.sbatch
```

The CPU job writes `$DATA/runs_centered/mlp_slim/val/{metrics.json,residuals.png}`
and `$DATA/runs_centered/mlp_slim/diagnostics/{diagnostics.json,val_test_diagnostics.png}`.
`diagnostics.json` includes 68% residual intervals, bias, R², a constant
training-mean baseline, and the fraction of predictions at each GroupSum limit.
The plot's four columns show truth/prediction distributions, residuals,
prediction versus truth, and output count use for x, y, and beta.

### Rerun MLP Slim with charge-preserving projections

The original LNN MLP used OR across each row/column of ADC comparator bits.
That retains whether any pixel crosses each threshold but discards how many
pixels do. `--projection charge_sum` instead uses fixed Boolean adders to sum
the 16 two-bit ADC codes along each axis and frame. The six output bits encode
the exact integer sum, equivalent to the paper's average pooling up to its
fixed factor of 1/16. The learned layers remain TorchLogix logic gates, and
the whole binary-input model lowers to a TorchLogix `Circuit` and C/JSON.
This uses more gates in the fixed projection, so hardware area must be assessed
after physics performance is known. `export/readout.json` records the circuit
gate count for each run; this is a logic count, not an ASIC area measurement.

From any Della shell, using the `mbileska/smart-pixels-ml` checkout:

```bash
cd ~/smart-pixels-ml
git switch wip-mila
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
module load anaconda3/2026.7
conda activate "$DATA/env"
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
python -m unittest lnn_regression.test_charge_projection
JOB=$(sbatch --parsable --time=08:00:00 \
  --export=ALL,ARCH=mlp,TASK=slim,PROJECTION=charge_sum \
  lnn_regression/della_train.sbatch)
echo "Charge-sum MLP Slim job: $JOB"
squeue -j "$JOB"
```

The new run is `$DATA/runs_centered/mlp_slim_charge_sum`; the old OR run stays
at `$DATA/runs_centered/mlp_slim`. If the eight-hour allocation ends before
epoch 1000, submit the same `sbatch` command again to resume. When training,
test evaluation, and export finish, compare the two runs:

```bash
RUN="$DATA/runs_centered/mlp_slim_charge_sum"
cat "$RUN/test/metrics.json"
python -m lnn_regression.compare \
  --metrics "$DATA/runs_centered/mlp_slim/test/metrics.json" \
            "$RUN/test/metrics.json" \
  --out "$DATA/runs_centered/or_vs_charge_sum.png"
DIAG=$(sbatch --parsable --export=ALL,RUN_NAME=mlp_slim_charge_sum \
  lnn_regression/della_diagnose.sbatch)
echo "Diagnostic job: $DIAG"
```

### Test OR plus ordered charge thresholds for MLP Slim

The charge-sum run improved the Slim beta test MAE from 16.84° to 12.64°, but
worsened the x/y 68% residual widths from 60.95/11.51 µm to 77.39/28.87 µm.
Its x/y validation widths also worsened. None of the three GroupSum counts
reached 0 or 64, so the next test targets input encoding rather than readout
range. The binary charge sum may be difficult for the small learned gate block
to decode; this is a hypothesis to test.

`--projection or_charge_thresholds` retains the three ADC comparator OR signals
per axis, coordinate, and frame, and adds five ordered signals for charge-code
sums reaching 2, 4, 8, 16, or 32. A sum can range from 0 to 48. These signals
are computed by fixed Boolean gates from the same input, preserving the pure
logic circuit. They add gates, so compare circuit gate counts in
`export/readout.json` and do not treat them as an ASIC area estimate.

On Della, from any shell:

```bash
cd ~/smart-pixels-ml
git switch wip-mila
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
module load anaconda3/2026.7
conda activate "$DATA/env"
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
python -m unittest lnn_regression.test_charge_projection
JOB=$(sbatch --parsable --time=08:00:00 \
  --export=ALL,ARCH=mlp,TASK=slim,PROJECTION=or_charge_thresholds \
  lnn_regression/della_train.sbatch)
echo "OR plus charge thresholds MLP Slim job: $JOB"
squeue -j "$JOB"
```

The new run is `$DATA/runs_centered/mlp_slim_or_charge_thresholds`. If the
job hits the eight-hour limit before epoch 1000, repeat the `sbatch` command
to resume from `latest.pt`. Once it finishes successfully, run these commands
from any Della shell to compare all three projections and generate diagnostics:

```bash
cd ~/smart-pixels-ml
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
module load anaconda3/2026.7
conda activate "$DATA/env"
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
RUN="$DATA/runs_centered/mlp_slim_or_charge_thresholds"
cat "$RUN/test/metrics.json"
cat "$RUN/export/readout.json"
python -m lnn_regression.compare \
  --metrics "$DATA/runs_centered/mlp_slim/test/metrics.json" \
            "$DATA/runs_centered/mlp_slim_charge_sum/test/metrics.json" \
            "$RUN/test/metrics.json" \
  --out "$DATA/runs_centered/slim_projection_comparison.png"
DIAG=$(sbatch --parsable --export=ALL,RUN_NAME=mlp_slim_or_charge_thresholds \
  lnn_regression/della_diagnose.sbatch)
echo "Diagnostic job: $DIAG"
```

### Paper-style Conv1D Slim pilot

The MLP projection comparison motivates a spatial model. This Conv1D LNN
uses the paper's two axis projections: sum the 2-bit charge codes across the
16 pixels of each row or column, separately at each of the two time frames.
The integer sum (0–48) contains the same information as the paper's average
charge (sum/16). Six Boolean bits retain the exact sum. Each axis branch then
uses five TorchLogix convolutional kernels with a three-position window,
followed by two 16-output logic blocks and three 64-gate GroupSum readouts.
The output features are x, y, and cot(beta), with the same calibration and
splits as the completed MLP runs. This is a structural analogue of the
paper's Conv1D Slim; Boolean LUT gates replace its tanh and dense arithmetic.

The compact circuit input is 384 bits: two axes × two frames × 16 positions ×
six sum bits. `export/gates.v` and `export/compact_top.sv` implement the learned
gate network and three unsigned count outputs (7 bits each for 64 gates).
The upstream pixel-to-axis charge-sum projectors, including their adders, are
outside this compact RTL, as are the final per-feature tau/beta and physical
unit conversions. These costs must be included in any eFPGA comparison. The
384-bit interface fits the stated 632-input limit, but fitting logic RBBs and
meeting timing still requires synthesis and place-and-route on the target.
The C/JSON circuit retains calibrated GroupSum reductions; the count-only RTL
uses the **original** `group_sum_tau_beta` in `export/readout.json` when
converting counts to normalized predictions.

From any Della shell:

```bash
cd ~/smart-pixels-ml
git switch wip-mila
git pull --ff-only origin wip-mila
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
module load anaconda3/2026.7
conda activate "$DATA/env"
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
python -m unittest lnn_regression.test_conv1d_compact lnn_regression.test_charge_projection
JOB=$(sbatch --parsable --time=12:00:00 \
  --export=ALL,ARCH=conv1d,TASK=slim,PROJECTION=charge_sum \
  lnn_regression/della_train.sbatch)
echo "Conv1D Slim pilot job: $JOB"
squeue -j "$JOB"
```

This writes to `$DATA/runs_centered/conv1d_slim_charge_sum`. If the job reaches
the time limit before epoch 1000, submit the same `sbatch` command to resume
from `latest.pt`. After the job completes:

```bash
cd ~/smart-pixels-ml
DATA=/scratch/gpfs/IOJALVO/mb7126/lnn_regression
module load anaconda3/2026.7
conda activate "$DATA/env"
export MPLCONFIGDIR="$DATA/mplconfig"
mkdir -p "$MPLCONFIGDIR"
RUN="$DATA/runs_centered/conv1d_slim_charge_sum"
cat "$RUN/test/metrics.json"
cat "$RUN/export/readout.json"
python -m lnn_regression.compare \
  --metrics "$DATA/runs_centered/mlp_slim/test/metrics.json" \
            "$DATA/runs_centered/mlp_slim_charge_sum/test/metrics.json" \
            "$DATA/runs_centered/mlp_slim_or_charge_thresholds/test/metrics.json" \
            "$RUN/test/metrics.json" \
  --out "$DATA/runs_centered/slim_conv1d_pilot_comparison.png"
DIAG=$(sbatch --parsable --export=ALL,RUN_NAME=conv1d_slim_charge_sum \
  lnn_regression/della_diagnose.sbatch)
echo "Conv1D diagnostic job: $DIAG"
```

The training script accepts `READOUT_BITS`, `EPOCHS`, and `BATCH_SIZE` through
the Slurm environment for later width studies. Non-default readout widths get
their own run suffix (`_g96`, for example). First measure this 64-gate pilot's
physics and hardware size before expanding the width sweep.

## Local smoke check

For a small prepared dataset, first run `python -m lnn_regression.calibrate
--data PATH`, then use `python -m lnn_regression.run train --data PATH --out
PATH --architecture mlp --task slim --epochs 2 --batch-size 16`.
This repo does not contain the 15 GB of public input archives, so full physical
metrics are produced only after the Della jobs run.
