"""Train the repository's Keras Conv1D Slim on the exact LNN prepared shards.

This is a paired-data baseline, not a claim to reproduce the paper's private
3src/2sc split or its model-specific learned ADC thresholds.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# QKeras 0.9 uses the Keras 2 API. TensorFlow 2.16 must select its legacy API
# before TensorFlow is imported; the reference model imports `keras` directly.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_INDICES = (0, 1, 3)
FEATURES = ("x", "y", "cotB")


def reference_model(variant):
    import tensorflow as tf
    import tf_keras

    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        raise RuntimeError("Set TF_USE_LEGACY_KERAS=1 before importing TensorFlow")
    sys.modules["keras"] = tf_keras
    helpers = Path(__file__).resolve().parents[1] / "two_bit_optimization_helpers"
    sys.path.insert(0, str(helpers))
    if variant == "qkeras":
        from models.conv1d_model_quantized import QConv1D_Slim
        model = QConv1D_Slim((16, 16, 2))
    elif variant == "float":
        from models.conv1d_model_nonquantized import Conv1D_Slim
        model = Conv1D_Slim((16, 16, 2))
    else:
        raise ValueError(variant)
    if model.count_params() != 2649:
        raise RuntimeError(f"Reference Conv1D Slim should have 2649 parameters, got {model.count_params()}")
    return tf, model


def load_split(folder):
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No prepared shards in {folder}")
    images, labels = [], []
    for file in files:
        with np.load(file) as shard:
            x, y = shard["x"], shard["y"]
        if x.ndim != 4 or x.shape[1:] != (2, 16, 16) or y.shape != (len(x), 4):
            raise ValueError(f"Unexpected prepared shard shape in {file}: {x.shape}, {y.shape}")
        if x.min() < 0 or x.max() > 3 or not np.isfinite(y).all():
            raise ValueError(f"Invalid ADC codes or labels in {file}")
        images.append(x)
        labels.append(y)
    # Zenodo and the paper's generator use (time, y, x); Keras uses channels-last.
    x = np.concatenate(images).transpose(0, 2, 3, 1).astype(np.float32)
    y = np.concatenate(labels).astype(np.float32)
    return x, y


def normalization(data, mode):
    metadata_path = data / "metadata.json"
    calibration_path = data / "calibration.json"
    metadata = json.loads(metadata_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if metadata["frames"] != [0, 19] or metadata["label_names"] != calibration["label_names"]:
        raise ValueError("Reference baseline requires the calibrated two-frame prepared data")
    for split in ("train", "val", "test"):
        if metadata["counts"][split] != calibration["prepared_counts"][split]:
            raise ValueError(f"Metadata/calibration count mismatch for {split}")
    if mode == "centered":
        center = np.asarray(calibration["label_center"], np.float32)[list(LABEL_INDICES)]
        scale = np.asarray(calibration["label_scale"], np.float32)[list(LABEL_INDICES)]
        note = "Same centered target normalization as the completed LNN runs."
    elif mode == "paper_style":
        center = np.zeros(3, np.float32)
        scale = np.asarray(metadata["label_scale"], np.float32)[list(LABEL_INDICES)]
        note = ("Uncentered labels divided by the prepared data's per-file 99th-percentile "
                "scale. The original notebook derives its scale from 3src validation files; "
                "this public 3sExtended split is not identical.")
    else:
        raise ValueError(mode)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f"Invalid label scale: {scale}")
    manifest = {
        "data": str(data), "normalization": mode, "label_center": center.tolist(),
        "label_scale": scale.tolist(), "normalization_note": note,
        "source_thresholds_electrons": metadata["thresholds_electrons"],
        "frames": metadata["frames"],
        "split_counts": {k: metadata["counts"][k] for k in ("train", "val", "test")},
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    }
    return center, scale, manifest


def minimum_interval_68(residual):
    values = np.sort(np.asarray(residual))
    n = max(1, int(np.ceil(0.68 * len(values))))
    widths = values[n - 1:] - values[:len(values) - n + 1]
    i = int(np.argmin(widths))
    return float(values[i]), float(values[i + n - 1])


def assess_split(model, data, split, out, center, scale, batch_size, run_info):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for file in sorted((data / split).glob("*.npz")):
        with np.load(file) as shard:
            images, labels = shard["x"], shard["y"]
        for start in range(0, len(labels), batch_size):
            x = images[start:start + batch_size].transpose(0, 2, 3, 1).astype(np.float32)
            y = labels[start:start + batch_size].astype(np.float32)
            prediction = model(x, training=False).numpy() * scale + center
            records.append(np.column_stack((y, prediction)))
    if not records:
        raise ValueError(f"Empty {split} split")
    values = np.concatenate(records)
    df = pd.DataFrame(values, columns=("xtrue", "ytrue", "cotAtrue", "cotBtrue",
                                       "x", "y", "cotB"))
    df.to_parquet(out / "predictions.parquet", index=False)
    metrics = {"n": len(df), "split": split, "task": "slim",
               "architecture": "conv1d_reference", "variant": run_info["variant"],
               "projection": "average_charge",
               "normalization": run_info["normalization"],
               "best_epoch": run_info["best_epoch"],
               "best_val_loss": run_info["best_val_loss"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for ax, key in zip(axes, FEATURES):
        prediction = df[key].to_numpy()
        truth = df[key + "true"].to_numpy()
        if key == "cotB":
            prediction = np.degrees(np.arctan2(1.0, prediction))
            truth = np.degrees(np.arctan2(1.0, truth))
            unit = "deg"
        else:
            unit = "um"
        residual = prediction - truth
        lo, hi = minimum_interval_68(residual)
        metrics[key] = {"mean": float(np.mean(residual)),
                        "std": float(np.std(residual)),
                        "mae": float(np.mean(np.abs(residual))),
                        "interval_68": [lo, hi], "interval_68_width": hi - lo,
                        "unit": unit}
        ax.hist(residual, bins=100, histtype="step", density=True)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.15)
        ax.set(xlabel=f"{key} residual [{unit}]", ylabel="Density")
    fig.tight_layout()
    fig.savefig(out / "residuals.png", dpi=180)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def train(args):
    import tensorflow as tf

    center, scale, manifest = normalization(args.data, args.normalization)
    manifest.update(variant=args.variant, seed=args.seed, batch_size=args.batch_size,
                    learning_rate=1e-3, architecture_source=(
                        "two_bit_optimization_helpers/models/conv1d_model_quantized.py" if args.variant == "qkeras"
                        else "two_bit_optimization_helpers/models/conv1d_model_nonquantized.py"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing run manifest differs; choose a new --out directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    tf.keras.utils.set_random_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}; GPUs: {gpus}", flush=True)
    if args.require_gpu and not gpus:
        raise RuntimeError("TensorFlow sees no GPU; fix the baseline environment before using a GPU job")
    tf, model = reference_model(args.variant)

    # Exact Slim SSE form from two_bit_optimization_helpers/loss.py.
    def slim_sse(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred)) / 3.0

    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=slim_sse)
    state = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                epoch=tf.Variable(0, dtype=tf.int64),
                                best_epoch=tf.Variable(0, dtype=tf.int64),
                                best_val_loss=tf.Variable(np.inf, dtype=tf.float32))
    manager = tf.train.CheckpointManager(state, str(args.out / "latest_tf"), max_to_keep=1)
    if manager.latest_checkpoint:
        state.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resuming from completed epoch {int(state.epoch.numpy())}", flush=True)
    elif (args.out / "history.csv").exists():
        raise FileExistsError("History exists but no resumable checkpoint")

    class SaveEveryEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val = float(logs["val_loss"])
            if val < float(state.best_val_loss.numpy()):
                state.best_val_loss.assign(val)
                state.best_epoch.assign(epoch + 1)
                model.save_weights(str(args.out / "best.weights.h5"))
            state.epoch.assign(epoch + 1)
            manager.save(checkpoint_number=epoch + 1)

    if int(state.epoch.numpy()) < args.epochs:
        x_train, y_train = load_split(args.data / "train")
        x_val, y_val = load_split(args.data / "val")
        if len(y_train) != manifest["split_counts"]["train"] or len(y_val) != manifest["split_counts"]["val"]:
            raise ValueError("Prepared shard counts changed after manifest creation")
        y_train = (y_train[:, LABEL_INDICES] - center) / scale
        y_val = (y_val[:, LABEL_INDICES] - center) / scale
        model.fit(x_train, y_train, batch_size=args.batch_size,
                  validation_data=(x_val, y_val),
                  epochs=args.epochs, initial_epoch=int(state.epoch.numpy()),
                  shuffle=True, verbose=2,
                  callbacks=[SaveEveryEpoch(),
                             tf.keras.callbacks.CSVLogger(str(args.out / "history.csv"),
                                                          append=bool(int(state.epoch.numpy())))])
    best_path = args.out / "best.weights.h5"
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    model.load_weights(str(best_path))
    run_info = {"variant": args.variant, "normalization": args.normalization,
                "best_epoch": int(state.best_epoch.numpy()),
                "best_val_loss": float(state.best_val_loss.numpy())}
    (args.out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    for split in ("val", "test"):
        assess_split(model, args.data, split, args.out / split, center, scale,
                     args.batch_size, run_info)
    history = pd.read_csv(args.out / "history.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.epoch, history.loss, label="Train")
    ax.plot(history.epoch, history.val_loss, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "test" / "history.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=("qkeras", "float"), default="qkeras")
    p.add_argument("--normalization", choices=("centered", "paper_style"), default="centered")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-gpu", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
