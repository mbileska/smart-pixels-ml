"""Train and evaluate pure logic-gate Smartpixels regressors."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import LogicRegressor, regression_loss


def split_batches(folder, batch_size, shuffle=False, seed=42):
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz shards in {folder}")
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(files)
    x_parts, y_parts, buffered = [], [], 0
    for file in files:
        with np.load(file) as data:
            x, y = data["x"], data["y"]
        order = rng.permutation(len(y)) if shuffle else np.arange(len(y))
        x_parts.append(x[order])
        y_parts.append(y[order])
        buffered += len(y)
        if buffered >= batch_size:
            all_x, all_y = np.concatenate(x_parts), np.concatenate(y_parts)
            for start in range(0, len(all_y) - batch_size + 1, batch_size):
                yield all_x[start:start + batch_size], all_y[start:start + batch_size]
            tail = len(all_y) % batch_size
            x_parts, y_parts = ([all_x[-tail:]], [all_y[-tail:]]) if tail else ([], [])
            buffered = tail
    if buffered:
        yield np.concatenate(x_parts), np.concatenate(y_parts)


def model_batch(model, x, y, center, scale, task, device):
    x = torch.from_numpy(x).to(device)
    y = (torch.from_numpy(y).to(device) - center) / scale
    return regression_loss(model(x), y, task)


def evaluate_loss(model, folder, batch_size, center, scale, task, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in split_batches(folder, batch_size):
            loss = model_batch(model, x, y, center, scale, task, device)
            total += loss.item() * len(y)
            count += len(y)
    return total / count


def train(args):
    data = Path(args.data)
    metadata = json.loads((data / "metadata.json").read_text())
    calibration_path = data / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(f"{calibration_path}; first run python -m lnn_regression.calibrate --data {data}")
    calibration = json.loads(calibration_path.read_text())
    if calibration["label_names"] != metadata["label_names"]:
        raise ValueError("Calibration and prepared data disagree on labels")
    if calibration["prepared_counts"] != {name: metadata["counts"][name]
                                           for name in ("train", "val", "test")}:
        raise ValueError("Calibration and prepared data disagree on split counts")
    outside = {name: calibration["split_stats"][name]["outside_readout_count"]
               for name in ("train", "val", "test")}
    if any(any(counts) for counts in outside.values()):
        raise ValueError(f"Some labels cannot be represented by GroupSum: {outside}; "
                         "rerun calibration with a larger --margin before training")
    frames = len(metadata["frames"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = LogicRegressor(args.architecture, args.task, frames, args.readout_bits,
                           str(device), calibration["readout_bounds"], args.projection).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    center = torch.tensor(calibration["label_center"], dtype=torch.float32, device=device)
    scale = torch.tensor(calibration["label_scale"], dtype=torch.float32, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out / "history.csv"
    best = math.inf
    start_epoch = 1
    if args.resume:
        latest_path = args.out / "latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        if (latest["architecture"], latest["task"], latest["frames"], latest["readout_bits"],
                latest.get("projection", "or"), latest.get("calibration")) != (
                    args.architecture, args.task, frames, args.readout_bits,
                    args.projection, calibration):
            raise ValueError("Resume configuration does not match latest checkpoint")
        model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        best = latest["best_val_loss"]
        start_epoch = latest["epoch"] + 1
    elif history_path.exists():
        raise FileExistsError(f"{history_path} exists; use --resume or a new --out")
    with history_path.open("a" if args.resume else "w", newline="") as history:
        writer = csv.writer(history)
        if not args.resume:
            writer.writerow(("epoch", "train_loss", "val_loss"))
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for x, y in split_batches(data / "train", args.batch_size, shuffle=True, seed=args.seed + epoch):
                optimizer.zero_grad(set_to_none=True)
                loss = model_batch(model, x, y, center, scale, args.task, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
            val_loss = evaluate_loss(model, data / "val", args.batch_size, center,
                                     scale, args.task, device)
            train_loss = total / count
            writer.writerow((epoch, train_loss, val_loss))
            history.flush()
            print(f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "architecture": args.architecture,
                            "task": args.task, "frames": frames, "readout_bits": args.readout_bits,
                            "label_center": calibration["label_center"],
                            "label_scale": calibration["label_scale"],
                            "calibration": calibration, "projection": args.projection,
                            "seed": args.seed,
                            "best_epoch": epoch, "best_val_loss": best,
                            "data_metadata": metadata}, args.out / "best.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "architecture": args.architecture, "task": args.task, "frames": frames,
                        "readout_bits": args.readout_bits, "label_center": calibration["label_center"],
                        "label_scale": calibration["label_scale"], "calibration": calibration,
                        "projection": args.projection,
                        "epoch": epoch, "best_val_loss": best}, args.out / "latest.pt")


def minimum_interval_68(residual):
    values = np.sort(np.asarray(residual))
    n = max(1, int(np.ceil(0.68 * len(values))))
    widths = values[n - 1:] - values[:len(values) - n + 1]
    i = int(np.argmin(widths))
    return [float(values[i]), float(values[i + n - 1])]


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    metadata = json.loads((Path(args.data) / "metadata.json").read_text())
    calibration = checkpoint.get("calibration")
    if len(metadata["frames"]) != checkpoint["frames"]:
        raise ValueError("Checkpoint and prepared data disagree on time frames")
    if calibration is not None:
        current_calibration = json.loads((Path(args.data) / "calibration.json").read_text())
        if current_calibration != calibration:
            raise ValueError("Checkpoint and prepared data disagree on calibration")
    elif metadata["label_scale"] != checkpoint["label_scale"]:
        raise ValueError("Legacy checkpoint and prepared data disagree on scale")
    device = torch.device(args.device)
    model = LogicRegressor(checkpoint["architecture"], checkpoint["task"],
                           checkpoint["frames"], checkpoint["readout_bits"], str(device),
                           calibration["readout_bounds"] if calibration else None,
                           checkpoint.get("projection", "or")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # TorchLogix selects hard LUTs in eval mode.
    scale = np.asarray(checkpoint["label_scale"], dtype=np.float32)
    center = np.asarray(checkpoint.get("label_center", [0.0] * 4), dtype=np.float32)
    rows = []
    with torch.no_grad():
        for x, y in split_batches(Path(args.data) / args.split, args.batch_size):
            output = model(torch.from_numpy(x).to(device))
            mean = output["mean"].cpu().numpy()
            sigma = output.get("sigma")
            sigma = sigma.cpu().numpy() if sigma is not None else None
            names = (0, 1, 3) if checkpoint["task"] == "slim" else (0, 1, 2, 3)
            for i in range(len(y)):
                record = {"xtrue": float(y[i, 0]), "ytrue": float(y[i, 1]),
                          "cotAtrue": float(y[i, 2]), "cotBtrue": float(y[i, 3])}
                for j, k in enumerate(names):
                    key = ("x", "y", "cotA", "cotB")[k]
                    record[key] = float(mean[i, j] * scale[k] + center[k])
                    if sigma is not None:
                        record["sigma" + key] = float(sigma[i, j] * scale[k])
                rows.append(record)
    if not rows:
        raise ValueError("Evaluation split is empty")
    args.out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "predictions.parquet", index=False)
    metrics = {"n": len(df), "split": args.split, "task": checkpoint["task"],
               "architecture": checkpoint["architecture"], "best_epoch": checkpoint["best_epoch"],
               "best_val_loss": checkpoint["best_val_loss"],
               "projection": checkpoint.get("projection", "or")}
    plot_keys = ["x", "y", "cotB"] if checkpoint["task"] == "slim" else ["x", "y", "cotA", "cotB"]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(4.2 * len(plot_keys), 3.8))
    pulls = {}
    for ax, key in zip(axes, plot_keys):
        if key.startswith("cot"):
            pred = np.degrees(np.arctan2(1.0, df[key].to_numpy()))
            truth = np.degrees(np.arctan2(1.0, df[key + "true"].to_numpy()))
            unit = "deg"
        else:
            pred = df[key].to_numpy()
            truth = df[key + "true"].to_numpy()
            unit = "um"
        residual = pred - truth  # Paper convention, R_v = v - v_true.
        lo, hi = minimum_interval_68(residual)
        result = {"mean": float(np.mean(residual)), "std": float(np.std(residual)),
                  "mae": float(np.mean(np.abs(residual))), "interval_68": [lo, hi],
                  "interval_68_width": hi - lo, "unit": unit}
        if "sigma" + key in df:
            sig = df["sigma" + key].to_numpy()
            if key.startswith("cot"):
                # First-order propagation from cot(theta) to theta.
                sig = np.degrees(sig / (1.0 + df[key].to_numpy() ** 2))
            pull = residual / np.maximum(sig, 1e-9)
            pulls[key] = pull
            result.update(mean_sigma=float(np.mean(sig)), pull_mean=float(np.mean(pull)),
                          pull_std=float(np.std(pull)), coverage_1sigma=float(np.mean(np.abs(pull) <= 1)))
        metrics[key] = result
        ax.hist(residual, bins=100, histtype="step", density=True)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.15, label="Minimum 68% interval")
        ax.set(xlabel=f"{key} residual [{unit}]", ylabel="Density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "residuals.png", dpi=180)
    plt.close(fig)
    if pulls:
        fig, axes = plt.subplots(1, len(pulls), figsize=(4.2 * len(pulls), 3.8))
        for ax, (key, pull) in zip(axes, pulls.items()):
            ax.hist(pull, bins=100, range=(-6, 6), density=True, histtype="step")
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set(xlabel=f"{key} pull", ylabel="Density")
        fig.tight_layout()
        fig.savefig(args.out / "pulls.png", dpi=180)
        plt.close(fig)
    history_path = Path(args.checkpoint).parent / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.epoch, history.train_loss, label="Train")
        ax.plot(history.epoch, history.val_loss, label="Validation")
        ax.set(xlabel="Epoch", ylabel="Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "history.png", dpi=180)
        plt.close(fig)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    train_p = commands.add_parser("train")
    train_p.add_argument("--data", type=Path, required=True)
    train_p.add_argument("--out", type=Path, required=True)
    train_p.add_argument("--architecture", choices=("mlp", "conv1d", "conv2d"), default="mlp")
    train_p.add_argument("--task", choices=("slim", "full", "max"), default="slim")
    train_p.add_argument("--projection", choices=("or", "charge_sum", "or_charge_thresholds"),
                         default="or", help="MLP axis projection: OR, binary charge sum, or OR "
                         "plus ordered charge thresholds")
    train_p.add_argument("--epochs", type=int, default=1000)
    train_p.add_argument("--batch-size", type=int, default=5000)
    train_p.add_argument("--readout-bits", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--resume", action="store_true", help="Continue from latest.pt in --out")
    train_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_p = commands.add_parser("evaluate")
    eval_p.add_argument("--data", type=Path, required=True)
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.add_argument("--out", type=Path, required=True)
    eval_p.add_argument("--split", choices=("val", "test"), default="test")
    eval_p.add_argument("--batch-size", type=int, default=5000)
    eval_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
