"""Compare validation/test regression behavior and GroupSum count usage."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .run import evaluate, minimum_interval_68


FEATURES = (("x", 0, "µm"), ("y", 1, "µm"), ("cotB", 3, "deg"))


def degrees_from_cot(values):
    return np.degrees(np.arctan2(1.0, values))


def summarize(pred_raw, truth_raw, feature, index, units, calibration, bits):
    center = calibration["label_center"][index]
    scale = calibration["label_scale"][index]
    lower, upper = calibration["readout_bounds"]["mean"][index]
    count = ((pred_raw - center) / scale - lower) * bits / (upper - lower)
    rounded_count = np.rint(count).astype(int)
    if feature == "cotB":
        pred, truth = degrees_from_cot(pred_raw), degrees_from_cot(truth_raw)
        baseline = degrees_from_cot(np.asarray(center))
    else:
        pred, truth, baseline = pred_raw, truth_raw, center
    residual = pred - truth
    lo, hi = minimum_interval_68(residual)
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "unit": units, "n": len(truth),
        "truth_mean": float(truth.mean()), "truth_std": float(truth.std()),
        "prediction_mean": float(pred.mean()), "prediction_std": float(pred.std()),
        "residual_mean": float(residual.mean()), "residual_std": float(residual.std()),
        "residual_mae": float(np.abs(residual).mean()),
        "interval_68": [lo, hi], "interval_68_width": hi - lo,
        "r2": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else None,
        "train_mean_baseline_mae": float(np.abs(baseline - truth).mean()),
        "count_levels_used": int(np.unique(rounded_count).size),
        "count_min": int(rounded_count.min()), "count_max": int(rounded_count.max()),
        "count_zero_fraction": float(np.mean(rounded_count == 0)),
        "count_max_fraction": float(np.mean(rounded_count == bits)),
        "most_common_count_fraction": float(np.bincount(
            np.clip(rounded_count, 0, bits), minlength=bits + 1).max() / len(count)),
        "truth_outside_readout_fraction": float(np.mean(
            ((truth_raw - center) / scale < lower) |
            ((truth_raw - center) / scale > upper))),
    }, pred, truth, residual, rounded_count


def plot_report(values, bits, out):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    colors = {"val": "tab:blue", "test": "tab:orange"}
    rng = np.random.default_rng(42)
    for row, (feature, _, units) in enumerate(FEATURES):
        val, test = values[feature]["val"], values[feature]["test"]
        merged = np.concatenate((val[1], val[2], test[1], test[2]))
        bins = np.linspace(merged.min(), merged.max(), 81)
        ax = axes[row, 0]
        for split, (_, pred, truth, _, _) in (("val", val), ("test", test)):
            ax.hist(truth, bins=bins, density=True, histtype="step", color=colors[split],
                    label=f"{split} truth")
            ax.hist(pred, bins=bins, density=True, histtype="step", linestyle="--",
                    color=colors[split], label=f"{split} prediction")
        ax.set(xlabel=f"{feature} [{units}]", ylabel="Density", title="Truth and prediction")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        residuals = np.concatenate((val[3], test[3]))
        low, high = np.quantile(residuals, [0.005, 0.995])
        for split, result in (("val", val), ("test", test)):
            ax.hist(result[3], bins=80, range=(low, high), density=True,
                    histtype="step", color=colors[split], label=split)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set(xlabel=f"Prediction − truth [{units}]", ylabel="Density", title="Residual")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 2]
        for split, result in (("val", val), ("test", test)):
            _, pred, truth, _, _ = result
            indices = rng.choice(len(truth), min(len(truth), 3000), replace=False)
            ax.scatter(truth[indices], pred[indices], s=3, alpha=0.2,
                       color=colors[split], label=split)
        minimum, maximum = float(merged.min()), float(merged.max())
        ax.plot((minimum, maximum), (minimum, maximum), color="black", linewidth=0.7)
        ax.set(xlabel=f"Truth [{units}]", ylabel=f"Prediction [{units}]",
               title="Sampled predictions")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 3]
        for split, result in (("val", val), ("test", test)):
            ax.hist(result[4], bins=np.arange(bits + 2) - 0.5, density=True,
                    histtype="step", color=colors[split], label=split)
        ax.set(xlabel="GroupSum gate count", ylabel="Fraction per count",
               title="Readout count usage", xlim=(-1, bits + 1))
        if row == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Defaults to RUN/diagnostics")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = args.out or args.run / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["task"] != "slim" or saved.get("calibration") is None:
        raise ValueError("This diagnostic expects a centered Slim checkpoint")
    val_dir = args.run / "val"
    val_metrics = val_dir / "metrics.json"
    val_is_current = ((val_dir / "predictions.parquet").exists() and
                      val_metrics.exists() and
                      json.loads(val_metrics.read_text()).get("best_epoch") == saved["best_epoch"])
    if not val_is_current:
        evaluate(argparse.Namespace(data=args.data, checkpoint=checkpoint,
                                    out=val_dir, split="val", batch_size=args.batch_size,
                                    device=args.device))
    test_path = args.run / "test" / "predictions.parquet"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    test_metrics = args.run / "test" / "metrics.json"
    if not test_metrics.exists() or json.loads(test_metrics.read_text()).get("best_epoch") != saved["best_epoch"]:
        raise ValueError("Test predictions do not match the best checkpoint")
    calibration = saved["calibration"]
    bits = saved["readout_bits"]
    report = {"checkpoint": str(checkpoint), "readout_gates_per_feature": bits,
              "best_epoch": saved["best_epoch"],
              "projection": saved.get("projection", "or"), "features": {}}
    values = {}
    for feature, index, units in FEATURES:
        values[feature] = {}
        report["features"][feature] = {}
        for split, path in (("val", val_dir / "predictions.parquet"),
                            ("test", test_path)):
            frame = pd.read_parquet(path, columns=[feature, feature + "true"])
            result = summarize(frame[feature].to_numpy(dtype=np.float64),
                               frame[feature + "true"].to_numpy(dtype=np.float64),
                               feature, index, units, calibration, bits)
            report["features"][feature][split] = result[0]
            values[feature][split] = result
        val = report["features"][feature]["val"]
        test = report["features"][feature]["test"]
        report["features"][feature]["test_to_val_width_ratio"] = (
            test["interval_68_width"] / val["interval_68_width"]
            if val["interval_68_width"] > 0 else None)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(values, bits, out / "val_test_diagnostics.png")
    print(f"Wrote {out / 'diagnostics.json'} and {out / 'val_test_diagnostics.png'}")
    for feature, entry in report["features"].items():
        val, test = entry["val"], entry["test"]
        print(f"{feature}: val/test 68% widths {val['interval_68_width']:.3f}/"
              f"{test['interval_68_width']:.3f} {val['unit']}; "
              f"biases {val['residual_mean']:.3f}/{test['residual_mean']:.3f}; "
              f"count limits test {test['count_zero_fraction']:.2%}/"
              f"{test['count_max_fraction']:.2%}")


if __name__ == "__main__":
    main()
