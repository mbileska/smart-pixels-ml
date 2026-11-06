"""Fit label normalization and per-feature GroupSum ranges on prepared labels."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import LABELS


def labels_in(folder):
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No prepared shards in {folder}")
    parts = []
    for file in files:
        with np.load(file) as shard:
            parts.append(shard["y"])
    labels = np.concatenate(parts).astype(np.float64)
    if labels.ndim != 2 or labels.shape[1] != len(LABELS) or not np.isfinite(labels).all():
        raise ValueError(f"Invalid labels in {folder}")
    return labels


def fit(train, target_span=1, margin=0.1):
    """Training extrema fit inside the count range; margin extends both ends."""
    if target_span not in (1, 2) or margin <= 0:
        raise ValueError("target_span must be 1 or 2 and margin must be positive")
    center = train.mean(axis=0)
    scale = np.max(np.abs(train - center), axis=0) / target_span
    if np.any(scale <= 0):
        raise ValueError("Each label must vary in the training split")
    normalized = (train - center) / scale
    low, high = normalized.min(axis=0), normalized.max(axis=0)
    span = high - low
    # Also reserve headroom beyond +/- target_span on both sides. A feature's
    # training extrema may be asymmetric around its mean, and the held-out
    # physical sample can extend farther on the shorter side.
    mean_bounds = np.column_stack((np.minimum(low - margin * span,
                                               -target_span * (1 + margin)),
                                    np.maximum(high + margin * span,
                                               target_span * (1 + margin))))
    sigma_upper = np.maximum(mean_bounds[:, 1] - mean_bounds[:, 0], 1.0)
    sigma_bounds = np.column_stack((np.full(4, 0.02), sigma_upper))
    # A Cholesky off-diagonal can be either sign and must cover a full
    # feature-scale correlation. Its row feature determines the bound.
    offdiag_rows = (1, 2, 2, 3, 3, 3)
    offdiag_bounds = [[-float(sigma_upper[i]), float(sigma_upper[i])]
                      for i in offdiag_rows]
    return {
        "version": 1, "source": "train", "target_span": target_span, "margin": margin,
        "label_names": list(LABELS), "label_center": center.tolist(),
        "label_scale": scale.tolist(), "train_min": train.min(axis=0).tolist(),
        "train_max": train.max(axis=0).tolist(),
        "normalized_train_min": low.tolist(), "normalized_train_max": high.tolist(),
        "readout_bounds": {"mean": mean_bounds.tolist(), "sigma": sigma_bounds.tolist(),
                           "offdiag": offdiag_bounds},
    }


def split_stats(labels, calibration):
    center = np.asarray(calibration["label_center"])
    scale = np.asarray(calibration["label_scale"])
    z = (labels - center) / scale
    bounds = np.asarray(calibration["readout_bounds"]["mean"])
    outside = (z < bounds[:, 0]) | (z > bounds[:, 1])
    return {"count": len(labels), "min": labels.min(axis=0).tolist(),
            "max": labels.max(axis=0).tolist(), "normalized_min": z.min(axis=0).tolist(),
            "normalized_max": z.max(axis=0).tolist(),
            "outside_readout_count": outside.sum(axis=0).astype(int).tolist()}


def plot_labels(splits, calibration, output):
    center = np.asarray(calibration["label_center"])
    scale = np.asarray(calibration["label_scale"])
    bounds = np.asarray(calibration["readout_bounds"]["mean"])
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for i, name in enumerate(LABELS):
        raw_all = np.concatenate([labels[:, i] for labels in splits.values()])
        raw_bins = np.linspace(raw_all.min(), raw_all.max(), 81)
        normalized_bins = (raw_bins - center[i]) / scale[i]
        for split, labels in splits.items():
            raw = labels[:, i]
            z = (raw - center[i]) / scale[i]
            for j, values in enumerate((raw, z)):
                bins = raw_bins if j == 0 else normalized_bins
                axes[i, j].hist(values, bins=bins, density=True, histtype="step",
                                linewidth=1.1, color=colors[split], label=split)
        axes[i, 0].set(xlabel=f"{name} (original units)", ylabel="Density")
        axes[i, 1].set(xlabel=f"{name} (centered / scaled)", ylabel="Density")
        axes[i, 1].axvline(0, color="black", linewidth=0.7)
        for boundary in bounds[i]:
            axes[i, 1].axvline(boundary, color="tab:red", linestyle="--", linewidth=0.9)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].text(0.98, 0.95, "dashed: GroupSum limits", transform=axes[0, 1].transAxes,
                    va="top", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared 2frame_2bit directory")
    parser.add_argument("--target-span", type=int, choices=(1, 2), default=1)
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Extra fraction of each feature's observed normalized range per end")
    parser.add_argument("--readout-bits", type=int, default=64)
    args = parser.parse_args()
    if args.readout_bits <= 0:
        parser.error("--readout-bits must be positive")
    metadata = json.loads((args.data / "metadata.json").read_text())
    if metadata["label_names"] != list(LABELS):
        raise ValueError("Prepared label order differs from calibration label order")
    splits = {name: labels_in(args.data / name) for name in ("train", "val", "test")}
    calibration = fit(splits["train"], args.target_span, args.margin)
    calibration["prepared_counts"] = {name: len(labels) for name, labels in splits.items()}
    calibration["split_stats"] = {name: split_stats(labels, calibration)
                                  for name, labels in splits.items()}
    widths = np.diff(np.asarray(calibration["readout_bounds"]["mean"]), axis=1)[:, 0]
    calibration["mean_step_at_readout_bits"] = {
        "bits": args.readout_bits, "normalized": (widths / args.readout_bits).tolist(),
        "original_units": (widths * np.asarray(calibration["label_scale"]) /
                           args.readout_bits).tolist()}
    output = args.data / "calibration.json"
    plot = args.data / "label_normalization.png"
    output.write_text(json.dumps(calibration, indent=2) + "\n")
    plot_labels(splits, calibration, plot)
    print(f"Wrote {output} and {plot}")
    for name, stats in calibration["split_stats"].items():
        print(f"{name}: {stats['count']} rows; outside GroupSum counts: {stats['outside_readout_count']}")
    print("Per-feature mean GroupSum (tau, beta) at", args.readout_bits, "bits:")
    for name, (low, high), step in zip(LABELS, calibration["readout_bounds"]["mean"],
                                       calibration["mean_step_at_readout_bits"]["original_units"]):
        tau = args.readout_bits / (high - low)
        print(f"  {name}: tau={tau:.8g}, beta={low * tau:.8g}, "
              f"range=[{low:.5g}, {high:.5g}], original-unit step={step:.5g}")


if __name__ == "__main__":
    main()
