"""Paper-style summary of residual mean, minimum 68% interval, and mean sigma."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    metrics = [json.loads(path.read_text()) for path in args.metrics]
    labels = []
    for m in metrics:
        label = f"{m['architecture']} {m['task']} {m.get('projection', 'or')}"
        if "variant" in m:
            label += f" {m['variant']} {m['normalization']}"
        labels.append(label)
    fig, axes = plt.subplots(1, 4, figsize=(15, max(4, 0.55 * len(metrics) + 2.5)))
    keys = ("x", "y", "cotA", "cotB")
    for ax, key in zip(axes, keys):
        for i, m in enumerate(metrics):
            if key not in m:
                continue
            entry = m[key]
            lo, hi = entry["interval_68"]
            ax.plot([lo, hi], [i, i], color=f"C{i % 10}", linewidth=4, alpha=0.6)
            ax.scatter(entry["mean"], i, color=f"C{i % 10}", marker="o", s=30)
            if "mean_sigma" in entry:
                sig = entry["mean_sigma"]
                ax.plot([entry["mean"] - sig, entry["mean"] + sig],
                        [i + 0.15, i + 0.15], color=f"C{i % 10}", linewidth=1)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set(xlabel=f"Residual [{metrics[0].get(key, metrics[-1].get(key, {})).get('unit', 'um')}]",
               title={"x": "x", "y": "y", "cotA": "alpha", "cotB": "beta"}[key])
        ax.set_yticks(np.arange(len(metrics)), labels if key == "x" else [])
        ax.invert_yaxis()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(args.out)


if __name__ == "__main__":
    main()
