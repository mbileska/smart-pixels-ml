"""Convert the public Zenodo recon3D/labels Parquet pairs to small training shards."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LABELS = ("x-midplane", "y-midplane", "cotAlpha", "cotBeta")
ZENODO_RECORD = "10.5281/zenodo.18472791"


def find_pairs(root):
    root = Path(root)
    images = sorted(root.rglob("recon3D_*.parquet"))
    if not images:
        combined = sorted(root.rglob("part.*.parquet"))
        if combined:
            return [(image, None) for image in combined]
    labels = {p.name.removeprefix("labels_"): p for p in root.rglob("labels_*.parquet")}
    pairs = []
    for image in images:
        suffix = image.name.removeprefix("recon3D_")
        if suffix not in labels:
            raise FileNotFoundError(f"No labels_{suffix} for {image}")
        pairs.append((image, labels[suffix]))
    if not pairs:
        raise FileNotFoundError(f"No recon3D_*.parquet or part.*.parquet files below {root}; unzip the Zenodo archive first")
    return pairs


def batches(image, labels, frames, batch_size):
    image_file = pq.ParquetFile(image)
    if labels is not None:
        label_file = pq.ParquetFile(labels)
        if image_file.metadata.num_rows != label_file.metadata.num_rows:
            raise ValueError(f"Row count mismatch: {image} and {labels}")
    wanted = [str(t * 256 + i) for t in frames for i in range(256)]
    absent = set(wanted) - set(image_file.schema.names)
    if absent:
        raise ValueError(f"Missing charge columns in {image}: {sorted(absent)[:5]}")
    label_table = (label_file if labels is not None else image_file).read(columns=list(LABELS))
    y_all = np.column_stack([label_table[c].to_numpy() for c in LABELS]).astype(np.float32)
    offset = 0
    has_edge = labels is None and "chargeOriginal_atEdge" in image_file.schema.names
    for record_batch in image_file.iter_batches(batch_size=batch_size, columns=wanted + (["chargeOriginal_atEdge"] if has_edge else [])):
        n = record_batch.num_rows
        charge = np.column_stack([record_batch.column(i).to_numpy() for i in range(len(wanted))])
        charge = charge.astype(np.float32).reshape(n, len(frames), 16, 16)
        edge = record_batch.column(len(wanted)).to_numpy() if has_edge else None
        yield charge, y_all[offset:offset + n], edge
        offset += n
    assert offset == len(y_all)


def contained(charge):
    # The paper's criterion uses total charge in unique boundary pixels at 4 ns.
    last = charge[:, -1]
    edge = last[:, 0, :].sum(1) + last[:, -1, :].sum(1)
    edge += last[:, 1:-1, 0].sum(1) + last[:, 1:-1, -1].sum(1)
    return edge < 50


def prepare(source, out, kind, frames, thresholds, seed, batch_size, max_files=None):
    pairs = find_pairs(source)
    if max_files is not None:
        pairs = pairs[:max_files]
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "val": 0, "test": 0, "rejected": 0}
    scale = np.zeros(4, dtype=np.float64)
    shard = 0
    for image, labels in pairs:
        file_train_labels = []
        for charge, y, edge in batches(image, labels, frames, batch_size):
            keep = (edge < 50 if edge is not None else contained(charge))
            keep &= np.isfinite(y).all(1) & np.isfinite(charge).all(axis=(1, 2, 3))
            counts["rejected"] += int((~keep).sum())
            charge, y = charge[keep], y[keep]
            if not len(y):
                continue
            codes = np.searchsorted(thresholds, charge, side="right").astype(np.uint8)
            if kind == "trainval":
                validation = rng.random(len(y)) < 0.2
                splits = (("train", ~validation), ("val", validation))
            else:
                splits = (("test", np.ones(len(y), dtype=bool)),)
            for split, selection in splits:
                if not selection.any():
                    continue
                target = out / split
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f"shard_{shard:05d}.npz", x=codes[selection], y=y[selection])
                counts[split] += int(selection.sum())
                if split == "train":
                    file_train_labels.append(np.abs(y[selection]))
            shard += 1
        if file_train_labels:
            # The repository takes the maximum of the per-file 99th percentiles.
            scale = np.maximum(scale, np.percentile(np.concatenate(file_train_labels), 99, axis=0))
    if kind == "trainval":
        if not counts["train"] or not counts["val"] or np.any(scale == 0):
            raise ValueError(f"Empty split or invalid scale: {counts}, {scale}")
    else:
        if not counts["test"]:
            raise ValueError("Empty test split")
    return counts, scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-source", type=Path, required=True, help="Unpacked 3sExtended archive")
    p.add_argument("--test-source", type=Path, required=True, help="Unpacked 2s archive")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(2, 20), default=2)
    p.add_argument("--thresholds", type=float, nargs=3, default=(247.8, 668.4, 1662.9))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-files", type=int, help="Small exploratory run; omit for full dataset")
    args = p.parse_args()
    if not args.thresholds[0] < args.thresholds[1] < args.thresholds[2]:
        p.error("Thresholds must be strictly increasing")
    if any((args.out / x).exists() for x in ("train", "val", "test")):
        p.error("Output splits already exist; choose a new --out directory")
    frames = (0, 19) if args.frames == 2 else tuple(range(20))
    train_counts, scale = prepare(args.train_source, args.out, "trainval", frames,
                                  args.thresholds, args.seed, args.batch_size, args.max_files)
    test_counts, _ = prepare(args.test_source, args.out, "test", frames,
                             args.thresholds, args.seed, args.batch_size, args.max_files)
    metadata = {"zenodo_record": ZENODO_RECORD, "frames": list(frames),
                "thresholds_electrons": list(args.thresholds), "label_names": list(LABELS),
                "label_scale": scale.tolist(), "seed": args.seed,
                "counts": {"train": train_counts["train"], "val": train_counts["val"],
                           "test": test_counts["test"], "rejected_train": train_counts["rejected"],
                           "rejected_test": test_counts["rejected"]},
                "train_source": str(args.train_source), "test_source": str(args.test_source),
                "split_note": "Seeded 80/20 event split of contained 3sExtended; 2s used only for test."}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
