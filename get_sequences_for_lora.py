import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pysam


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def multi_hot(labels, num_labels):
    encoded = np.eye(num_labels, dtype=np.int64)[labels].sum(axis=0)
    if not np.isin(encoded, [0, 1]).all():
        raise ValueError("A BED label list contains duplicate class IDs")
    return encoded


def prepare_data(args):
    bed = pd.read_csv(args.bed_path, header="infer", sep="\t", low_memory=False)
    if bed.empty:
        raise ValueError(f"BED file '{args.bed_path}' contains no records")
    if args.train_size <= 0 or args.test_size <= 0:
        raise ValueError("Train and test sizes must both be positive")
    if args.label_depth <= 0:
        raise ValueError("--label_depth must be positive")
    if bed.shape[1] < 3:
        raise ValueError("BED file must contain chromosome, start, and end columns")
    split_values = bed.iloc[:, -1].astype(str).str.strip().str.lower()
    train_candidates = np.flatnonzero(split_values == "train")
    test_candidates = np.flatnonzero(split_values == "test")
    if args.train_size > len(train_candidates):
        raise ValueError(
            f"Requested {args.train_size} training samples, but the BED file "
            f"contains only {len(train_candidates)} rows marked 'train' in its last column"
        )
    if args.test_size > len(test_candidates):
        raise ValueError(
            f"Requested {args.test_size} test samples, but the BED file "
            f"contains only {len(test_candidates)} rows marked 'test' in its last column"
        )

    rng = np.random.default_rng(args.seed)
    train_indices = rng.choice(train_candidates, size=args.train_size, replace=False)
    test_indices = rng.choice(test_candidates, size=args.test_size, replace=False)
    selected_indices = np.concatenate((train_indices, test_indices))

    if args.hdf5_file:
        with h5py.File(args.hdf5_file, "r") as label_file:
            if args.hdf5_dataset not in label_file:
                raise ValueError(
                    f"HDF5 file '{args.hdf5_file}' has no dataset "
                    f"'{args.hdf5_dataset}'"
                )
            label_dataset = label_file[args.hdf5_dataset]
            if label_dataset.ndim != 2:
                raise ValueError(
                    f"HDF5 labels in '{args.hdf5_file}' must be a 2D multi-hot array"
                )
            label_indices = selected_indices + args.start_offset
            if label_indices.max() >= label_dataset.shape[0]:
                raise ValueError(
                    "Selected BED records exceed the available HDF5 labels; "
                    "check --start_offset and the labels file"
                )
            sorted_label_indices = np.sort(label_indices)
            sorted_labels = np.asarray(label_dataset[sorted_label_indices])
            selected_order = np.argsort(label_indices)
            selected_labels = sorted_labels[np.argsort(selected_order)]
    else:
        if "label" not in bed.columns:
            raise ValueError("BED file needs a 'label' column when no HDF5 labels are provided")
        label_column = "label"

        selected_labels = []
        for index in selected_indices:
            value = bed.iloc[index][label_column]
            class_ids = (
                list(map(int, str(value).split(",")))
                if pd.notna(value) and str(value).strip()
                else []
            )
            if any(class_id < 0 or class_id >= args.label_depth for class_id in class_ids):
                raise ValueError(
                    f"Label class ID in BED row {index} is outside "
                    f"the range 0..{args.label_depth - 1}"
                )
            selected_labels.append(multi_hot(class_ids, args.label_depth))
        selected_labels = np.asarray(selected_labels)

    if selected_labels.shape[1] == 0:
        raise ValueError("Labels must contain at least one class")
    if not np.isfinite(selected_labels).all() or not np.isin(selected_labels, [0, 1]).all():
        raise ValueError("Multi-hot labels must contain only finite 0 and 1 values")

    sequences = []
    with pysam.FastaFile(args.fasta_path) as fasta:
        for index in selected_indices:
            row = bed.iloc[index]
            chrom, start, end = row.iloc[0], int(row.iloc[1]), int(row.iloc[2])
            strand = "+"
            if args.read_strand:
                strand_column = (
                    bed.columns.get_loc("strand")
                    if "strand" in bed.columns
                    else 3
                )
                if strand_column >= len(row):
                    raise ValueError(
                        "--read_strand requires a 'strand' column or a fourth BED column"
                    )
                strand = row.iloc[strand_column]

            sequence = fasta.fetch(
                chrom, max(0, start - args.flank), end + args.flank
            )
            if strand == "-":
                sequence = sequence.translate(
                    str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
                )[::-1]
            sequences.append(sequence)

    split_index = len(train_indices)
    train_sequences = sequences[:split_index]
    test_sequences = sequences[split_index:]
    train_labels = selected_labels[:split_index]
    test_labels = selected_labels[split_index:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_sequences, split_labels in (
        ("train", train_sequences, train_labels),
        ("test", test_sequences, test_labels),
    ):
        csv_path = output_dir / f"{split}.csv"
        labels_path = output_dir / f"{split}_labels.npy"
        pd.DataFrame({"sequence": split_sequences}).to_csv(csv_path, index=False)
        np.save(labels_path, split_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DNA sequences and multi-hot train/test data from a BED file"
    )
    parser.add_argument("--fasta_path", required=True, help="Reference FASTA file")
    parser.add_argument("--bed_path", required=True, help="Tab-separated BED/label file")
    parser.add_argument("--train_size", required=True, type=int, help="Number of training samples")
    parser.add_argument("--test_size", required=True, type=int, help="Number of test samples")
    parser.add_argument("--label_depth", type=int, default=2, help="Number of label classes for BED labels")
    parser.add_argument("--hdf5_file", help="Optional HDF5 file containing multi-hot labels")
    parser.add_argument("--hdf5_dataset", default="labels", help="Dataset name within the HDF5 labels file")
    parser.add_argument("--read_strand", action="store_true", help="Read strand and reverse-complement negative-strand sequences")
    parser.add_argument("--flank", type=int, default=0, help="Bases to add on either side of each interval")
    parser.add_argument("--start_offset", type=int, default=0, help="Offset into HDF5 labels")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the randomized split")
    parser.add_argument("--output_dir", default="prepared_data", help="Directory for CSV and NPY outputs")
    args = parser.parse_args()

    if args.flank < 0:
        parser.error("--flank must be non-negative")
    if args.start_offset < 0:
        parser.error("--start_offset must be non-negative")
    prepare_data(args)


if __name__ == "__main__":
    main()
# python get_sequences.py --fasta_path /home/dj_djenny/benchmarks/BEND/data/genomes/GRCh38.primary_assembly.genome.fa --bed_path /home/dj_djenny/benchmarks/BEND/data/histone_modification/histone_modification.bed --train_size 2500 --test_size 500 --output_dir ./hist_data --label_depth 18
