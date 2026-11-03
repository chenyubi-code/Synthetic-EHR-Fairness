"""Restore feature names and convert generated matrices to binary tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convert_matrix(matrix, columns, probabilities=False, seed=202601):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns) or len(set(columns)) != len(columns):
        raise ValueError("Matrix width must match the unique saved feature names.")
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Generated values must be finite and between 0 and 1.")
    if probabilities:
        matrix = np.random.default_rng(seed).binomial(1, matrix)
    elif not np.isin(matrix, [0, 1]).all():
        raise ValueError("Non-binary values: use --probabilities for Bernoulli sampling.")
    result = pd.DataFrame(matrix.astype(np.int8), columns=columns)
    return result.drop(columns=["ehrdiff_padding"], errors="ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--columns-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--seed", type=int, default=202601)
    args = parser.parse_args()
    frame = convert_matrix(np.load(args.input_npy, allow_pickle=False),
                           json.loads(args.columns_json.read_text()), args.probabilities, args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
