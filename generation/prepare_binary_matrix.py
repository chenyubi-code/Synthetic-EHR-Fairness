"""Prepare an in-memory binary matrix for neural generators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_binary_matrix(
    data: pd.DataFrame,
    add_padding_column: bool = False,
    sensitive_columns=("age80", "male", "white", "private"),
) -> tuple[np.ndarray, list[str]]:
    """Convert a patient-level table to a binary feature matrix and column list."""
    features = data.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    if features.empty or not features.columns.is_unique:
        raise ValueError("Expected non-empty data with unique feature names.")
    required = set(sensitive_columns)
    if len(sensitive_columns) != 4 or len(required) != 4:
        raise ValueError("Exactly four distinct sensitive columns are required.")
    if not required.issubset(features.columns):
        raise ValueError("Missing one or more of the four sensitive attributes.")
    features = features.apply(pd.to_numeric, errors="raise")
    values = features.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Features must contain complete binary 0/1 values.")
    features = features.astype(np.float32)

    if add_padding_column:
        features["ehrdiff_padding"] = np.float32(0)

    return features.to_numpy(dtype=np.float32), features.columns.astype(str).tolist()


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--columns-json", type=Path, required=True)
    parser.add_argument("--pad-ehrdiff", action="store_true")
    parser.add_argument("--sensitive-columns", nargs=4,
                        default=["age80", "male", "white", "private"])
    args = parser.parse_args()
    matrix, columns = prepare_binary_matrix(pd.read_csv(args.input_csv), args.pad_ehrdiff, args.sensitive_columns)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    args.columns_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, matrix, allow_pickle=False)
    args.columns_json.write_text(json.dumps(columns, indent=2) + "\n")
