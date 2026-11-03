#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HALO pkl output to cross-sectional CSV.")
    parser.add_argument("--synthetic_pkl", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    synthetic = pickle.load(open(args.synthetic_pkl, "rb"))
    data_dir = Path(args.data_dir)
    index_to_code = pickle.load(open(data_dir / "indexToCode.pkl", "rb"))
    label_cols = pickle.load(open(data_dir / "idToLabel.pkl", "rb"))
    index_to_code = {int(k): str(v) for k, v in index_to_code.items()}
    phecode_cols = [index_to_code[i] for i in sorted(index_to_code)]

    rows = []
    for i, patient in enumerate(synthetic):
        labels = np.asarray(patient["labels"]).astype(int).tolist()
        if len(labels) != len(label_cols):
            raise ValueError(f"Subject {i} has {len(labels)} labels, expected {len(label_cols)}.")
        row = {col: int(labels[j]) for j, col in enumerate(label_cols)}
        row.update({col: 0 for col in phecode_cols})
        for visit in patient.get("visits") or []:
            for code_idx in sorted(set(int(x) for x in visit)):
                if code_idx in index_to_code:
                    row[index_to_code[code_idx]] = 1
        rows.append(row)

    out = pd.DataFrame(rows)[list(label_cols) + phecode_cols].astype(int)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print("Shape:", out.shape)


if __name__ == "__main__":
    main()
