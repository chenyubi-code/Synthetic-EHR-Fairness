#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLS = ["age80", "male", "white", "private"]
ID_COLS = ["subject_id", "hadm_id"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build HALO train/validation pkl files.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_seed", type=int, default=202600)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    missing = [col for col in ID_COLS + LABEL_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required HALO columns: {missing}")

    phecode_cols = [c for c in df.columns if c not in ID_COLS + LABEL_COLS]
    df["subject_id"] = df["subject_id"].astype(str)
    df["hadm_id"] = df["hadm_id"].astype(str)
    df[LABEL_COLS + phecode_cols] = df[LABEL_COLS + phecode_cols].fillna(0)
    for col in LABEL_COLS + phecode_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)

    df = df.sort_values(["subject_id", "hadm_id"])
    code_to_index = {code: i for i, code in enumerate(phecode_cols)}
    index_to_code = {i: code for code, i in code_to_index.items()}

    data = []
    empty_visits = 0
    skipped_patients = 0
    inconsistent_labels = 0
    for _, group in df.groupby("subject_id", sort=False):
        group = group.sort_values("hadm_id")
        labels = group.iloc[0][LABEL_COLS].astype(int).to_numpy()
        if not np.all(group[LABEL_COLS].astype(int).to_numpy() == labels):
            inconsistent_labels += 1

        visits = []
        for _, row in group.iterrows():
            visit_codes = [code_to_index[c] for c in phecode_cols if int(row[c]) == 1]
            if visit_codes:
                visits.append(visit_codes)
            else:
                empty_visits += 1

        if visits:
            data.append({"labels": labels, "visits": visits})
        else:
            skipped_patients += 1

    if not data:
        raise ValueError("No HALO patients left after processing.")

    if args.val_fraction > 0:
        train_dataset, val_dataset = train_test_split(
            data,
            test_size=args.val_fraction,
            random_state=args.split_seed,
            shuffle=True,
        )
    else:
        train_dataset = data
        val_dataset = data

    pickle.dump(code_to_index, open(out_dir / "codeToIndex.pkl", "wb"))
    pickle.dump(index_to_code, open(out_dir / "indexToCode.pkl", "wb"))
    pickle.dump(LABEL_COLS, open(out_dir / "idToLabel.pkl", "wb"))
    pickle.dump(train_dataset, open(out_dir / "trainDataset.pkl", "wb"))
    pickle.dump(val_dataset, open(out_dir / "valDataset.pkl", "wb"))

    print("Input:", args.input_csv)
    print("Output:", out_dir)
    print("Phecode columns:", len(phecode_cols))
    print("Patients:", len(data))
    print("Train patients:", len(train_dataset))
    print("Validation patients:", len(val_dataset))
    print("Empty visits skipped:", empty_visits)
    print("Patients skipped:", skipped_patients)
    print("Patients with inconsistent labels:", inconsistent_labels)


if __name__ == "__main__":
    main()
