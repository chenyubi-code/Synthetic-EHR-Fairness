"""In-memory implementations of the Resample and SBS generators."""
from __future__ import annotations

import numpy as np
import pandas as pd


SENSITIVE_COLUMNS = ["age80", "male", "white", "private"]


def prepare_cohort(data: pd.DataFrame, sensitive_columns=SENSITIVE_COLUMNS) -> pd.DataFrame:
    """Return a binary feature frame without patient identifiers."""
    missing = [column for column in sensitive_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing sensitive columns: {missing}")

    try:
        from .prepare_binary_matrix import prepare_binary_matrix
    except ImportError:
        from prepare_binary_matrix import prepare_binary_matrix
    matrix, columns = prepare_binary_matrix(data, sensitive_columns=sensitive_columns)
    return pd.DataFrame(matrix.astype(np.int8), columns=columns)


def generate_resample(
    data: pd.DataFrame,
    n_rows: int,
    seed: int,
    sensitive_columns=SENSITIVE_COLUMNS,
) -> pd.DataFrame:
    """Bootstrap complete patient rows with replacement."""
    sensitive_columns = list(sensitive_columns)
    real = prepare_cohort(data, sensitive_columns)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(real), size=n_rows)
    return real.iloc[indices].reset_index(drop=True)


def generate_sbs(
    data: pd.DataFrame,
    n_rows: int,
    seed: int,
    sensitive_columns=SENSITIVE_COLUMNS,
) -> pd.DataFrame:
    """Sample sensitive strata, then sample phenotypes within each stratum."""
    sensitive_columns = list(sensitive_columns)
    real = prepare_cohort(data, sensitive_columns)
    phenotype_columns = [
        column for column in real.columns if column not in sensitive_columns
    ]
    grouped = real.groupby(sensitive_columns, sort=True, dropna=False)
    strata = grouped.size().reset_index(name="count")
    strata = strata[strata["count"] > 0].reset_index(drop=True)
    stratum_probabilities = strata["count"].to_numpy(dtype=float)
    stratum_probabilities /= stratum_probabilities.sum()

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(
        len(strata),
        size=n_rows,
        replace=True,
        p=stratum_probabilities,
    )
    sensitive = (
        strata.iloc[sampled_indices][sensitive_columns]
        .reset_index(drop=True)
        .astype(np.int8)
    )

    prevalence = grouped[phenotype_columns].mean().reset_index()
    prevalence["_stratum"] = prevalence[sensitive_columns].astype(str).agg(
        "\r".join,
        axis=1,
    )
    prevalence = prevalence.set_index("_stratum")
    sampled_strata = sensitive.astype(str).agg("\r".join, axis=1)
    probabilities = prevalence.loc[
        sampled_strata,
        phenotype_columns,
    ].to_numpy(dtype=np.float64)
    phenotypes = rng.binomial(1, probabilities).astype(np.int8)

    generated = pd.concat(
        [sensitive, pd.DataFrame(phenotypes, columns=phenotype_columns)],
        axis=1,
    )
    return generated[real.columns].astype(np.int8)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("Resample", "SBS"), required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-generate", type=int, default=30000)
    parser.add_argument("--sensitive-columns", nargs=4, default=SENSITIVE_COLUMNS)
    args = parser.parse_args()
    if args.n_generate <= 0:
        parser.error("--n-generate must be positive")
    generator = generate_resample if args.method == "Resample" else generate_sbs
    generated = generator(pd.read_csv(args.input_csv), args.n_generate, args.seed, args.sensitive_columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(args.output_csv, index=False)
