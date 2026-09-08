"""In-memory implementations of the Resample and SBS generators."""

import numpy as np
import pandas as pd


SENSITIVE_COLUMNS = ["age80", "male", "white", "private"]


def prepare_cohort(data: pd.DataFrame) -> pd.DataFrame:
    """Return a binary feature frame without patient identifiers."""
    missing = [column for column in SENSITIVE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing sensitive columns: {missing}")

    features = data.drop(columns=["subject_id"], errors="ignore")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (features > 0).astype(np.int8)


def generate_resample(
    data: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap complete patient rows with replacement."""
    real = prepare_cohort(data)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(real), size=n_rows)
    return real.iloc[indices].reset_index(drop=True)


def generate_sbs(
    data: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample sensitive strata, then sample phenotypes within each stratum."""
    real = prepare_cohort(data)
    phenotype_columns = [
        column for column in real.columns if column not in SENSITIVE_COLUMNS
    ]
    grouped = real.groupby(SENSITIVE_COLUMNS, sort=True, dropna=False)
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
        strata.iloc[sampled_indices][SENSITIVE_COLUMNS]
        .reset_index(drop=True)
        .astype(np.int8)
    )

    prevalence = grouped[phenotype_columns].mean().reset_index()
    prevalence["_stratum"] = prevalence[SENSITIVE_COLUMNS].astype(str).agg(
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
