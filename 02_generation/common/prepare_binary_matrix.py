"""Prepare an in-memory binary matrix for neural generators."""

import numpy as np
import pandas as pd


def prepare_binary_matrix(
    data: pd.DataFrame,
    add_padding_column: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Convert a patient-level table to a binary feature matrix and column list."""
    features = data.drop(columns=["subject_id"], errors="ignore")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
    features = (features > 0).astype(np.float32)

    if add_padding_column:
        features["ehrdiff_padding"] = np.float32(0)

    return features.to_numpy(dtype=np.float32), features.columns.astype(str).tolist()
