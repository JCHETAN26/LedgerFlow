"""
Time-based train/val/test splitter for LedgerFlow.

This is event-stream data, so splits are strictly time-ordered — never random.
Users are ordered by their per-user ``decision_time`` and the earliest go to
train, the latest to test. A random split would leak future behaviour into the
training set; ordering by time mirrors how the model is used in production.

Run as a module (``python -m LedgerFlow.data.splitter``) it executes the DVC
``split`` stage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..params import load_params

logger = logging.getLogger(__name__)

TIME_COL = "decision_time"


def time_based_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, val, test) ordered by ``time_col``.

    The latest ``test_size`` fraction becomes the test set, the ``val_size``
    fraction immediately before it becomes validation, and everything earlier is
    training.

    Args:
        df: Feature matrix including ``time_col``.
        test_size: Fraction of rows (most recent) for the test set.
        val_size: Fraction of rows for the validation set.
        time_col: Column to order by.

    Returns:
        ``(train, val, test)`` DataFrames, each time-contiguous and disjoint.
    """
    if time_col not in df.columns:
        raise ValueError(f"Split requires a '{time_col}' column for time ordering")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    ordered = df.sort_values(time_col, kind="stable")
    n = len(ordered)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test

    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]
    return train, val, test


def split_main(
    features_path: str = "data/processed/features.parquet",
    output_dir: str = "data/splits",
) -> dict[str, str]:
    """Run the split stage and write train/val/test Parquet files."""
    params = load_params()
    test_size = params["evaluation"]["test_size"]
    val_size = params["evaluation"]["val_size"]

    df = pd.read_parquet(features_path)
    train, val, test = time_based_split(df, test_size=test_size, val_size=val_size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, engine="pyarrow", compression="snappy")
        paths[name] = str(path)
        logger.info(
            "%s split: %d rows (%.1f%% positive)",
            name,
            len(part),
            100.0 * part["label"].mean() if len(part) else 0.0,
        )

    return paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    split_main()
