"""
Feature pipeline for LedgerFlow.

The :class:`FeaturePipeline` computes all registered features and assembles them
into a single wide DataFrame (one row per user). Both the batch (training) and
single-user (inference) entry points call the same ``run()`` method, which calls
each feature's ``compute()`` — this single code path is what prevents
training-serving skew (see :mod:`LedgerFlow` Phase 3).

Run as a module (``python -m LedgerFlow.pipeline``) it executes the DVC
``featurize`` stage: read validated raw events, compute the feature matrix,
attach labels, and write ``data/processed/features.parquet`` plus a runtime
metric to ``reports/feature_runtime.json``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from .features.base import BaseFeature
from .features.time_windows import ALL_FEATURES

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Compute a wide feature matrix from raw events.

    Args:
        features: Feature instances to compute. Defaults to all 35 registered
            time-window features.
    """

    def __init__(self, features: list[BaseFeature] | None = None):
        self.features = list(features) if features is not None else list(ALL_FEATURES)
        self.feature_names = [f.name for f in self.features]

    def run(
        self,
        df: pd.DataFrame,
        reference_time: pd.Timestamp,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Compute all features and return a wide, null-free DataFrame.

        Args:
            df: Raw event log (validated) for one or many users.
            reference_time: Point in time as of which features are computed.
            n_jobs: joblib worker count. ``-1`` uses all cores; ``1`` runs
                serially (used by the single-user path).

        Returns:
            DataFrame indexed by ``user_id`` with one column per feature, in
            registry order. Users with no events in a window get 0.
        """
        reference_time = pd.Timestamp(reference_time)

        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(feat.compute)(df, reference_time) for feat in self.features
        )

        wide = pd.concat(results, axis=1) if results else pd.DataFrame()
        # Guarantee every feature column exists and columns are in registry order,
        # even if some feature produced an empty Series.
        wide = wide.reindex(columns=self.feature_names)
        wide.index.name = "user_id"

        # Users with no activity in a window had 0 events / 0 spend.
        return wide.fillna(0.0)

    def transform_batch(
        self,
        df: pd.DataFrame,
        reference_time: pd.Timestamp,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Training path: compute features for all users in ``df``."""
        return self.run(df, reference_time, n_jobs=n_jobs)

    def transform_single(
        self,
        user_history: pd.DataFrame,
        reference_time: pd.Timestamp,
    ) -> dict:
        """Inference path: compute features for a single user.

        Uses the exact same ``compute()`` calls as :meth:`transform_batch`, so a
        user's feature vector is identical whether computed in a batch or alone.

        Args:
            user_history: Event history for exactly one user.
            reference_time: Point in time as of which features are computed.

        Returns:
            Mapping of feature name -> value for that user.
        """
        if user_history.empty:
            return dict.fromkeys(self.feature_names, 0.0)

        user_id = user_history["user_id"].iloc[0]
        result = self.run(user_history, reference_time, n_jobs=1)
        # Ensure exactly the target user's row exists (filled with 0 if no events).
        row = result.reindex([user_id]).fillna(0.0).loc[user_id]
        return dict(row.to_dict())


# --------------------------------------------------------------------------- #
# DVC `featurize` stage entry point
# --------------------------------------------------------------------------- #

def featurize_main(
    events_path: str = "data/raw/events.parquet",
    labels_path: str = "data/raw/labels.parquet",
    output_path: str = "data/processed/features.parquet",
    metrics_path: str = "reports/feature_runtime.json",
) -> str:
    """Run the featurize stage end-to-end and write outputs.

    Reads validated raw events, computes the 35-feature matrix as of the most
    recent event timestamp, joins per-user labels, and writes the processed
    Parquet plus a runtime metric.

    Returns:
        Path to the written features Parquet file.
    """
    events = pd.read_parquet(events_path)
    labels = pd.read_parquet(labels_path)

    reference_time = events["event_timestamp"].max()
    logger.info(
        "Featurizing %d events for %d users as of %s",
        len(events),
        events["user_id"].nunique(),
        reference_time,
    )

    pipeline = FeaturePipeline()

    start = time.perf_counter()
    features = pipeline.transform_batch(events, reference_time)
    runtime_seconds = time.perf_counter() - start

    # Labels define the user universe: a user with no purchases at all has an
    # all-zero feature row rather than being dropped from training.
    full = labels.set_index("user_id").join(features, how="left")
    full[pipeline.feature_names] = full[pipeline.feature_names].fillna(0.0)
    full["label"] = full["label"].fillna(0).astype("int64")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, engine="pyarrow", compression="snappy")

    metrics = {
        "n_users": int(len(features)),
        "n_features": int(len(pipeline.feature_names)),
        "runtime_seconds": round(runtime_seconds, 4),
        "reference_time": reference_time.isoformat(),
    }
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "Wrote %d x %d feature matrix to %s in %.3fs",
        len(features),
        len(pipeline.feature_names),
        out_path,
        runtime_seconds,
    )
    return str(out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    featurize_main()
