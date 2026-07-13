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

    def _compute_all(
        self, df: pd.DataFrame, reference_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Serial: compute every feature for ``df`` as of ``reference_time``.

        Returns the wide frame (columns in registry order) before null-filling.
        This is the shared, un-parallelised core used by the single-user and
        point-in-time paths so they call the exact same ``compute()`` code.
        """
        reference_time = pd.Timestamp(reference_time)
        series = [feat.compute(df, reference_time) for feat in self.features]
        wide = pd.concat(series, axis=1) if series else pd.DataFrame()
        return wide.reindex(columns=self.feature_names)

    def transform_batch(
        self,
        df: pd.DataFrame,
        reference_time: pd.Timestamp,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Training path (single as-of time): features for all users in ``df``.

        Every user is evaluated as of the same ``reference_time``. For the
        leakage-safe per-user variant, see :meth:`transform_point_in_time`.
        """
        return self.run(df, reference_time, n_jobs=n_jobs)

    def transform_point_in_time(
        self,
        df: pd.DataFrame,
        reference_times: pd.Series | dict,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Training path (point-in-time): each user as of their own time.

        A user's feature vector reflects only events up to *their* reference
        time (e.g. their ``decision_time``), which matches how the time-based
        split orders users — eliminating the feature-time vs split-time skew.
        Each user is computed through :meth:`transform_single`, so the batch
        and serving paths remain bit-for-bit identical.

        Args:
            df: Event log for many users.
            reference_times: Mapping ``user_id -> reference Timestamp``
                (a dict or a pandas Series indexed by user_id).
            n_jobs: joblib worker count across users.

        Returns:
            DataFrame indexed by ``user_id`` (one row per entry in
            ``reference_times``), one column per feature, null-free.
        """
        if isinstance(reference_times, pd.Series):
            reference_times = reference_times.to_dict()

        groups = dict(tuple(df.groupby("user_id")))
        user_ids = list(reference_times)

        rows = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._user_row)(groups.get(uid), reference_times[uid])
            for uid in user_ids
        )
        out = pd.DataFrame(rows, index=user_ids)
        out.index.name = "user_id"
        return out[self.feature_names].fillna(0.0)

    def transform_samples(
        self,
        df: pd.DataFrame,
        samples: pd.DataFrame,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Point-in-time features for arbitrary decision points.

        Unlike :meth:`transform_point_in_time` (which assumes one row per user),
        this supports **multiple decision points per user** — e.g. a daily
        snapshot per card — without duplicating the event log. Work is
        parallelised **one task per user** (not per sample): each card's history
        is shipped to a worker once, which then computes all of that card's
        decision points through the same :meth:`transform_single` path. This
        keeps the batch-vs-serving parity guarantee while avoiding re-serialising
        a card's history for every snapshot (which OOMs at ~400K daily
        snapshots). The result is indexed by ``sample_id`` in input order.

        Args:
            df: Event log for many users.
            samples: DataFrame with columns ``sample_id``, ``user_id``,
                ``decision_time``. A user may appear in many rows, each with a
                different ``decision_time``.
            n_jobs: joblib worker count across users.

        Returns:
            DataFrame indexed by ``sample_id`` (one row per sample), one column
            per feature, null-free.
        """
        required = {"sample_id", "user_id", "decision_time"}
        missing = required - set(samples.columns)
        if missing:
            raise ValueError(f"samples is missing columns: {sorted(missing)}")

        groups = dict(tuple(df.groupby("user_id")))
        # One task per card: hand the worker that card's whole history once plus
        # all its (sample_id, decision_time) pairs.
        per_user = [
            (uid, sub["sample_id"].tolist(), sub["decision_time"].tolist())
            for uid, sub in samples.groupby("user_id", sort=False)
        ]

        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._user_rows)(groups.get(uid), sample_ids, times)
            for uid, sample_ids, times in per_user
        )

        rows: dict[str, dict] = {}
        for res in results:
            rows.update(res)
        out = pd.DataFrame.from_dict(rows, orient="index")
        # Restore input order (groupby reorders by user).
        out = out.reindex(samples["sample_id"].tolist())
        out.index.name = "sample_id"
        return out[self.feature_names].fillna(0.0)

    def _user_row(
        self, user_history: pd.DataFrame | None, reference_time: pd.Timestamp
    ) -> dict:
        """Feature dict for one user's history (picklable joblib worker)."""
        if user_history is None or user_history.empty:
            return dict.fromkeys(self.feature_names, 0.0)
        return self.transform_single(user_history, reference_time)

    def _user_rows(
        self,
        user_history: pd.DataFrame | None,
        sample_ids: list,
        times: list,
    ) -> dict[str, dict]:
        """Feature dicts for many decision points of ONE user (joblib worker).

        Computes every ``(sample_id, decision_time)`` for a single card from one
        in-memory copy of its history, each through :meth:`transform_single` so
        parity with the serving path is preserved.
        """
        if user_history is None or user_history.empty:
            zero = dict.fromkeys(self.feature_names, 0.0)
            return {sid: dict(zero) for sid in sample_ids}
        return {
            sid: self.transform_single(user_history, pd.Timestamp(t))
            for sid, t in zip(sample_ids, times, strict=True)
        }

    def transform_single(
        self,
        user_history: pd.DataFrame,
        reference_time: pd.Timestamp,
    ) -> dict:
        """Inference path: compute features for a single user.

        Uses the exact same ``compute()`` calls as the batch paths, so a user's
        feature vector is identical whether computed in a batch or alone.

        Args:
            user_history: Event history for exactly one user.
            reference_time: Point in time as of which features are computed.

        Returns:
            Mapping of feature name -> value for that user.
        """
        if user_history.empty:
            return dict.fromkeys(self.feature_names, 0.0)

        user_id = user_history["user_id"].iloc[0]
        wide = self._compute_all(user_history, reference_time)
        # Ensure exactly the target user's row exists (filled with 0 if no events).
        row = wide.reindex([user_id]).fillna(0.0).loc[user_id]
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

    Reads validated raw events and per-user labels, then computes the 35-feature
    matrix **point-in-time**: each user is evaluated as of their own
    ``decision_time`` (the same timestamp the split orders by), so no feature
    reflects events from after the user's decision moment. Writes the processed
    Parquet plus a runtime metric.

    Returns:
        Path to the written features Parquet file.
    """
    events = pd.read_parquet(events_path)
    labels = pd.read_parquet(labels_path)

    pipeline = FeaturePipeline()

    # Two label shapes are supported, both featurized point-in-time through the
    # same compute() path:
    #   * one row per user (synthetic / production) -> key on user_id
    #   * many snapshots per user with a sample_id (e.g. Sparkov daily) -> key
    #     on sample_id, so a card can contribute many (decision_time) rows
    #     without duplicating its event log.
    snapshot_mode = "sample_id" in labels.columns
    key = "sample_id" if snapshot_mode else "user_id"

    start = time.perf_counter()
    if snapshot_mode:
        logger.info(
            "Featurizing %d events for %d samples across %d users point-in-time "
            "(decision_time %s..%s)",
            len(events),
            len(labels),
            labels["user_id"].nunique(),
            labels["decision_time"].min(),
            labels["decision_time"].max(),
        )
        features = pipeline.transform_samples(
            events, labels[["sample_id", "user_id", "decision_time"]]
        )
    else:
        reference_times = labels.set_index("user_id")["decision_time"]
        logger.info(
            "Featurizing %d events for %d users point-in-time "
            "(decision_time %s..%s)",
            len(events),
            labels["user_id"].nunique(),
            reference_times.min(),
            reference_times.max(),
        )
        features = pipeline.transform_point_in_time(events, reference_times)
    runtime_seconds = time.perf_counter() - start

    # Labels define the row universe: a user/sample with no purchases in any
    # window gets an all-zero feature row rather than being dropped.
    full = labels.set_index(key).join(features, how="left")
    full[pipeline.feature_names] = full[pipeline.feature_names].fillna(0.0)
    full["label"] = full["label"].fillna(0).astype("int64")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, engine="pyarrow", compression="snappy")

    metrics = {
        "n_rows": int(len(features)),
        "n_users": int(labels["user_id"].nunique()),
        "n_features": int(len(pipeline.feature_names)),
        "snapshot_mode": snapshot_mode,
        "runtime_seconds": round(runtime_seconds, 4),
        "decision_time_min": labels["decision_time"].min().isoformat(),
        "decision_time_max": labels["decision_time"].max().isoformat(),
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
