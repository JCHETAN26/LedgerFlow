"""
Time-window aggregation features for LedgerFlow.

This module builds the 35 time-window aggregation features as
5 time windows x 7 aggregations over purchase events:

    windows      = ["1h", "6h", "24h", "7d", "30d"]
    aggregations = ["count", "sum", "mean", "std", "min", "max", "last"]

Every feature shares a single ``compute()`` implementation
(:class:`TimeWindowAggregation`) so that batch (training) and single-user
(inference) paths can never diverge. The factory
:func:`create_all_time_window_features` instantiates and registers all 35.
"""

from __future__ import annotations

import pandas as pd

from ..registry import register_feature
from .base import BaseFeature

# The 5 windows and 7 aggregations that make up the 35 features.
WINDOWS: list[str] = ["1h", "6h", "24h", "7d", "30d"]
AGGREGATIONS: list[str] = ["count", "sum", "mean", "std", "min", "max", "last"]

# Required input columns for every time-window feature.
REQUIRED_COLUMNS = ("user_id", "event_type", "event_timestamp", "amount")

# Human-readable description fragments per aggregation.
_AGG_DESCRIPTIONS = {
    "count": "Number of purchase events in the last {window}",
    "sum": "Total purchase amount in the last {window}",
    "mean": "Average purchase amount in the last {window}",
    "std": "Standard deviation of purchase amounts in the last {window}",
    "min": "Smallest purchase amount in the last {window}",
    "max": "Largest purchase amount in the last {window}",
    "last": "Most recent purchase amount in the last {window}",
}


class TimeWindowAggregation(BaseFeature):
    """A single time-window aggregation over purchase events.

    One class, parameterised by ``window`` and ``aggregation``, backs all 35
    features. ``count`` produces an integer event count; the other six
    aggregations operate on the ``amount`` column and produce floats that are
    nullable (a user with no purchases in the window has no value, which the
    pipeline later fills with 0).

    Args:
        window: A pandas offset alias, e.g. "1h", "24h", "7d", "30d".
        aggregation: One of :data:`AGGREGATIONS`.
    """

    def __init__(self, window: str, aggregation: str):
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Expected one of {AGGREGATIONS}."
            )

        self.window = window
        self.aggregation = aggregation

        if aggregation == "count":
            self.name = f"purchase_count_{window}"
            self.output_dtype = "int"
            self.nullable = False
        else:
            self.name = f"purchase_amount_{aggregation}_{window}"
            self.output_dtype = "float"
            # Users with no purchases in the window have no aggregate value;
            # the pipeline fills these with 0 when building the wide frame.
            self.nullable = True

        self.description = _AGG_DESCRIPTIONS[aggregation].format(window=window)

        register_feature(self)

    def _filter_window(
        self, df: pd.DataFrame, reference_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Return purchase events in ``(reference_time - window, reference_time]``."""
        cutoff = reference_time - pd.Timedelta(self.window)
        mask = (
            (df["event_type"] == "purchase")
            & (df["event_timestamp"] > cutoff)
            & (df["event_timestamp"] <= reference_time)
        )
        return df.loc[mask]

    def compute(
        self, df: pd.DataFrame, reference_time: pd.Timestamp
    ) -> pd.Series:
        """Compute the aggregation for all users as of ``reference_time``.

        Returns a Series indexed by ``user_id`` and named ``self.name``,
        containing only users who had at least one qualifying purchase in the
        window. Missing users are filled downstream by the pipeline.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name}: input DataFrame missing required columns {missing}"
            )

        filtered = self._filter_window(df, reference_time)

        if self.aggregation == "count":
            result = filtered.groupby("user_id").size()
            return result.astype("int64").rename(self.name)

        # Amount-based aggregations ignore rows with a null amount.
        filtered = filtered[filtered["amount"].notna()]
        grouped = filtered.groupby("user_id")["amount"]

        if self.aggregation == "last":
            # "Most recent" purchase amount in the window.
            ordered = filtered.sort_values("event_timestamp")
            result = ordered.groupby("user_id")["amount"].last()
        else:
            result = grouped.agg(self.aggregation)

        return result.astype("float64").rename(self.name)


def create_all_time_window_features() -> list[BaseFeature]:
    """Create and register all 35 time-window aggregation features.

    Returns:
        List of 35 :class:`TimeWindowAggregation` instances, ordered window-major.
    """
    return [
        TimeWindowAggregation(window=window, aggregation=aggregation)
        for window in WINDOWS
        for aggregation in AGGREGATIONS
    ]


# Pre-create all features for easy import (also populates the registry).
ALL_FEATURES: list[BaseFeature] = create_all_time_window_features()
