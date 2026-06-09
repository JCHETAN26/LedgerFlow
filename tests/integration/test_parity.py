"""
Phase 3 — Training-serving consistency.

The whole point of routing batch and single-user inference through the same
``compute()`` calls is that a user's feature vector must be identical either way.
This test computes features for many users at once (the training path), then for
each of 10 users individually (the serving path), and asserts they match to 6
decimal places.
"""

import numpy as np
import pandas as pd
import pytest

from LedgerFlow.data.synthetic import generate_synthetic_events
from LedgerFlow.pipeline import FeaturePipeline


@pytest.fixture(scope="module")
def synthetic():
    events, labels = generate_synthetic_events(n_users=60, days=30, seed=7)
    reference_time = events["event_timestamp"].max()
    return events, reference_time, labels


def test_batch_single_parity(synthetic):
    events, reference_time, _labels = synthetic
    pipe = FeaturePipeline()

    batch = pipe.transform_batch(events, reference_time)

    users = list(batch.index[:10])
    assert len(users) == 10

    for user_id in users:
        history = events[events["user_id"] == user_id]
        single = pipe.transform_single(history, reference_time)

        for feature_name in pipe.feature_names:
            batch_value = float(batch.loc[user_id, feature_name])
            single_value = float(single[feature_name])
            assert single_value == pytest.approx(batch_value, abs=1e-6), (
                f"Skew in {feature_name} for {user_id}: "
                f"batch={batch_value} single={single_value}"
            )


def test_point_in_time_single_parity(synthetic):
    # The point-in-time batch path must match transform_single computed at each
    # user's own decision_time, to 6 decimals.
    events, _reference_time, labels = synthetic
    pipe = FeaturePipeline()

    ref_times = labels.set_index("user_id")["decision_time"]
    pit = pipe.transform_point_in_time(events, ref_times)

    for user_id in list(ref_times.index[:10]):
        history = events[events["user_id"] == user_id]
        single = pipe.transform_single(history, ref_times[user_id])
        for feature_name in pipe.feature_names:
            assert float(single[feature_name]) == pytest.approx(
                float(pit.loc[user_id, feature_name]), abs=1e-6
            ), f"PIT skew in {feature_name} for {user_id}"


def test_point_in_time_differs_from_global(synthetic):
    # Sanity: per-user point-in-time is genuinely different from a single
    # global reference time (otherwise it isn't doing anything).
    events, reference_time, labels = synthetic
    pipe = FeaturePipeline()

    ref_times = labels.set_index("user_id")["decision_time"]
    pit = pipe.transform_point_in_time(events, ref_times)
    glob = pipe.transform_batch(events, reference_time)

    common = glob.index.intersection(pit.index)
    total_diff = (
        (glob.loc[common, pipe.feature_names] - pit.loc[common, pipe.feature_names])
        .abs()
        .to_numpy()
        .sum()
    )
    assert total_diff > 0


def test_parity_for_user_with_no_purchases(synthetic):
    events, reference_time, _labels = synthetic
    pipe = FeaturePipeline()

    # Construct a user that only has non-purchase events.
    history = pd.DataFrame(
        {
            "user_id": ["lonely"] * 2,
            "event_type": ["view", "click"],
            "event_timestamp": [
                reference_time - pd.Timedelta("1h"),
                reference_time - pd.Timedelta("2h"),
            ],
            "amount": [None, None],
        }
    )
    single = pipe.transform_single(history, reference_time)
    assert all(v == 0.0 for v in single.values())


def test_no_feature_has_separate_online_offline_path():
    # Structural guarantee: every path funnels through the same compute() calls.
    import inspect

    pipe = FeaturePipeline()
    assert pipe.transform_batch.__func__ is FeaturePipeline.transform_batch

    # transform_single and the point-in-time batch share _compute_all, which is
    # the single place feat.compute() is invoked serially.
    assert "self._compute_all(" in inspect.getsource(
        FeaturePipeline.transform_single
    )
    assert "self.transform_single(" in inspect.getsource(
        FeaturePipeline._user_row
    )
    assert "feat.compute(" in inspect.getsource(FeaturePipeline._compute_all)
