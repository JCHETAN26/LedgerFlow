"""
Unit tests for the FeaturePipeline (assembly, null-handling, single-user path).

The training-serving parity test lives in tests/integration/test_parity.py.
"""

import numpy as np
import pandas as pd
import pytest

from LedgerFlow.features.time_windows import ALL_FEATURES
from LedgerFlow.pipeline import FeaturePipeline

REFERENCE = pd.Timestamp("2024-01-15 12:00:00")


@pytest.fixture
def events():
    rows = [
        ("a", "purchase", REFERENCE - pd.Timedelta(minutes=30), 5.0),
        ("a", "purchase", REFERENCE - pd.Timedelta(hours=3), 10.0),
        ("b", "purchase", REFERENCE - pd.Timedelta(minutes=45), 20.0),
        ("b", "login", REFERENCE - pd.Timedelta(minutes=10), None),
        ("c", "view", REFERENCE - pd.Timedelta(minutes=5), None),
    ]
    return pd.DataFrame(
        rows, columns=["user_id", "event_type", "event_timestamp", "amount"]
    )


def test_run_shape_and_columns(events):
    pipe = FeaturePipeline()
    wide = pipe.run(events, REFERENCE, n_jobs=1)
    assert list(wide.columns) == [f.name for f in ALL_FEATURES]
    assert wide.shape[1] == 35
    assert wide.index.name == "user_id"


def test_run_has_no_nulls(events):
    pipe = FeaturePipeline()
    wide = pipe.run(events, REFERENCE, n_jobs=1)
    assert not wide.isnull().any().any()


def test_users_with_no_purchases_filled_zero(events):
    pipe = FeaturePipeline()
    wide = pipe.run(events, REFERENCE, n_jobs=1)
    # user_c only has non-purchase events; if present its row is all zeros.
    if "c" in wide.index:
        assert (wide.loc["c"] == 0).all()


def test_transform_batch_matches_run(events):
    pipe = FeaturePipeline()
    a = pipe.run(events, REFERENCE, n_jobs=1)
    b = pipe.transform_batch(events, REFERENCE, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)


def test_transform_single_returns_all_features(events):
    pipe = FeaturePipeline()
    user_a = events[events["user_id"] == "a"]
    result = pipe.transform_single(user_a, REFERENCE)
    assert isinstance(result, dict)
    assert set(result.keys()) == {f.name for f in ALL_FEATURES}


def test_transform_single_known_values(events):
    pipe = FeaturePipeline()
    user_a = events[events["user_id"] == "a"]
    result = pipe.transform_single(user_a, REFERENCE)
    assert result["purchase_count_24h"] == pytest.approx(2.0)
    assert result["purchase_amount_sum_24h"] == pytest.approx(15.0)
    assert result["purchase_amount_max_24h"] == pytest.approx(10.0)


def test_transform_single_empty_history():
    pipe = FeaturePipeline()
    empty = pd.DataFrame(
        {"user_id": [], "event_type": [], "event_timestamp": [], "amount": []}
    )
    result = pipe.transform_single(empty, REFERENCE)
    assert set(result.keys()) == {f.name for f in ALL_FEATURES}
    assert all(v == 0.0 for v in result.values())


def test_transform_point_in_time_per_user(events):
    pipe = FeaturePipeline()
    # Evaluate user "a" at REFERENCE, user "b" 40 minutes earlier.
    ref_times = {"a": REFERENCE, "b": REFERENCE - pd.Timedelta("40min")}
    pit = pipe.transform_point_in_time(events, ref_times, n_jobs=1)

    assert list(pit.index) == ["a", "b"]
    assert list(pit.columns) == pipe.feature_names
    # a within 24h has 2 purchases (5.0 @ -30m, 10.0 @ -3h).
    assert pit.loc["a", "purchase_count_24h"] == pytest.approx(2.0)
    # b's purchase (-45m) is 5 min before b's reference time -> counted in 1h.
    assert pit.loc["b", "purchase_count_1h"] == pytest.approx(1.0)


def test_transform_point_in_time_accepts_series(events):
    pipe = FeaturePipeline()
    ref_times = pd.Series(
        {"a": REFERENCE, "b": REFERENCE}, name="decision_time"
    )
    pit = pipe.transform_point_in_time(events, ref_times, n_jobs=1)
    assert set(pit.index) == {"a", "b"}
    assert not pit.isnull().any().any()


def test_transform_point_in_time_unknown_user_is_zero(events):
    pipe = FeaturePipeline()
    ref_times = {"ghost": REFERENCE}  # no events for this user
    pit = pipe.transform_point_in_time(events, ref_times, n_jobs=1)
    assert (pit.loc["ghost"] == 0).all()


def test_custom_feature_subset():
    subset = ALL_FEATURES[:3]
    pipe = FeaturePipeline(features=subset)
    assert pipe.feature_names == [f.name for f in subset]
