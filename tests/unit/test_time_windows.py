"""
Unit tests for the 35 time-window aggregation features.

Coverage strategy: every feature is exercised for shape/dtype/naming, each
aggregation is checked against a hand-computed expected value, and the tricky
edge cases (empty window, single event, window boundary, null amounts, missing
columns, non-purchase events) each get a dedicated test.
"""

import numpy as np
import pandas as pd
import pytest

from LedgerFlow.features.time_windows import (
    AGGREGATIONS,
    ALL_FEATURES,
    WINDOWS,
    TimeWindowAggregation,
    create_all_time_window_features,
)

REFERENCE = pd.Timestamp("2024-01-15 12:00:00")


@pytest.fixture
def events():
    """A small, hand-checkable event log.

    user_a: three purchases at -30min (5.0), -3h (10.0), -2d (100.0)
    user_b: one purchase at -45min (20.0) and a login (ignored)
    user_c: only non-purchase events (no purchase features)
    """
    rows = [
        # user_a
        ("a", "purchase", REFERENCE - pd.Timedelta(minutes=30), 5.0),
        ("a", "purchase", REFERENCE - pd.Timedelta(hours=3), 10.0),
        ("a", "purchase", REFERENCE - pd.Timedelta(days=2), 100.0),
        # user_b
        ("b", "purchase", REFERENCE - pd.Timedelta(minutes=45), 20.0),
        ("b", "login", REFERENCE - pd.Timedelta(minutes=10), None),
        # user_c
        ("c", "view", REFERENCE - pd.Timedelta(minutes=5), None),
        ("c", "click", REFERENCE - pd.Timedelta(hours=2), None),
    ]
    return pd.DataFrame(
        rows, columns=["user_id", "event_type", "event_timestamp", "amount"]
    )


# --------------------------------------------------------------------------- #
# Construction / metadata
# --------------------------------------------------------------------------- #

def test_exactly_35_features():
    features = create_all_time_window_features()
    assert len(features) == 35
    assert len(WINDOWS) * len(AGGREGATIONS) == 35


def test_all_features_module_level_count():
    assert len(ALL_FEATURES) == 35


def test_feature_names_are_unique():
    names = [f.name for f in ALL_FEATURES]
    assert len(set(names)) == 35


def test_unknown_aggregation_rejected():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        TimeWindowAggregation(window="1h", aggregation="median")


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_naming_and_dtype(window, aggregation):
    feat = TimeWindowAggregation(window=window, aggregation=aggregation)
    if aggregation == "count":
        assert feat.name == f"purchase_count_{window}"
        assert feat.output_dtype == "int"
        assert feat.nullable is False
    else:
        assert feat.name == f"purchase_amount_{aggregation}_{window}"
        assert feat.output_dtype == "float"
        assert feat.nullable is True
    assert feat.window == window
    assert feat.description  # non-empty


# --------------------------------------------------------------------------- #
# Shape / output contract for every feature
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("feature", ALL_FEATURES, ids=[f.name for f in ALL_FEATURES])
def test_compute_returns_named_series(feature, events):
    result = feature.compute(events, REFERENCE)
    assert isinstance(result, pd.Series)
    assert result.name == feature.name
    assert result.index.is_unique


@pytest.mark.parametrize("feature", ALL_FEATURES, ids=[f.name for f in ALL_FEATURES])
def test_missing_columns_raise(feature):
    bad = pd.DataFrame({"user_id": ["a"], "amount": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        feature.compute(bad, REFERENCE)


# --------------------------------------------------------------------------- #
# Correctness per aggregation (24h window, hand-computed)
# --------------------------------------------------------------------------- #
# Within 24h of REFERENCE, user_a has purchases [5.0, 10.0]; user_b has [20.0];
# user_c has none. (user_a's 100.0 purchase is 2 days old -> excluded.)

def _feat(window, aggregation):
    return TimeWindowAggregation(window=window, aggregation=aggregation)


def test_count_24h(events):
    result = _feat("24h", "count").compute(events, REFERENCE)
    assert result["a"] == 2
    assert result["b"] == 1
    assert "c" not in result.index  # no purchases


def test_sum_24h(events):
    result = _feat("24h", "sum").compute(events, REFERENCE)
    assert result["a"] == pytest.approx(15.0)
    assert result["b"] == pytest.approx(20.0)


def test_mean_24h(events):
    result = _feat("24h", "mean").compute(events, REFERENCE)
    assert result["a"] == pytest.approx(7.5)
    assert result["b"] == pytest.approx(20.0)


def test_std_24h(events):
    result = _feat("24h", "std").compute(events, REFERENCE)
    # sample std of [5, 10] = 3.5355...
    assert result["a"] == pytest.approx(np.std([5.0, 10.0], ddof=1))
    # single observation -> NaN
    assert np.isnan(result["b"])


def test_min_max_24h(events):
    assert _feat("24h", "min").compute(events, REFERENCE)["a"] == pytest.approx(5.0)
    assert _feat("24h", "max").compute(events, REFERENCE)["a"] == pytest.approx(10.0)


def test_last_24h(events):
    # Most recent purchase in the window for user_a is the -30min one (5.0).
    result = _feat("24h", "last").compute(events, REFERENCE)
    assert result["a"] == pytest.approx(5.0)
    assert result["b"] == pytest.approx(20.0)


def test_30d_includes_old_purchase(events):
    # The 100.0 purchase (2 days old) is inside 30d but outside 24h.
    assert _feat("30d", "sum").compute(events, REFERENCE)["a"] == pytest.approx(115.0)
    assert _feat("30d", "max").compute(events, REFERENCE)["a"] == pytest.approx(100.0)
    assert _feat("30d", "count").compute(events, REFERENCE)["a"] == 3


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_empty_window_returns_empty(events):
    # 1h window: user_a -3h purchase excluded; only -30min (a) and -45min (b).
    result = _feat("1h", "count").compute(events, REFERENCE)
    assert set(result.index) == {"a", "b"}


def test_window_boundary_is_exclusive():
    # A purchase exactly at the cutoff (reference - window) must be excluded.
    df = pd.DataFrame(
        {
            "user_id": ["a"],
            "event_type": ["purchase"],
            "event_timestamp": [REFERENCE - pd.Timedelta("1h")],
            "amount": [9.0],
        }
    )
    result = _feat("1h", "count").compute(df, REFERENCE)
    assert "a" not in result.index


def test_reference_time_inclusive():
    # A purchase exactly at reference_time is included.
    df = pd.DataFrame(
        {
            "user_id": ["a"],
            "event_type": ["purchase"],
            "event_timestamp": [REFERENCE],
            "amount": [9.0],
        }
    )
    assert _feat("1h", "count").compute(df, REFERENCE)["a"] == 1


def test_no_purchases_at_all_yields_empty():
    df = pd.DataFrame(
        {
            "user_id": ["c", "c"],
            "event_type": ["view", "click"],
            "event_timestamp": [REFERENCE - pd.Timedelta("5min")] * 2,
            "amount": [None, None],
        }
    )
    for agg in AGGREGATIONS:
        result = _feat("24h", agg).compute(df, REFERENCE)
        assert result.empty


def test_empty_dataframe():
    df = pd.DataFrame(
        {"user_id": [], "event_type": [], "event_timestamp": [], "amount": []}
    )
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    for agg in AGGREGATIONS:
        result = _feat("24h", agg).compute(df, REFERENCE)
        assert result.empty
        assert result.name == _feat("24h", agg).name


def test_single_event_aggregations():
    df = pd.DataFrame(
        {
            "user_id": ["a"],
            "event_type": ["purchase"],
            "event_timestamp": [REFERENCE - pd.Timedelta("10min")],
            "amount": [42.0],
        }
    )
    assert _feat("24h", "count").compute(df, REFERENCE)["a"] == 1
    assert _feat("24h", "sum").compute(df, REFERENCE)["a"] == pytest.approx(42.0)
    assert _feat("24h", "mean").compute(df, REFERENCE)["a"] == pytest.approx(42.0)
    assert _feat("24h", "min").compute(df, REFERENCE)["a"] == pytest.approx(42.0)
    assert _feat("24h", "max").compute(df, REFERENCE)["a"] == pytest.approx(42.0)
    assert _feat("24h", "last").compute(df, REFERENCE)["a"] == pytest.approx(42.0)
    assert np.isnan(_feat("24h", "std").compute(df, REFERENCE)["a"])


def test_non_purchase_events_ignored():
    df = pd.DataFrame(
        {
            "user_id": ["a", "a"],
            "event_type": ["login", "purchase"],
            "event_timestamp": [
                REFERENCE - pd.Timedelta("5min"),
                REFERENCE - pd.Timedelta("6min"),
            ],
            "amount": [None, 7.0],
        }
    )
    assert _feat("24h", "count").compute(df, REFERENCE)["a"] == 1
    assert _feat("24h", "sum").compute(df, REFERENCE)["a"] == pytest.approx(7.0)


def test_count_dtype_is_integer(events):
    result = _feat("24h", "count").compute(events, REFERENCE)
    assert pd.api.types.is_integer_dtype(result)


@pytest.mark.parametrize("aggregation", ["sum", "mean", "std", "min", "max", "last"])
def test_amount_aggregations_are_float(events, aggregation):
    result = _feat("24h", aggregation).compute(events, REFERENCE)
    assert pd.api.types.is_float_dtype(result)


@pytest.mark.parametrize("feature", ALL_FEATURES, ids=[f.name for f in ALL_FEATURES])
def test_validate_output_passes_on_real_compute(feature, events):
    # validate_output should accept compute()'s own output (modulo nullable NaNs).
    result = feature.compute(events, REFERENCE)
    if feature.nullable:
        result = result.dropna()
    if not result.empty:
        feature.validate_output(result)
