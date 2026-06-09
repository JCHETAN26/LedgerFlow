"""
Unit tests for the feature registry / catalog.
"""

import pytest

# Importing the features module populates the registry as a side effect.
from LedgerFlow.features.time_windows import ALL_FEATURES  # noqa: F401
from LedgerFlow.registry import (
    FEATURE_REGISTRY,
    export_to_markdown,
    get_feature_info,
    list_features,
    register_feature,
    validate_registry,
)


def test_registry_has_all_35_features():
    for feature in ALL_FEATURES:
        assert feature.name in FEATURE_REGISTRY


def test_every_feature_has_nonempty_description():
    for name, info in list_features().items():
        assert info["description"], f"{name} has an empty description"


def test_validate_registry_passes():
    assert validate_registry() is True


def test_registry_entries_have_required_metadata():
    required = {"description", "window", "dtype", "added_by", "added_date"}
    for name, info in FEATURE_REGISTRY.items():
        assert required <= set(info.keys()), f"{name} missing metadata"


def test_get_feature_info_known():
    info = get_feature_info("purchase_count_24h")
    assert info["window"] == "24h"
    assert info["dtype"] == "int"


def test_get_feature_info_unknown_raises():
    with pytest.raises(KeyError):
        get_feature_info("does_not_exist")


def test_added_date_is_deterministic():
    # All time-window features share the same deterministic added_date so the
    # exported catalog is reproducible in CI.
    dates = {FEATURE_REGISTRY[f.name]["added_date"] for f in ALL_FEATURES}
    assert dates == {"2026-06-08"}


def test_export_to_markdown_contains_table():
    md = export_to_markdown()
    assert "| Feature |" in md
    assert "`purchase_count_24h`" in md
    # header + separator + 35 rows (at least)
    assert md.count("\n") >= 36


def test_register_feature_idempotent():
    feature = ALL_FEATURES[0]
    before = len(FEATURE_REGISTRY)
    register_feature(feature)
    assert len(FEATURE_REGISTRY) == before
