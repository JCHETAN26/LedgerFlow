"""
Unit tests for the BaseFeature abstract class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from LedgerFlow.features.base import BaseFeature


class SampleFeature(BaseFeature):
    """Concrete implementation for testing."""
    
    def __init__(self):
        self.name = "test_feature"
        self.description = "Test feature for unit tests"
        self.output_dtype = "float"
        self.window = "1h"
        self.nullable = False
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        # Simple implementation: count events per user
        return df.groupby("user_id").size().rename(self.name).astype(float)


def test_base_feature_abstract_method():
    """Test that BaseFeature cannot be instantiated directly."""
    with pytest.raises(TypeError):
        feature = BaseFeature()


def test_concrete_feature_implementation():
    """Test that a concrete feature can be instantiated and used."""
    feature = SampleFeature()
    
    # Test attributes
    assert feature.name == "test_feature"
    assert feature.description == "Test feature for unit tests"
    assert feature.output_dtype == "float"
    assert feature.window == "1h"
    assert feature.nullable is False
    
    # Test compute method with sample data
    df = pd.DataFrame({
        "user_id": ["user1", "user1", "user2", "user3"],
        "event_timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
        "event_type": ["purchase", "login", "purchase", "view"],
        "amount": [10.0, None, 20.0, None],
    })
    
    reference_time = pd.Timestamp("2024-01-01 04:00:00")
    result = feature.compute(df, reference_time)
    
    # Test result properties
    assert isinstance(result, pd.Series)
    assert result.name == "test_feature"
    assert len(result) == 3  # 3 unique users
    assert result["user1"] == 2.0
    assert result["user2"] == 1.0
    assert result["user3"] == 1.0


def test_validate_output():
    """Test the validate_output method."""
    feature = SampleFeature()
    
    # Valid output
    valid_series = pd.Series(
        [1.0, 2.0, 3.0], 
        index=["user1", "user2", "user3"],
        name="test_feature"
    )
    feature.validate_output(valid_series)
    
    # Invalid: wrong name
    wrong_name_series = pd.Series(
        [1.0, 2.0, 3.0],
        index=["user1", "user2", "user3"],
        name="wrong_name"
    )
    with pytest.raises(AssertionError):
        feature.validate_output(wrong_name_series)
    
    # Invalid: null values when nullable=False
    null_series = pd.Series(
        [1.0, np.nan, 3.0],
        index=["user1", "user2", "user3"],
        name="test_feature"
    )
    with pytest.raises(AssertionError):
        feature.validate_output(null_series)


def test_nullable_feature():
    """Test a feature with nullable=True."""
    
    class NullableFeature(SampleFeature):
        def __init__(self):
            super().__init__()
            self.nullable = True
    
    feature = NullableFeature()
    
    # Should accept null values
    null_series = pd.Series(
        [1.0, np.nan, 3.0],
        index=["user1", "user2", "user3"],
        name="test_feature"
    )
    feature.validate_output(null_series)  # Should not raise


def test_feature_with_different_dtype():
    """Test features with different output dtypes."""
    
    class IntFeature(BaseFeature):
        def __init__(self):
            self.name = "int_feature"
            self.description = "Integer feature"
            self.output_dtype = "int"
            self.window = "1h"
            self.nullable = False
        
        def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
            return pd.Series([1, 2, 3], index=["a", "b", "c"], name=self.name)
    
    class BoolFeature(BaseFeature):
        def __init__(self):
            self.name = "bool_feature"
            self.description = "Boolean feature"
            self.output_dtype = "bool"
            self.window = "1h"
            self.nullable = False
        
        def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
            return pd.Series([True, False, True], index=["a", "b", "c"], name=self.name)
    
    int_feature = IntFeature()
    bool_feature = BoolFeature()
    
    int_result = int_feature.compute(pd.DataFrame(), pd.Timestamp.now())
    bool_result = bool_feature.compute(pd.DataFrame(), pd.Timestamp.now())
    
    int_feature.validate_output(int_result)
    bool_feature.validate_output(bool_result)
    
    assert pd.api.types.is_integer_dtype(int_result)
    assert pd.api.types.is_bool_dtype(bool_result)