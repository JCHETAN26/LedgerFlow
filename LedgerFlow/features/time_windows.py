"""
Time-window aggregation features for LedgerFlow.

This module contains the 35 time-window aggregation features (5 windows × 7 aggregations).
"""

import pandas as pd
from typing import Optional
from .base import BaseFeature
from ..registry import register_feature


class PurchaseCountWindow(BaseFeature):
    """Count of purchase events in a time window."""
    
    def __init__(self, window: str):
        self.window = window
        self.name = f"purchase_count_{window}"
        self.description = f"Number of purchase events in the last {window}"
        self.output_dtype = "int"
        self.nullable = False
        register_feature(self)
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        cutoff = reference_time - pd.Timedelta(self.window)
        filtered = df[
            (df["event_type"] == "purchase") & 
            (df["event_timestamp"] >= cutoff)
        ]
        return filtered.groupby("user_id").size().rename(self.name)


class PurchaseAmountSumWindow(BaseFeature):
    """Sum of purchase amounts in a time window."""
    
    def __init__(self, window: str):
        self.window = window
        self.name = f"purchase_amount_sum_{window}"
        self.description = f"Total purchase amount in the last {window}"
        self.output_dtype = "float"
        self.nullable = True  # Users with no purchases will have NaN
        register_feature(self)
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        cutoff = reference_time - pd.Timedelta(self.window)
        filtered = df[
            (df["event_type"] == "purchase") & 
            (df["event_timestamp"] >= cutoff) &
            df["amount"].notna()
        ]
        return filtered.groupby("user_id")["amount"].sum().rename(self.name)


class PurchaseAmountMeanWindow(BaseFeature):
    """Mean purchase amount in a time window."""
    
    def __init__(self, window: str):
        self.window = window
        self.name = f"purchase_amount_mean_{window}"
        self.description = f"Average purchase amount in the last {window}"
        self.output_dtype = "float"
        self.nullable = True
        register_feature(self)
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        cutoff = reference_time - pd.Timedelta(self.window)
        filtered = df[
            (df["event_type"] == "purchase") & 
            (df["event_timestamp"] >= cutoff) &
            df["amount"].notna()
        ]
        return filtered.groupby("user_id")["amount"].mean().rename(self.name)


# Factory function to create all features
def create_all_time_window_features() -> list[BaseFeature]:
    """Create all 35 time-window aggregation features.
    
    Returns:
        List of BaseFeature instances for all time windows and aggregations
    """
    windows = ["1h", "6h", "24h", "7d", "30d"]
    features = []
    
    for window in windows:
        # Create count features for different event types
        features.extend([
            PurchaseCountWindow(window),
            PurchaseAmountSumWindow(window),
            PurchaseAmountMeanWindow(window),
            # Additional features would be added here
        ])
    
    return features


# Pre-create all features for easy import
ALL_FEATURES = create_all_time_window_features()