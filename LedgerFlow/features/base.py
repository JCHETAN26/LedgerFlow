"""
Base feature abstract class that all features must inherit from.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeature(ABC):
    """Abstract base class for all features in LedgerFlow.

    Every feature in the library inherits from this class and implements
    the `compute()` method.

    Attributes:
        name: Machine-readable name, e.g., "purchase_count_24h"
        description: Human-readable explanation of what the feature represents
        output_dtype: Expected data type of the output ("float", "int", or "bool")
        window: Time window for the feature, e.g., "1h", "24h", "7d"
        nullable: Whether null values are allowed in the output (default: False)
    """

    name: str
    description: str
    output_dtype: str
    window: str
    nullable: bool = False
    added_by: str = "ledgerflow"
    added_date: str = "2026-06-08"

    @abstractmethod
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        """Compute the feature for all users as of reference_time.

        Args:
            df: DataFrame containing event data with at minimum:
                - user_id: identifier for each user
                - event_timestamp: timestamp of each event
                - event_type: type of event (purchase, login, etc.)
                - amount: transaction amount (for purchase events)
            reference_time: The point in time as of which features are computed

        Returns:
            A pandas Series indexed by user_id, named self.name, containing
            the computed feature values for each user.

        Raises:
            ValueError: If required columns are missing from df
            TypeError: If data types are incorrect
        """
        pass

    def validate_output(self, result: pd.Series) -> None:
        """Validate the output of the compute method.

        Args:
            result: The Series returned by compute()

        Raises:
            AssertionError: If validation fails
        """
        assert result.name == self.name, (
            f"Series name {result.name} != expected {self.name}"
        )

        if not self.nullable:
            assert not result.isnull().any(), f"{self.name} has null values"

        # Check dtype based on output_dtype
        if self.output_dtype == "int":
            assert pd.api.types.is_integer_dtype(result), (
                f"{self.name} should be integer dtype"
            )
        elif self.output_dtype == "float":
            assert pd.api.types.is_float_dtype(result), (
                f"{self.name} should be float dtype"
            )
        elif self.output_dtype == "bool":
            assert pd.api.types.is_bool_dtype(result), (
                f"{self.name} should be boolean dtype"
            )
