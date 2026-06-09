"""
Data validation schemas for LedgerFlow using Pandera.

This module defines strict schemas for validating raw and processed data
to ensure data quality before feature engineering.
"""

import logging

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)


# Raw event log schema
RawEventSchema = DataFrameSchema({
    "event_id": Column(
        str,
        nullable=False,
        unique=True,
        description="Unique identifier for each event",
        checks=[
            Check.str_length(min_value=1, max_value=100),
            Check.str_matches(
                r"^[a-zA-Z0-9_-]+$",
                error="event_id contains invalid characters",
            ),
        ]
    ),
    "user_id": Column(
        str,
        nullable=False,
        description="Identifier for the user who generated the event",
        checks=[
            Check.str_length(min_value=1, max_value=50),
            Check.str_matches(
                r"^[a-zA-Z0-9_-]+$",
                error="user_id contains invalid characters",
            ),
        ]
    ),
    "event_type": Column(
        str,
        nullable=False,
        description="Type of event",
        checks=[
            Check.isin(["purchase", "login", "view", "click"]),
        ]
    ),
    "event_timestamp": Column(
        pd.Timestamp,
        nullable=False,
        description="Timestamp when the event occurred",
        checks=[
            Check.greater_than_or_equal_to(pd.Timestamp("2020-01-01")),
            Check.less_than_or_equal_to(pd.Timestamp("2030-12-31")),
        ]
    ),
    "amount": Column(
        float,
        nullable=True,
        description="Transaction amount (only for purchase events)",
        checks=[
            Check.greater_than_or_equal_to(0, ignore_na=True),
            # Reasonable upper bound on a single transaction.
            Check.less_than_or_equal_to(1_000_000, ignore_na=True),
        ]
    ),
    "session_id": Column(
        str,
        nullable=True,
        description="Session identifier for the event",
        checks=[
            Check.str_length(min_value=1, max_value=100, ignore_na=True),
        ]
    ),
},
strict=True,  # Reject columns not in schema
coerce=True,  # Try to coerce types
)


def validate_raw_events(
    df: pd.DataFrame,
    schema: DataFrameSchema = RawEventSchema,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Validate raw event data against schema.

    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against (default: RawEventSchema)
        fail_fast: If True, raise on the first error; otherwise collect all errors.

    Returns:
        Validated DataFrame (with type coercion applied)

    Raises:
        pa.errors.SchemaError: If validation fails
    """
    logger.info(f"Validating {len(df)} events against schema")

    try:
        # Validate the DataFrame
        validated_df = schema.validate(df, lazy=not fail_fast)

        # Additional business logic validations
        _validate_business_rules(validated_df)

        logger.info("✅ Data validation passed")
        return validated_df

    except pa.errors.SchemaError as e:
        logger.error(f"❌ Data validation failed: {e}")

        # Log detailed error information
        if hasattr(e, 'schema_errors'):
            for error in e.schema_errors:
                logger.error(f"  - {error}")

        raise


def _validate_business_rules(df: pd.DataFrame) -> None:
    """Validate business rules that aren't captured in the schema.

    Args:
        df: Validated DataFrame

    Raises:
        ValueError: If business rules are violated
    """
    # Rule 1: Purchase events must have amount
    purchase_mask = df["event_type"] == "purchase"
    purchase_without_amount = df[purchase_mask & df["amount"].isna()]

    if not purchase_without_amount.empty:
        error_msg = (
            f"Found {len(purchase_without_amount)} purchase events without amount"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Rule 2: Non-purchase events should not have amount (or should have null)
    non_purchase_with_amount = df[~purchase_mask & df["amount"].notna()]

    if not non_purchase_with_amount.empty:
        error_msg = (
            f"Found {len(non_purchase_with_amount)} non-purchase events with amount"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Rule 3: Check for duplicate event_ids (schema should catch this, but double-check)
    duplicates = df[df["event_id"].duplicated()]
    if not duplicates.empty:
        error_msg = f"Found {len(duplicates)} duplicate event_ids"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Rule 4: Check timestamp ordering (optional, can be expensive for large datasets)
    if len(df) > 0:
        time_span = df["event_timestamp"].max() - df["event_timestamp"].min()
        if time_span.days > 365 * 5:  # 5 years
            logger.warning(
                f"Data spans {time_span.days} days - "
                "consider filtering to a smaller time range"
            )

    logger.debug("Business rule validation passed")


def generate_validation_report(
    df: pd.DataFrame, schema: DataFrameSchema = RawEventSchema
) -> dict:
    """Generate a detailed validation report without raising exceptions.

    Args:
        df: DataFrame to validate
        schema: Schema to validate against

    Returns:
        Dictionary with validation results and statistics
    """
    report = {
        "total_events": len(df),
        "validation_passed": False,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    try:
        # Try validation
        schema.validate(df, lazy=True)
        report["validation_passed"] = True

    except pa.errors.SchemaErrors as e:
        report["errors"] = [str(error) for error in e.schema_errors]

    # Gather statistics
    if not df.empty:
        ts_min = df["event_timestamp"].min()
        ts_max = df["event_timestamp"].max()
        purchases = df[df["event_type"] == "purchase"]
        has_amount = "amount" in df.columns
        report["statistics"] = {
            "date_range": {
                "min": ts_min.isoformat() if pd.notna(ts_min) else None,
                "max": ts_max.isoformat() if pd.notna(ts_max) else None,
            },
            "user_count": df["user_id"].nunique(),
            "event_type_distribution": df["event_type"].value_counts().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "purchase_stats": {
                "count": len(purchases),
                "total_amount": purchases["amount"].sum() if has_amount else None,
                "avg_amount": purchases["amount"].mean() if has_amount else None,
            } if has_amount else {},
        }

    return report


def save_validation_report(report: dict, output_path: str) -> None:
    """Save validation report to JSON file.

    Args:
        report: Validation report dictionary
        output_path: Path to save JSON file
    """
    import json
    from datetime import datetime

    # Add metadata
    report_with_metadata = {
        "generated_at": datetime.now().isoformat(),
        "schema_version": "1.0.0",
        **report,
    }

    with open(output_path, "w") as f:
        json.dump(report_with_metadata, f, indent=2, default=str)

    logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import json
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=== Pandera Schema Validation Test ===")

    # Create sample data
    sample_data = pd.DataFrame({
        "event_id": ["evt_001", "evt_002", "evt_003", "evt_004"],
        "user_id": ["user_001", "user_002", "user_001", "user_003"],
        "event_type": ["purchase", "login", "view", "purchase"],
        "event_timestamp": pd.to_datetime([
            "2024-01-15 10:30:00",
            "2024-01-15 10:31:00",
            "2024-01-15 10:32:00",
            "2024-01-15 10:33:00",
        ]),
        "amount": [25.50, None, None, 100.00],
        "session_id": ["sess_001", "sess_002", "sess_001", "sess_003"],
    })

    print(f"Sample data:\n{sample_data}")

    try:
        # Test validation
        validated = validate_raw_events(sample_data)
        print("\n✅ Validation passed!")
        print(f"Validated data:\n{validated}")

        # Test validation report
        report = generate_validation_report(sample_data)
        print(f"\nValidation report: {json.dumps(report, indent=2)}")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
