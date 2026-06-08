"""
Unit tests for Pandera data validators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tempfile
import os

from LedgerFlow.data.validators import (
    RawEventSchema,
    validate_raw_events,
    generate_validation_report,
    save_validation_report,
    _validate_business_rules,
)


class TestRawEventSchema:
    """Tests for the RawEventSchema."""
    
    def test_valid_data_passes(self):
        """Test that valid data passes schema validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_002", "evt_003"],
            "user_id": ["user_001", "user_002", "user_003"],
            "event_type": ["purchase", "login", "view"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
                "2024-01-15 10:32:00",
            ]),
            "amount": [25.50, None, None],
            "session_id": ["sess_001", "sess_002", "sess_003"],
        })
        
        validated = RawEventSchema.validate(df)
        assert len(validated) == 3
        assert validated["event_id"].iloc[0] == "evt_001"
    
    def test_missing_required_column_fails(self):
        """Test that missing required column fails validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            # Missing event_type
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(Exception):
            RawEventSchema.validate(df)
    
    def test_invalid_event_type_fails(self):
        """Test that invalid event_type fails validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["invalid_type"],  # Not in allowed list
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(Exception):
            RawEventSchema.validate(df)
    
    def test_negative_amount_fails(self):
        """Test that negative amount fails validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [-10.0],  # Negative amount
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(Exception):
            RawEventSchema.validate(df)
    
    def test_duplicate_event_id_fails(self):
        """Test that duplicate event_id fails validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_001"],  # Duplicate
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        with pytest.raises(Exception):
            RawEventSchema.validate(df)
    
    def test_type_coercion_works(self):
        """Test that type coercion works correctly."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": ["2024-01-15 10:30:00"],  # String instead of datetime
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        validated = RawEventSchema.validate(df)
        assert pd.api.types.is_datetime64_any_dtype(validated["event_timestamp"])
    
    def test_strict_schema_rejects_extra_columns(self):
        """Test that strict schema rejects columns not in schema."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
            "extra_column": ["should_fail"],  # Extra column not in schema
        })
        
        with pytest.raises(Exception):
            RawEventSchema.validate(df)


class TestValidateRawEvents:
    """Tests for the validate_raw_events function."""
    
    def test_validate_raw_events_success(self):
        """Test successful validation."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_002"],
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        validated = validate_raw_events(df)
        assert len(validated) == 2
    
    def test_validate_raw_events_fail_fast(self):
        """Test fail_fast parameter."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_002"],
            "user_id": ["user_001", "user_002"],
            "event_type": ["invalid", "invalid"],  # Both invalid
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        # With fail_fast=True, should raise on first error
        with pytest.raises(Exception):
            validate_raw_events(df, fail_fast=True)
    
    def test_business_rule_purchase_without_amount(self):
        """Test business rule: purchase events must have amount."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [None],  # Purchase without amount
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(ValueError, match="purchase events without amount"):
            validate_raw_events(df)
    
    def test_business_rule_non_purchase_with_amount(self):
        """Test business rule: non-purchase events should not have amount."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["login"],  # Non-purchase
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],  # Non-purchase with amount
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(ValueError, match="non-purchase events with amount"):
            validate_raw_events(df)


class TestBusinessRules:
    """Tests for business rule validation function."""
    
    def test_purchase_without_amount(self):
        """Test purchase events without amount."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [None],
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(ValueError, match="purchase events without amount"):
            _validate_business_rules(df)
    
    def test_non_purchase_with_amount(self):
        """Test non-purchase events with amount."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["login"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        with pytest.raises(ValueError, match="non-purchase events with amount"):
            _validate_business_rules(df)
    
    def test_duplicate_event_ids(self):
        """Test duplicate event_ids."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_001"],  # Duplicate
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        with pytest.raises(ValueError, match="duplicate event_ids"):
            _validate_business_rules(df)
    
    def test_valid_data_passes_business_rules(self):
        """Test that valid data passes business rules."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_002"],
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        # Should not raise
        _validate_business_rules(df)


class TestValidationReport:
    """Tests for validation report generation."""
    
    def test_generate_validation_report_success(self):
        """Test generating validation report for valid data."""
        df = pd.DataFrame({
            "event_id": ["evt_001", "evt_002"],
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 10:31:00",
            ]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })
        
        report = generate_validation_report(df)
        
        assert report["total_events"] == 2
        assert report["validation_passed"] == True
        assert len(report["errors"]) == 0
        assert "statistics" in report
        
        stats = report["statistics"]
        assert stats["user_count"] == 2
        assert stats["event_type_distribution"] == {"purchase": 1, "login": 1}
        assert stats["purchase_stats"]["count"] == 1
        assert stats["purchase_stats"]["total_amount"] == 25.50
    
    def test_generate_validation_report_with_errors(self):
        """Test generating validation report for invalid data."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["invalid_type"],  # Invalid
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        report = generate_validation_report(df)
        
        assert report["total_events"] == 1
        assert report["validation_passed"] == False
        assert len(report["errors"]) > 0
    
    def test_save_validation_report(self):
        """Test saving validation report to file."""
        df = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })
        
        report = generate_validation_report(df)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            save_validation_report(report, temp_path)
            
            # Verify file was created and contains JSON
            with open(temp_path, "r") as f:
                saved_data = json.load(f)
            
            assert saved_data["total_events"] == 1
            assert saved_data["validation_passed"] == True
            assert "generated_at" in saved_data
            assert "schema_version" in saved_data
            
        finally:
            os.unlink(temp_path)


def test_schema_documentation():
    """Test that schema has proper documentation."""
    assert RawEventSchema.columns["event_id"].description == "Unique identifier for each event"
    assert RawEventSchema.columns["event_type"].description == "Type of event"
    assert RawEventSchema.columns["event_timestamp"].description == "Timestamp when the event occurred"
    
    # Check that event_type has allowed values
    event_type_checks = RawEventSchema.columns["event_type"].checks
    assert any("isin" in str(check).lower() for check in event_type_checks)


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys
    sys.exit(pytest.main([__file__, "-v"]))