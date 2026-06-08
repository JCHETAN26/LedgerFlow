#!/usr/bin/env python3
"""
Test script for Phase 1: Data Ingestion & Validation.
This tests the complete pipeline without requiring actual database access.
"""

import os
import sys
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_schema_validation():
    """Test Pandera schema validation."""
    print("=== Testing Schema Validation ===")
    
    from LedgerFlow.data.validators import RawEventSchema, validate_raw_events
    
    # Create valid test data
    df = pd.DataFrame({
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
    
    print(f"Test data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    try:
        validated = validate_raw_events(df)
        print("✅ Schema validation passed")
        print(f"   Validated rows: {len(validated)}")
        return True
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False


def test_loader_mocked():
    """Test loader with mocked database."""
    print("\n=== Testing Data Loader (Mocked) ===")
    
    # Mock environment variables
    test_env = {
        "LEDGERFLOW_DB_NAME": "test_db",
        "LEDGERFLOW_DB_USER": "test_user",
        "LEDGERFLOW_DB_PASSWORD": "test_pass",
    }
    
    original_env = {}
    for key in test_env:
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = test_env[key]
    
    try:
        from LedgerFlow.data.loader import get_database_connection
        
        # Mock sqlalchemy
        import unittest.mock as mock
        
        with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
            mock_engine = mock.Mock()
            mock_create_engine.return_value = mock_engine
            
            engine = get_database_connection()
            
            # Verify correct connection URL was used
            call_args = mock_create_engine.call_args[0][0]
            assert "postgresql://test_user:test_pass" in call_args
            assert "test_db" in call_args
            
            print("✅ Database connection creation passed")
            return True
            
    except Exception as e:
        print(f"❌ Loader test failed: {e}")
        return False
    finally:
        # Restore environment variables
        for key in test_env:
            if key in original_env:
                os.environ[key] = original_env[key]
            else:
                del os.environ[key]


def test_ingestion_pipeline():
    """Test complete ingestion pipeline with mocked data."""
    print("\n=== Testing Complete Ingestion Pipeline ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "data" / "raw"
        
        # Create mock data
        mock_data = pd.DataFrame({
            "event_id": [f"evt_{i:03d}" for i in range(100)],
            "user_id": [f"user_{(i % 10):03d}" for i in range(100)],
            "event_type": ["purchase"] * 40 + ["login"] * 30 + ["view"] * 20 + ["click"] * 10,
            "event_timestamp": pd.date_range("2024-01-01", periods=100, freq="H"),
            "amount": [10.0 * (i % 5) if i < 40 else None for i in range(100)],
            "session_id": [f"sess_{(i // 5):03d}" for i in range(100)],
        })
        
        # Mock the loader to return our test data
        import unittest.mock as mock
        
        with mock.patch("LedgerFlow.data.ingest.load_recent_events") as mock_load:
            mock_load.return_value = mock_data
            
            from LedgerFlow.data.ingest import ingest_raw_events
            
            try:
                parquet_path = ingest_raw_events(
                    output_dir=str(output_dir),
                    days=7,
                    max_users=5,
                    create_validation_report=True,
                )
                
                print(f"✅ Ingestion pipeline passed")
                print(f"   Output directory: {output_dir}")
                print(f"   Parquet file: {parquet_path}")
                
                # Verify files were created
                assert Path(parquet_path).exists()
                
                # List created files
                files = list(output_dir.glob("*"))
                print(f"   Created files: {len(files)}")
                for f in files:
                    print(f"     - {f.name}")
                
                return True
                
            except Exception as e:
                print(f"❌ Ingestion pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                return False


def test_business_rules():
    """Test business rule validation."""
    print("\n=== Testing Business Rules ===")
    
    from LedgerFlow.data.validators import _validate_business_rules
    
    # Test 1: Valid data
    valid_df = pd.DataFrame({
        "event_id": ["evt_001", "evt_002"],
        "user_id": ["user_001", "user_002"],
        "event_type": ["purchase", "login"],
        "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00", "2024-01-15 10:31:00"]),
        "amount": [25.50, None],
        "session_id": ["sess_001", "sess_002"],
    })
    
    try:
        _validate_business_rules(valid_df)
        print("✅ Valid data passes business rules")
    except Exception as e:
        print(f"❌ Valid data failed business rules: {e}")
        return False
    
    # Test 2: Purchase without amount
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "amount"] = None  # Purchase without amount
    
    try:
        _validate_business_rules(invalid_df)
        print("❌ Purchase without amount should fail")
        return False
    except ValueError as e:
        print("✅ Purchase without amount correctly fails")
    
    return True


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1: Data Ingestion & Validation - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Schema Validation", test_schema_validation),
        ("Data Loader (Mocked)", test_loader_mocked),
        ("Business Rules", test_business_rules),
        ("Complete Pipeline", test_ingestion_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All Phase 1 tests passed!")
        print("\nNext steps:")
        print("1. Set up actual PostgreSQL database with event logs")
        print("2. Configure environment variables:")
        print("   - LEDGERFLOW_DB_NAME")
        print("   - LEDGERFLOW_DB_USER")
        print("   - LEDGERFLOW_DB_PASSWORD")
        print("3. Run: python -m LedgerFlow.data.ingest")
        print("4. Track with DVC: dvc add data/raw/")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()