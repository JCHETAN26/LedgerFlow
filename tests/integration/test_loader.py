"""
Integration tests for PostgreSQL data loader.

Note: These tests require a PostgreSQL database to run.
They are marked as integration tests and can be skipped if database is not available.
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Check if we should skip integration tests
skip_integration = not all(
    os.getenv(var) for var in
    ["LEDGERFLOW_DB_NAME", "LEDGERFLOW_DB_USER", "LEDGERFLOW_DB_PASSWORD"]
)


@pytest.mark.skipif(skip_integration, reason="Database environment variables not set")
class TestPostgresLoaderIntegration:
    """Integration tests requiring a real PostgreSQL database.

    Run them by pointing LEDGERFLOW_DB_* at a database (e.g. the Dockerized
    Postgres in docker-compose.yml). The class seeds its own ``event_logs``
    table with synthetic data, so the read paths are genuinely exercised.
    """

    @pytest.fixture(scope="class", autouse=True)
    def seed_database(self):
        """Create and populate the event_logs table with synthetic events."""
        from sqlalchemy import text

        from LedgerFlow.data.loader import get_database_connection
        from LedgerFlow.data.synthetic import generate_synthetic_events

        # End the synthetic window "now" so load_recent_events(days=...) finds rows.
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        events, _ = generate_synthetic_events(n_users=150, days=10, seed=99, end=end)

        engine = get_database_connection()
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS event_logs"))
        events.to_sql("event_logs", engine, index=False, if_exists="replace")
        # Store on the class so per-test instances can read it.
        type(self)._seeded_count = len(events)
        yield
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS event_logs"))

    def test_database_connection(self):
        """Test that we can connect to the database."""
        from sqlalchemy import text

        from LedgerFlow.data.loader import get_database_connection

        engine = get_database_connection()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_load_events_roundtrip_and_validate(self):
        """Full range load returns every seeded row and passes validation."""
        from LedgerFlow.data.loader import load_events_from_postgres
        from LedgerFlow.data.validators import validate_raw_events

        df = load_events_from_postgres(
            start_date=datetime.now() - timedelta(days=11),
            end_date=datetime.now() + timedelta(days=1),
            chunk_size=5000,
        )
        expected_columns = [
            "event_id", "user_id", "event_type",
            "event_timestamp", "amount", "session_id",
        ]
        for col in expected_columns:
            assert col in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"])
        assert len(df) == self._seeded_count
        # The real loaded data passes the same strict schema as ingestion.
        validate_raw_events(df)

    def test_load_recent_events(self):
        """Recent-window load returns a subset within the requested range."""
        from LedgerFlow.data.loader import load_recent_events

        df = load_recent_events(days=2, chunk_size=100)
        assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"])
        if not df.empty:
            start_date = datetime.now() - timedelta(days=2)
            assert df["event_timestamp"].min() >= start_date - timedelta(seconds=1)


class TestPostgresLoaderUnit:
    """Unit tests that mock database interactions."""

    def test_get_database_connection_missing_env(self):
        """Test that missing environment variables raise error."""
        from LedgerFlow.data.loader import get_database_connection

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LEDGERFLOW_DB_NAME"):
                get_database_connection()

    def test_missing_user_raises(self):
        from LedgerFlow.data.loader import get_database_connection

        env = {"LEDGERFLOW_DB_NAME": "db"}  # user & password absent
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="LEDGERFLOW_DB_USER"):
                get_database_connection()

    def test_missing_password_raises(self):
        from LedgerFlow.data.loader import get_database_connection

        env = {"LEDGERFLOW_DB_NAME": "db", "LEDGERFLOW_DB_USER": "u"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="LEDGERFLOW_DB_PASSWORD"):
                get_database_connection()

    def test_empty_result_warns_and_returns_empty(self):
        from LedgerFlow.data.loader import load_events_from_postgres

        mock_engine = MagicMock()
        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = iter([])  # no chunks
            with patch(
                "LedgerFlow.data.loader.get_database_connection",
                return_value=mock_engine,
            ):
                result = load_events_from_postgres(table_name="t")
        assert result.empty

    def test_get_database_connection_success(self):
        """Test successful database connection creation."""
        from LedgerFlow.data.loader import get_database_connection

        test_env = {
            "LEDGERFLOW_DB_NAME": "test_db",
            "LEDGERFLOW_DB_USER": "test_user",
            "LEDGERFLOW_DB_PASSWORD": "test_pass",
            "LEDGERFLOW_DB_HOST": "localhost",
            "LEDGERFLOW_DB_PORT": "5432",
        }

        with patch.dict(os.environ, test_env):
            with patch("LedgerFlow.data.loader.create_engine") as mock_create_engine:
                mock_engine = Mock()
                mock_create_engine.return_value = mock_engine

                engine = get_database_connection()

                # Verify engine was created with correct URL
                mock_create_engine.assert_called_once()
                call_args = mock_create_engine.call_args[0][0]
                assert "postgresql://test_user:test_pass@localhost:5432/test_db" in call_args
                assert engine == mock_engine

    def test_load_events_from_postgres_mocked(self):
        """Test loading events with mocked database."""
        from LedgerFlow.data.loader import load_events_from_postgres

        # Create mock data
        mock_data = pd.DataFrame({
            "event_id": ["evt_001", "evt_002"],
            "user_id": ["user_001", "user_002"],
            "event_type": ["purchase", "login"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00", "2024-01-15 10:31:00"]),
            "amount": [25.50, None],
            "session_id": ["sess_001", "sess_002"],
        })

        # Mock engine and connection (MagicMock supports the `with` protocol)
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        # Mock pandas read_sql to return our test data as a single chunk
        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = iter([mock_data])

            with patch("LedgerFlow.data.loader.get_database_connection") as mock_get_conn:
                mock_get_conn.return_value = mock_engine

                # Call function
                result = load_events_from_postgres(
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 31),
                    user_ids=["user_001", "user_002"],
                    table_name="test_events",
                )

                # Verify result
                pd.testing.assert_frame_equal(result, mock_data)

                # Verify read_sql was called with correct query
                mock_read_sql.assert_called_once()
                query = mock_read_sql.call_args[0][0]
                assert "test_events" in str(query)

    def test_load_recent_events_mocked(self):
        """Test load_recent_events with mocked dependencies."""
        from LedgerFlow.data.loader import load_recent_events

        # Mock data
        mock_data = pd.DataFrame({
            "event_id": ["evt_001"],
            "user_id": ["user_001"],
            "event_type": ["purchase"],
            "event_timestamp": pd.to_datetime(["2024-01-15 10:30:00"]),
            "amount": [25.50],
            "session_id": ["sess_001"],
        })

        with patch("LedgerFlow.data.loader.load_events_from_postgres") as mock_load:
            mock_load.return_value = mock_data

            with patch("LedgerFlow.data.loader.datetime") as mock_datetime:
                # Mock current time
                mock_now = datetime(2024, 1, 15, 12, 0, 0)
                mock_datetime.now.return_value = mock_now

                # Call function
                result = load_recent_events(days=7)

                # Verify load_events_from_postgres was called with correct dates
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args[1]

                assert call_kwargs["start_date"] == datetime(2024, 1, 8, 12, 0, 0)  # 7 days before
                assert call_kwargs["end_date"] == mock_now

                # Verify result
                pd.testing.assert_frame_equal(result, mock_data)

    def test_load_recent_events_with_user_limit(self):
        """Test limiting to top users."""
        from LedgerFlow.data.loader import load_recent_events

        # Create mock data with multiple users
        mock_data = pd.DataFrame({
            "event_id": [f"evt_{i:03d}" for i in range(10)],
            "user_id": ["user_001"] * 5 + ["user_002"] * 3 + ["user_003"] * 2,
            "event_type": ["purchase"] * 10,
            "event_timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "amount": [10.0] * 10,
            "session_id": [f"sess_{i:03d}" for i in range(10)],
        })

        with patch("LedgerFlow.data.loader.load_events_from_postgres") as mock_load:
            mock_load.return_value = mock_data

            # Call with max_users=2 (should keep user_001 and user_002)
            result = load_recent_events(days=7, max_users=2)

            # Verify we only have users 001 and 002
            unique_users = result["user_id"].unique()
            assert set(unique_users) == {"user_001", "user_002"}
            assert "user_003" not in unique_users

    def test_error_handling(self):
        """Test error handling in loader functions."""
        from LedgerFlow.data.loader import get_database_connection

        test_env = {
            "LEDGERFLOW_DB_NAME": "test_db",
            "LEDGERFLOW_DB_USER": "test_user",
            "LEDGERFLOW_DB_PASSWORD": "test_pass",
        }

        with patch.dict(os.environ, test_env):
            with patch("LedgerFlow.data.loader.create_engine") as mock_create_engine:
                mock_create_engine.side_effect = Exception("Connection failed")

                with pytest.raises(Exception, match="Connection failed"):
                    get_database_connection()


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys

    # Set test environment variables
    os.environ["LEDGERFLOW_DB_NAME"] = "test"
    os.environ["LEDGERFLOW_DB_USER"] = "test"
    os.environ["LEDGERFLOW_DB_PASSWORD"] = "test"

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))
