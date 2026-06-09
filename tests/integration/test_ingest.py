"""
Integration tests for the PostgreSQL -> Parquet ingestion orchestrator.

The database query itself is mocked; these tests exercise the real
validate -> write-Parquet -> report logic that ingest_raw_events performs.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from LedgerFlow.data.ingest import ingest_raw_events, run_ingestion_pipeline


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"evt_{i:03d}" for i in range(6)],
            "user_id": ["user_001", "user_002", "user_001", "user_003", "user_002", "user_001"],
            "event_type": ["purchase", "login", "view", "purchase", "click", "purchase"],
            "event_timestamp": pd.to_datetime(
                [
                    "2024-01-15 10:30:00",
                    "2024-01-15 10:31:00",
                    "2024-01-15 10:32:00",
                    "2024-01-15 10:33:00",
                    "2024-01-15 10:34:00",
                    "2024-01-15 10:35:00",
                ]
            ),
            "amount": [25.5, None, None, 100.0, None, 12.0],
            "session_id": [f"sess_{i:03d}" for i in range(6)],
        }
    )


def test_ingest_raw_events_writes_parquet_and_report(tmp_path):
    with patch("LedgerFlow.data.ingest.load_recent_events", return_value=_valid_events()):
        parquet_path = ingest_raw_events(
            output_dir=str(tmp_path), days=1, create_validation_report=True
        )

    assert Path(parquet_path).exists()
    # Latest snapshot + a validation report + a schema doc are produced.
    assert (tmp_path / "events_latest.parquet").exists()
    assert any(tmp_path.glob("validation_report_*.json"))
    assert any(tmp_path.glob("parquet_schema_*.txt"))

    written = pd.read_parquet(tmp_path / "events_latest.parquet")
    assert len(written) == 6


def test_ingest_raw_events_empty_raises(tmp_path):
    empty = _valid_events().iloc[0:0]
    with patch("LedgerFlow.data.ingest.load_recent_events", return_value=empty):
        with pytest.raises(ValueError, match="No data loaded"):
            ingest_raw_events(output_dir=str(tmp_path))


def test_run_ingestion_pipeline_success(tmp_path):
    with patch("LedgerFlow.data.ingest.load_recent_events", return_value=_valid_events()):
        result = run_ingestion_pipeline(
            {"output_dir": str(tmp_path), "days": 1, "create_validation_report": False}
        )
    assert result["status"] == "success"
    assert "duration_seconds" in result


def test_run_ingestion_pipeline_failure(tmp_path):
    with patch(
        "LedgerFlow.data.ingest.load_recent_events",
        side_effect=RuntimeError("db down"),
    ):
        result = run_ingestion_pipeline({"output_dir": str(tmp_path)})
    assert result["status"] == "failed"
    assert "db down" in result["error"]
