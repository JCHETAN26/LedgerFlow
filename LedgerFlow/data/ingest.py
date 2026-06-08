"""
Data ingestion pipeline for LedgerFlow.

This module orchestrates the complete data ingestion process:
1. Load data from PostgreSQL
2. Validate with Pandera schema
3. Write to Parquet format
4. Track with DVC
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging
from typing import Optional

from .loader import load_events_from_postgres, load_recent_events
from .validators import validate_raw_events, generate_validation_report, save_validation_report

logger = logging.getLogger(__name__)


def ingest_raw_events(
    output_dir: str = "data/raw",
    days: int = 30,
    max_users: Optional[int] = None,
    table_name: str = "event_logs",
    date_column: str = "event_timestamp",
    user_column: str = "user_id",
    create_validation_report: bool = True,
) -> str:
    """Complete data ingestion pipeline.
    
    Args:
        output_dir: Directory to write Parquet files
        days: Number of days of history to load
        max_users: Optional maximum number of unique users
        table_name: Name of the event logs table
        date_column: Name of the timestamp column
        user_column: Name of the user ID column
        create_validation_report: Whether to generate validation report
    
    Returns:
        Path to the created Parquet file
    
    Raises:
        ValueError: If data loading or validation fails
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = output_path / f"events_{timestamp}.parquet"
    report_path = output_path / f"validation_report_{timestamp}.json"
    
    logger.info(f"Starting data ingestion pipeline")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Loading {days} days of data")
    
    try:
        # Step 1: Load data from PostgreSQL
        logger.info("Step 1: Loading data from PostgreSQL...")
        df = load_recent_events(
            days=days,
            max_users=max_users,
            table_name=table_name,
            date_column=date_column,
            user_column=user_column,
            chunk_size=10000,
        )
        
        if df.empty:
            raise ValueError("No data loaded from database")
        
        logger.info(f"Loaded {len(df)} events, {df['user_id'].nunique()} unique users")
        
        # Step 2: Validate data
        logger.info("Step 2: Validating data with Pandera schema...")
        validated_df = validate_raw_events(df)
        
        # Step 3: Generate validation report
        if create_validation_report:
            logger.info("Step 3: Generating validation report...")
            report = generate_validation_report(validated_df)
            save_validation_report(report, str(report_path))
            logger.info(f"Validation report saved to {report_path}")
        
        # Step 4: Write to Parquet
        logger.info("Step 4: Writing to Parquet format...")
        
        # Partition by date for better query performance
        validated_df["event_date"] = validated_df["event_timestamp"].dt.date
        
        # Write to Parquet with compression and partitioning
        validated_df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
            partition_cols=["event_date"] if len(validated_df) > 10000 else None,
            index=False,
        )
        
        # Also write a single file version for smaller datasets
        if len(validated_df) <= 10000:
            single_file_path = output_path / f"events_latest.parquet"
            validated_df.to_parquet(
                single_file_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            logger.info(f"Latest snapshot saved to {single_file_path}")
        
        # Step 5: Generate schema file for documentation
        schema_path = output_path / f"parquet_schema_{timestamp}.txt"
        with open(schema_path, "w") as f:
            f.write(f"Parquet Schema for {parquet_path}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Rows: {len(validated_df)}\n")
            f.write(f"Columns: {len(validated_df.columns)}\n\n")
            
            for col in validated_df.columns:
                dtype = str(validated_df[col].dtype)
                null_count = validated_df[col].isnull().sum()
                f.write(f"{col}: {dtype} (nulls: {null_count})\n")
        
        logger.info(f"Data ingestion complete!")
        logger.info(f"  - Parquet file: {parquet_path}")
        logger.info(f"  - Rows: {len(validated_df):,}")
        logger.info(f"  - Size: {parquet_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"  - Date range: {validated_df['event_timestamp'].min()} to {validated_df['event_timestamp'].max()}")
        
        return str(parquet_path)
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


def run_ingestion_pipeline(config: Optional[dict] = None) -> dict:
    """Run the complete ingestion pipeline with configuration.
    
    Args:
        config: Optional configuration dictionary. If not provided,
                uses default values.
    
    Returns:
        Dictionary with ingestion results and metadata
    """
    if config is None:
        config = {
            "output_dir": "data/raw",
            "days": 30,
            "max_users": None,
            "create_validation_report": True,
        }
    
    start_time = datetime.now()
    
    try:
        parquet_path = ingest_raw_events(**config)
        
        result = {
            "status": "success",
            "parquet_path": parquet_path,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "config": config,
        }
        
        return result
        
    except Exception as e:
        result = {
            "status": "failed",
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "config": config,
        }
        
        return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=== LedgerFlow Data Ingestion Pipeline ===")
    
    # Check for required environment variables
    required_vars = ["LEDGERFLOW_DB_NAME", "LEDGERFLOW_DB_USER", "LEDGERFLOW_DB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the ingestion pipeline.")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="LedgerFlow Data Ingestion Pipeline")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to load")
    parser.add_argument("--max-users", type=int, help="Maximum number of unique users to load")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory for Parquet files")
    parser.add_argument("--skip-report", action="store_true", help="Skip validation report generation")
    
    args = parser.parse_args()
    
    # Run ingestion pipeline
    config = {
        "output_dir": args.output_dir,
        "days": args.days,
        "max_users": args.max_users,
        "create_validation_report": not args.skip_report,
    }
    
    print(f"Configuration: {config}")
    
    try:
        result = run_ingestion_pipeline(config)
        
        if result["status"] == "success":
            print(f"\n✅ Ingestion successful!")
            print(f"   Output: {result['parquet_path']}")
            print(f"   Duration: {result['duration_seconds']:.2f} seconds")
            
            # Suggest DVC command
            print(f"\n💡 Next step: Track with DVC")
            print(f"   dvc add {result['parquet_path']}")
            print(f"   git add {result['parquet_path']}.dvc")
            
        else:
            print(f"\n❌ Ingestion failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)