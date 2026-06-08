"""
PostgreSQL data loader for LedgerFlow.

This module handles loading raw event logs from PostgreSQL using SQLAlchemy,
with configurable date ranges and user subsets.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)


def get_database_connection() -> Engine:
    """Create SQLAlchemy engine from environment variables.
    
    Reads connection parameters from environment variables:
    - LEDGERFLOW_DB_HOST: Database host (default: localhost)
    - LEDGERFLOW_DB_PORT: Database port (default: 5432)
    - LEDGERFLOW_DB_NAME: Database name (required)
    - LEDGERFLOW_DB_USER: Database user (required)
    - LEDGERFLOW_DB_PASSWORD: Database password (required)
    
    Returns:
        SQLAlchemy engine instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    # Read environment variables
    host = os.getenv("LEDGERFLOW_DB_HOST", "localhost")
    port = os.getenv("LEDGERFLOW_DB_PORT", "5432")
    database = os.getenv("LEDGERFLOW_DB_NAME")
    user = os.getenv("LEDGERFLOW_DB_USER")
    password = os.getenv("LEDGERFLOW_DB_PASSWORD")
    
    # Validate required parameters
    if not database:
        raise ValueError("LEDGERFLOW_DB_NAME environment variable is required")
    if not user:
        raise ValueError("LEDGERFLOW_DB_USER environment variable is required")
    if not password:
        raise ValueError("LEDGERFLOW_DB_PASSWORD environment variable is required")
    
    # Construct connection URL
    connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    logger.info(f"Creating database connection to {host}:{port}/{database}")
    return create_engine(connection_url, pool_pre_ping=True)


def load_events_from_postgres(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_ids: Optional[list[str]] = None,
    table_name: str = "event_logs",
    date_column: str = "event_timestamp",
    user_column: str = "user_id",
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Load event logs from PostgreSQL with optional filtering.
    
    Args:
        start_date: Optional start date for filtering (inclusive)
        end_date: Optional end date for filtering (exclusive)
        user_ids: Optional list of user IDs to filter
        table_name: Name of the event logs table (default: "event_logs")
        date_column: Name of the timestamp column (default: "event_timestamp")
        user_column: Name of the user ID column (default: "user_id")
        chunk_size: Number of rows to fetch at a time (for large datasets)
    
    Returns:
        pandas DataFrame with event log data
    
    Raises:
        ValueError: If database connection fails or query returns no data
        sqlalchemy.exc.SQLAlchemyError: For database errors
    """
    engine = get_database_connection()
    
    # Build WHERE clause conditions
    conditions = []
    params = {}
    
    if start_date:
        conditions.append(f"{date_column} >= :start_date")
        params["start_date"] = start_date
    
    if end_date:
        conditions.append(f"{date_column} < :end_date")
        params["end_date"] = end_date
    
    if user_ids:
        # Convert list to SQL IN clause
        user_placeholders = ", ".join([f":user_{i}" for i in range(len(user_ids))])
        conditions.append(f"{user_column} IN ({user_placeholders})")
        for i, user_id in enumerate(user_ids):
            params[f"user_{i}"] = user_id
    
    # Build query
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = text(f"""
        SELECT 
            event_id,
            {user_column} as user_id,
            event_type,
            {date_column} as event_timestamp,
            amount,
            session_id
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY {date_column}
    """)
    
    logger.info(f"Loading events from {table_name} with filters: {where_clause}")
    
    try:
        # Execute query and load into DataFrame
        with engine.connect() as connection:
            result = connection.execute(query, params)
            
            # Use pandas read_sql for efficient chunked loading
            df = pd.read_sql(
                query, 
                connection, 
                params=params,
                chunksize=chunk_size
            )
            
            # If chunksize is specified, concatenate chunks
            if chunk_size:
                df_chunks = []
                for chunk in df:
                    df_chunks.append(chunk)
                df = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()
            else:
                df = df
            
        logger.info(f"Loaded {len(df)} events from database")
        
        if df.empty:
            logger.warning("Query returned empty result set")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load events from database: {e}")
        raise


def load_recent_events(
    days: int = 30,
    max_users: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to load recent events.
    
    Args:
        days: Number of days of history to load
        max_users: Optional maximum number of unique users to load
        **kwargs: Additional arguments passed to load_events_from_postgres
    
    Returns:
        pandas DataFrame with recent event data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Loading events from {start_date.date()} to {end_date.date()}")
    
    df = load_events_from_postgres(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    # Optionally limit to top users by event count
    if max_users and not df.empty and len(df["user_id"].unique()) > max_users:
        top_users = df["user_id"].value_counts().head(max_users).index
        df = df[df["user_id"].isin(top_users)]
        logger.info(f"Limited to top {max_users} users")
    
    return df


if __name__ == "__main__":
    # Example usage and simple test
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Check if environment variables are set
    if not all(os.getenv(var) for var in ["LEDGERFLOW_DB_NAME", "LEDGERFLOW_DB_USER", "LEDGERFLOW_DB_PASSWORD"]):
        print("Error: Database environment variables not set")
        print("Please set:")
        print("  - LEDGERFLOW_DB_NAME")
        print("  - LEDGERFLOW_DB_USER")
        print("  - LEDGERFLOW_DB_PASSWORD")
        sys.exit(1)
    
    try:
        # Test database connection
        engine = get_database_connection()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection test passed")
        
        # Try to load a small sample (last 7 days)
        print("Loading sample data (last 7 days)...")
        df = load_recent_events(days=7, chunk_size=1000)
        
        if not df.empty:
            print(f"✅ Successfully loaded {len(df)} events")
            print(f"   Columns: {', '.join(df.columns)}")
            print(f"   Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
            print(f"   Unique users: {df['user_id'].nunique()}")
        else:
            print("⚠️  No data loaded (table might be empty or filters too restrictive)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)