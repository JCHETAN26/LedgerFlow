"""
Pytest configuration and shared fixtures for LedgerFlow tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_event_data():
    """Generate sample event data for testing."""
    np.random.seed(42)
    
    n_events = 1000
    n_users = 100
    
    user_ids = [f"user_{i}" for i in range(n_users)]
    event_types = ["purchase", "login", "view", "click"]
    
    data = {
        "event_id": [f"event_{i}" for i in range(n_events)],
        "user_id": np.random.choice(user_ids, n_events),
        "event_type": np.random.choice(event_types, n_events, p=[0.3, 0.2, 0.3, 0.2]),
        "event_timestamp": pd.date_range(
            start="2024-01-01",
            periods=n_events,
            freq="1h"
        ),
        "amount": np.random.exponential(scale=50, size=n_events),
        "session_id": [f"session_{i}" for i in range(n_events)],
    }
    
    # Set amount to None for non-purchase events
    mask = data["event_type"] != "purchase"
    data["amount"][mask] = None
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def reference_time():
    """Return a reference time for feature computation."""
    return pd.Timestamp("2024-01-15 12:00:00")