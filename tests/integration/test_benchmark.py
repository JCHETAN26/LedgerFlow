"""
Runtime benchmark (Phase 6.3).

Asserts the feature pipeline processes a fixed 10K-row synthetic dataset in well
under the 30-second budget. Runs serially (n_jobs=1) so the measurement is stable
across CI machines and not dominated by process-pool spin-up.
"""

import time

import pandas as pd
import pytest

from LedgerFlow.data.synthetic import generate_synthetic_events
from LedgerFlow.pipeline import FeaturePipeline

RUNTIME_BUDGET_SECONDS = 30.0


@pytest.mark.benchmark
def test_pipeline_runtime_under_budget():
    # ~10K events across 500 users.
    events, _ = generate_synthetic_events(n_users=500, days=30, seed=3)
    events = events.iloc[:10_000] if len(events) > 10_000 else events
    reference_time = events["event_timestamp"].max()

    pipe = FeaturePipeline()
    start = time.perf_counter()
    result = pipe.transform_batch(events, reference_time, n_jobs=1)
    elapsed = time.perf_counter() - start

    assert result.shape[1] == 35
    assert elapsed < RUNTIME_BUDGET_SECONDS, (
        f"Pipeline took {elapsed:.2f}s, exceeding {RUNTIME_BUDGET_SECONDS}s budget"
    )
