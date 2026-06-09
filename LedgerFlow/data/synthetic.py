"""
Synthetic event-log generator for LedgerFlow.

Real deployments ingest from PostgreSQL via :mod:`LedgerFlow.data.loader`. For
offline development, CI, and ``dvc repro`` we need a deterministic, dependency-free
data source. This module generates a schema-valid event log plus per-user labels
with real signal, so the downstream models have something to learn.

Run as a module (``python -m LedgerFlow.data.synthetic``) it backs the DVC
``ingest`` stage: writes ``data/raw/events.parquet`` and ``data/raw/labels.parquet``.

The label is a (noisy) function of each user's purchase frequency and spend, so
the time-window features are genuinely predictive — but never perfectly, so the
model metrics stay realistic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .validators import validate_raw_events

logger = logging.getLogger(__name__)

EVENT_TYPES = ["purchase", "login", "view", "click"]
EVENT_TYPE_PROBS = [0.30, 0.20, 0.30, 0.20]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def generate_synthetic_events(
    n_users: int = 2000,
    days: int = 30,
    seed: int = 42,
    end: str = "2024-01-31 00:00:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic event log and per-user labels.

    Args:
        n_users: Number of distinct users.
        days: Length of the history window, in days, ending at ``end``.
        seed: RNG seed for full reproducibility.
        end: Timestamp marking the end of the history window.

    Returns:
        ``(events, labels)`` where ``events`` matches ``RawEventSchema`` and
        ``labels`` has columns ``user_id``, ``label``, ``decision_time``.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=days)
    span_seconds = int((end_ts - start_ts).total_seconds())

    # Each user has a latent "activity level" driving how many events they emit.
    activity = rng.gamma(shape=2.0, scale=1.0, size=n_users)
    events_per_user = np.clip((activity * 8).astype(int) + 1, 1, 200)

    rows = []
    event_counter = 0
    per_user_purchase_count = np.zeros(n_users)
    per_user_purchase_sum = np.zeros(n_users)
    per_user_last_ts = np.full(n_users, start_ts.value, dtype=np.int64)

    for u in range(n_users):
        n_events = int(events_per_user[u])
        # Random event offsets within the window (seconds from start).
        offsets = np.sort(rng.integers(0, span_seconds, size=n_events))
        types = rng.choice(EVENT_TYPES, size=n_events, p=EVENT_TYPE_PROBS)

        for i in range(n_events):
            ts = start_ts + pd.Timedelta(seconds=int(offsets[i]))
            etype = types[i]
            if etype == "purchase":
                amount = float(np.round(rng.lognormal(mean=3.0, sigma=1.0), 2))
                per_user_purchase_count[u] += 1
                per_user_purchase_sum[u] += amount
            else:
                amount = None
            rows.append(
                (
                    f"evt_{event_counter:08d}",
                    f"user_{u:06d}",
                    etype,
                    ts,
                    amount,
                    f"sess_{u:06d}_{i // 5:03d}",
                )
            )
            event_counter += 1
        per_user_last_ts[u] = (start_ts + pd.Timedelta(seconds=int(offsets[-1]))).value

    events = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "user_id",
            "event_type",
            "event_timestamp",
            "amount",
            "session_id",
        ],
    )
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

    # Build a label correlated with purchase frequency and spend (plus noise).
    def _z(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / std if std > 0 else np.zeros_like(x)

    logit = (
        1.2 * _z(per_user_purchase_count)
        + 0.9 * _z(per_user_purchase_sum)
        + rng.normal(0.0, 0.8, size=n_users)
        - 1.0  # shift to keep the positive class a minority
    )
    prob = _sigmoid(logit)
    label = (rng.random(n_users) < prob).astype(int)

    labels = pd.DataFrame(
        {
            "user_id": [f"user_{u:06d}" for u in range(n_users)],
            "label": label,
            "decision_time": pd.to_datetime(per_user_last_ts),
        }
    )

    logger.info(
        "Generated %d events for %d users (%.1f%% positive labels)",
        len(events),
        n_users,
        100.0 * label.mean(),
    )
    return events, labels


def generate_main(
    output_dir: str = "data/raw",
    n_users: int = 2000,
    days: int = 30,
    seed: int = 42,
) -> tuple[str, str]:
    """Generate synthetic data, validate events, and write Parquet outputs.

    Returns:
        ``(events_path, labels_path)``.
    """
    events, labels = generate_synthetic_events(n_users=n_users, days=days, seed=seed)

    # Validate against the same strict schema used for real ingestion.
    events = validate_raw_events(events)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.parquet"
    labels_path = out_dir / "labels.parquet"

    events.to_parquet(events_path, engine="pyarrow", compression="snappy", index=False)
    labels.to_parquet(labels_path, engine="pyarrow", compression="snappy", index=False)

    summary = {
        "n_events": int(len(events)),
        "n_users": int(n_users),
        "positive_rate": round(float(labels["label"].mean()), 4),
        "seed": seed,
    }
    (out_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Wrote %s and %s", events_path, labels_path)
    return str(events_path), str(labels_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    generate_main()
