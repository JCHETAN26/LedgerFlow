"""
Sparkov credit-card-fraud ingest adapter for LedgerFlow.

The `Sparkov simulated fraud dataset
<https://www.kaggle.com/datasets/kartik2112/fraud-detection>`_ is a real-world
event log of credit-card transactions with a ground-truth ``is_fraud`` label.
This module maps it into LedgerFlow's :data:`RawEventSchema` so the *unchanged*
downstream pipeline (featurize -> split -> evaluate -> report) runs on it.

Column mapping
--------------
======================  =====================================================
Sparkov column          LedgerFlow
======================  =====================================================
``trans_num``           ``event_id``           (already unique hex ids)
``cc_num``              ``user_id``            (prefixed ``cc_``)
``amt``                 ``amount``             (real transaction $)
``trans_date_trans_time`` ``event_timestamp``
(all rows)              ``event_type = "purchase"``  (every row is a txn, so all
                        35 amount-aggregation features stay meaningful)
(none)                  ``session_id = NULL``
``is_fraud``            -> per-snapshot ``label`` (see below)
======================  =====================================================

Labels: leakage-safe daily snapshots
------------------------------------
The modeling unit is a **(card, active day) snapshot**, not a card. For each day
on which a card transacts we emit one row whose ``decision_time`` is *midnight*
of that day. Because the time-window features include only events with
``event_timestamp <= decision_time`` (strictly ``>`` cutoff), a snapshot at
midnight of day *D* sees only events from *before* day *D* — same-day
transactions are excluded. The ``label`` is 1 iff the card has a fraudulent
transaction within ``label_horizon_days`` starting at *D*. This is a realistic
"at the start of each active day, will fraud happen today?" framing with no
leakage.

Run as a module (``python -m LedgerFlow.data.sparkov``) it backs the DVC
``ingest`` stage: writes ``data/raw/events.parquet`` and
``data/raw/labels.parquet``. The raw CSV is **not** committed — download it from
Kaggle and point ``LEDGERFLOW_SPARKOV_CSV`` (or ``params.yaml``) at it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from ..params import load_params
from .validators import validate_raw_events

logger = logging.getLogger(__name__)

# Columns we actually read from the (wide) Sparkov CSV.
_USECOLS = ["trans_num", "cc_num", "amt", "trans_date_trans_time", "is_fraud"]


def _resolve_csv_path(csv_path: str | None) -> Path:
    """Find the Sparkov CSV, preferring an explicit arg, then env, then params."""
    if csv_path is None:
        csv_path = os.environ.get("LEDGERFLOW_SPARKOV_CSV")
    if csv_path is None:
        params = load_params()
        csv_path = params.get("sparkov", {}).get("csv_path")
    if not csv_path:
        raise FileNotFoundError(
            "No Sparkov CSV configured. Download fraudTrain.csv from "
            "https://www.kaggle.com/datasets/kartik2112/fraud-detection and set "
            "LEDGERFLOW_SPARKOV_CSV=/path/to/fraudTrain.csv (or sparkov.csv_path "
            "in params.yaml)."
        )
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sparkov CSV not found at '{path}'. Download it from Kaggle "
            "(kartik2112/fraud-detection) and point LEDGERFLOW_SPARKOV_CSV at it."
        )
    return path


def load_sparkov_events(csv_path: str | Path) -> pd.DataFrame:
    """Read a Sparkov CSV and map it to a ``RawEventSchema``-shaped DataFrame.

    Args:
        csv_path: Path to ``fraudTrain.csv`` / ``fraudTest.csv``.

    Returns:
        Event DataFrame with the six raw-event columns plus an ``is_fraud``
        helper column (dropped before validation by :func:`build_events`).
    """
    raw = pd.read_csv(csv_path, usecols=_USECOLS)
    events = pd.DataFrame(
        {
            "event_id": raw["trans_num"].astype(str),
            "user_id": "cc_" + raw["cc_num"].astype("int64").astype(str),
            "event_type": "purchase",
            "event_timestamp": pd.to_datetime(raw["trans_date_trans_time"]),
            "amount": raw["amt"].astype(float),
            "session_id": pd.Series([None] * len(raw), dtype="object"),
            "is_fraud": raw["is_fraud"].astype(int),
        }
    )
    return events


def build_snapshot_labels(
    events: pd.DataFrame,
    label_horizon_days: int = 1,
    snapshot_freq: str = "D",
) -> pd.DataFrame:
    """Build leakage-safe (card, active-period) snapshot labels.

    Args:
        events: Mapped events including the ``is_fraud`` helper column.
        label_horizon_days: A snapshot on period *P* is positive if the card has
            a fraudulent transaction in ``[P, P + horizon)``. Default 1 (same
            day).
        snapshot_freq: pandas floor frequency for the snapshot period — ``"D"``
            for daily (default), ``"W"`` for weekly, etc.

    Returns:
        DataFrame with columns ``sample_id``, ``user_id``, ``decision_time``,
        ``label`` — one row per (card, active period).
    """
    # Period start for each event. ``to_period(...).start_time`` works for both
    # fixed (``"D"``) and non-fixed (``"W"``, ``"M"``) frequencies — unlike
    # ``dt.floor``, which rejects weekly/monthly. For daily it is identical to
    # ``floor("D")`` (midnight of that day).
    period = events["event_timestamp"].dt.to_period(snapshot_freq).dt.start_time
    base = pd.DataFrame(
        {"user_id": events["user_id"], "period": period, "is_fraud": events["is_fraud"]}
    )

    # One snapshot per (card, active period). decision_time = start of the period.
    snapshots = (
        base[["user_id", "period"]].drop_duplicates().reset_index(drop=True)
    )

    # Periods in which each card actually had a fraudulent transaction.
    fraud_periods = (
        base.loc[base["is_fraud"] == 1, ["user_id", "period"]]
        .drop_duplicates()
        .rename(columns={"period": "fraud_period"})
    )

    # A fraud on period F labels snapshots P with P <= F < P + horizon, i.e.
    # P in {F, F-1, ..., F-(horizon-1)}. Step back one period at a time and mark
    # each covered (card, period) as positive.
    step = pd.tseries.frequencies.to_offset(snapshot_freq)
    covered = []
    cur = fraud_periods.rename(columns={"fraud_period": "period"})[
        ["user_id", "period"]
    ]
    for _ in range(max(1, label_horizon_days)):
        covered.append(cur)
        cur = cur.assign(period=cur["period"] - step)
    cover = pd.concat(covered, ignore_index=True).drop_duplicates()
    cover["label"] = 1

    labeled = snapshots.merge(cover, on=["user_id", "period"], how="left")
    labeled["label"] = labeled["label"].fillna(0).astype("int64")
    labeled = labeled.rename(columns={"period": "decision_time"})
    labeled["sample_id"] = (
        labeled["user_id"] + "_" + labeled["decision_time"].dt.strftime("%Y%m%d%H%M%S")
    )
    return labeled[["sample_id", "user_id", "decision_time", "label"]]


def build_events(events_with_fraud: pd.DataFrame) -> pd.DataFrame:
    """Drop the helper column and validate against the strict raw schema."""
    events = events_with_fraud.drop(columns=["is_fraud"])
    return validate_raw_events(events)


def sparkov_main(
    csv_path: str | None = None,
    output_dir: str = "data/raw",
    label_horizon_days: int | None = None,
    snapshot_freq: str | None = None,
) -> tuple[str, str]:
    """Ingest Sparkov: write validated events + snapshot labels as Parquet.

    Returns:
        ``(events_path, labels_path)``.
    """
    params = load_params().get("sparkov", {})
    if label_horizon_days is None:
        label_horizon_days = int(params.get("label_horizon_days", 1))
    if snapshot_freq is None:
        snapshot_freq = str(params.get("snapshot_freq", "D"))

    path = _resolve_csv_path(csv_path)
    logger.info("Ingesting Sparkov from %s", path)

    mapped = load_sparkov_events(path)
    labels = build_snapshot_labels(
        mapped, label_horizon_days=label_horizon_days, snapshot_freq=snapshot_freq
    )
    events = build_events(mapped)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.parquet"
    labels_path = out_dir / "labels.parquet"
    events.to_parquet(events_path, engine="pyarrow", compression="snappy", index=False)
    labels.to_parquet(labels_path, engine="pyarrow", compression="snappy", index=False)

    n_events = int(len(events))
    n_users = int(events["user_id"].nunique())
    n_samples = int(len(labels))
    positive_rate = round(float(labels["label"].mean()), 4)

    summary = {
        "source": "sparkov",
        "csv_path": str(path),
        "n_events": n_events,
        "n_users": n_users,
        "n_samples": n_samples,
        "positive_rate": positive_rate,
        "label_horizon_days": label_horizon_days,
        "snapshot_freq": snapshot_freq,
    }
    (out_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "Sparkov: %d events, %d cards -> %d snapshots (%.2f%% positive)",
        n_events,
        n_users,
        n_samples,
        100.0 * positive_rate,
    )
    return str(events_path), str(labels_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sparkov_main()
