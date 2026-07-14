"""
Sparkov ingest adapter tests.

These run against a tiny hand-built Sparkov-shaped CSV (no Kaggle download
needed), proving the column mapping, schema validation, leakage-safe daily
snapshot labels, and that the snapshot featurization path stays bit-for-bit
consistent with the single-user (serving) path.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from LedgerFlow.data import sparkov
from LedgerFlow.data.validators import validate_raw_events
from LedgerFlow.pipeline import FeaturePipeline, featurize_main


def _write_sparkov_csv(path) -> None:
    """A small but realistic Sparkov fragment: 2 cards across a few days."""
    rows = [
        # trans_num, cc_num, amt, trans_date_trans_time, is_fraud
        ("a1", 1111, 10.0, "2019-01-01 09:00:00", 0),
        ("a2", 1111, 20.0, "2019-01-01 18:00:00", 0),
        ("a3", 1111, 5.0, "2019-01-02 10:00:00", 0),
        ("a4", 1111, 999.0, "2019-01-03 02:00:00", 1),  # fraud on 01-03
        ("b1", 2222, 7.5, "2019-01-01 12:00:00", 0),
        ("b2", 2222, 8.5, "2019-01-02 12:00:00", 0),
    ]
    pd.DataFrame(
        rows,
        columns=["trans_num", "cc_num", "amt", "trans_date_trans_time", "is_fraud"],
    ).to_csv(path, index=False)


@pytest.fixture
def sparkov_csv(tmp_path):
    csv = tmp_path / "fraudMini.csv"
    _write_sparkov_csv(csv)
    return csv


def test_mapping_to_raw_schema(sparkov_csv):
    mapped = sparkov.load_sparkov_events(sparkov_csv)
    assert (mapped["event_type"] == "purchase").all()
    assert mapped["user_id"].tolist()[0] == "cc_1111"
    assert mapped["event_id"].is_unique
    # The mapped events (minus the helper column) satisfy the strict schema.
    validate_raw_events(mapped.drop(columns=["is_fraud"]))


def test_daily_snapshot_labels(sparkov_csv):
    mapped = sparkov.load_sparkov_events(sparkov_csv)
    labels = sparkov.build_snapshot_labels(mapped, label_horizon_days=1)

    # One snapshot per (card, active day): card 1111 active on 3 days, 2222 on 2.
    assert len(labels) == 5
    assert labels["sample_id"].is_unique
    assert set(labels.columns) == {"sample_id", "user_id", "decision_time", "label"}

    # Only the 2019-01-03 snapshot for card 1111 (the fraud day) is positive.
    positives = labels[labels["label"] == 1]
    assert len(positives) == 1
    pos = positives.iloc[0]
    assert pos["user_id"] == "cc_1111"
    assert pos["decision_time"] == pd.Timestamp("2019-01-03")


def test_label_horizon_widens_positive_window(sparkov_csv):
    mapped = sparkov.load_sparkov_events(sparkov_csv)
    # With a 3-day horizon, the fraud on 01-03 also labels the 01-02 and 01-01
    # snapshots of that card as positive.
    labels = sparkov.build_snapshot_labels(mapped, label_horizon_days=3)
    pos = labels[(labels["user_id"] == "cc_1111") & (labels["label"] == 1)]
    assert set(pos["decision_time"]) == {
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-01-02"),
        pd.Timestamp("2019-01-03"),
    }


def test_snapshot_decision_time_excludes_same_day_events(sparkov_csv):
    # decision_time is midnight, and features are computed strictly > cutoff and
    # <= reference_time, so a same-day purchase (timestamped after midnight) must
    # NOT appear in the snapshot's features -> the fraud-day snapshot for card
    # 1111 reflects only the two prior days of purchases.
    mapped = sparkov.load_sparkov_events(sparkov_csv)
    events = mapped.drop(columns=["is_fraud"])
    pipe = FeaturePipeline()

    dt = pd.Timestamp("2019-01-03")  # the fraud-day snapshot
    history = events[events["user_id"] == "cc_1111"]
    row = pipe.transform_single(history, dt)

    # 3 purchases occurred before midnight 01-03 (09:00, 18:00 on 01-01; 10:00 on
    # 01-02); the 02:00 fraud on 01-03 is after the decision time and excluded.
    assert row["purchase_count_30d"] == 3.0
    assert row["purchase_amount_sum_30d"] == pytest.approx(35.0)


def test_transform_samples_matches_single(sparkov_csv):
    mapped = sparkov.load_sparkov_events(sparkov_csv)
    events = mapped.drop(columns=["is_fraud"])
    samples = sparkov.build_snapshot_labels(mapped)[
        ["sample_id", "user_id", "decision_time"]
    ]
    pipe = FeaturePipeline()

    batch = pipe.transform_samples(events, samples)
    assert list(batch.index) == samples["sample_id"].tolist()

    by_id = samples.set_index("sample_id")
    for sample_id in batch.index:
        uid = by_id.loc[sample_id, "user_id"]
        dt = by_id.loc[sample_id, "decision_time"]
        single = pipe.transform_single(events[events["user_id"] == uid], dt)
        for fname in pipe.feature_names:
            assert float(batch.loc[sample_id, fname]) == pytest.approx(
                float(single[fname]), abs=1e-6
            ), f"snapshot skew in {fname} for {sample_id}"


def test_end_to_end_ingest_and_featurize(sparkov_csv, tmp_path):
    raw_dir = tmp_path / "raw"
    # Pin the cadence the assertions below rely on, independent of params.yaml
    # (whose default snapshot_freq may be "W" for the real large-scale run).
    events_path, labels_path = sparkov.sparkov_main(
        csv_path=str(sparkov_csv), output_dir=str(raw_dir), snapshot_freq="D"
    )

    summary = json.loads((raw_dir / "ingest_summary.json").read_text())
    assert summary["n_events"] == 6
    assert summary["n_users"] == 2
    assert summary["n_samples"] == 5

    out = featurize_main(
        events_path=events_path,
        labels_path=labels_path,
        output_path=str(tmp_path / "features.parquet"),
        metrics_path=str(tmp_path / "runtime.json"),
    )
    full = pd.read_parquet(out)
    # One row per snapshot, with label + decision_time carried for the splitter.
    assert len(full) == 5
    assert {"label", "decision_time"}.issubset(full.columns)
    assert full.index.name == "sample_id"

    metrics = json.loads((tmp_path / "runtime.json").read_text())
    assert metrics["snapshot_mode"] is True
    assert metrics["n_rows"] == 5
    assert metrics["n_users"] == 2


def test_missing_csv_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("LEDGERFLOW_SPARKOV_CSV", raising=False)
    with pytest.raises(FileNotFoundError):
        sparkov._resolve_csv_path(str(tmp_path / "does_not_exist.csv"))
