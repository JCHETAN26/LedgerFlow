"""
Tests for the ``ledgerflow`` command-line interface.

These exercise argument parsing, command dispatch, the run-stage orchestration
(with the heavy stage functions mocked), and the ``score`` error paths — without
running the full, slow pipeline.
"""

from __future__ import annotations

import pandas as pd

from LedgerFlow import cli


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert "ledgerflow" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage: ledgerflow" in capsys.readouterr().out


def test_run_orchestrates_all_stages_in_order(monkeypatch, tmp_path):
    """`run` calls ingest → featurize → split → evaluate → report once each."""
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "LedgerFlow.data.synthetic.generate_main",
        lambda **_: calls.append("ingest"),
    )
    monkeypatch.setattr(
        "LedgerFlow.pipeline.featurize_main", lambda *a, **k: calls.append("featurize")
    )
    monkeypatch.setattr(
        "LedgerFlow.data.splitter.split_main",
        lambda *a, **k: (calls.append("split"), {"train": "t"})[1],
    )
    monkeypatch.setattr(
        "LedgerFlow.evaluation.compare.evaluate_main",
        lambda *a, **k: calls.append("evaluate"),
    )
    monkeypatch.setattr(
        "LedgerFlow.evaluation.report.generate_memo",
        lambda *a, **k: (calls.append("report"), "memo.md")[1],
    )

    assert cli.main(["run", "--source", "synthetic", "--n-users", "10"]) == 0
    assert calls == ["ingest", "featurize", "split", "evaluate", "report"]


def test_run_skip_report(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        "LedgerFlow.data.synthetic.generate_main", lambda **_: calls.append("ingest")
    )
    monkeypatch.setattr(
        "LedgerFlow.pipeline.featurize_main", lambda *a, **k: calls.append("featurize")
    )
    monkeypatch.setattr(
        "LedgerFlow.data.splitter.split_main",
        lambda *a, **k: (calls.append("split"), {})[1],
    )
    monkeypatch.setattr(
        "LedgerFlow.evaluation.compare.evaluate_main",
        lambda *a, **k: calls.append("evaluate"),
    )
    monkeypatch.setattr(
        "LedgerFlow.evaluation.report.generate_memo",
        lambda *a, **k: calls.append("report"),
    )

    assert cli.main(["run", "--source", "synthetic", "--skip-report"]) == 0
    assert "report" not in calls


def test_run_sparkov_passes_snapshot_and_horizon(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_sparkov(csv_path=None, label_horizon_days=None, snapshot_freq=None):
        captured.update(
            csv_path=csv_path, horizon=label_horizon_days, freq=snapshot_freq
        )

    monkeypatch.setattr("LedgerFlow.data.sparkov.sparkov_main", fake_sparkov)
    monkeypatch.setattr("LedgerFlow.pipeline.featurize_main", lambda *a, **k: None)
    monkeypatch.setattr("LedgerFlow.data.splitter.split_main", lambda *a, **k: {})
    monkeypatch.setattr(
        "LedgerFlow.evaluation.compare.evaluate_main", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "LedgerFlow.evaluation.report.generate_memo", lambda *a, **k: "m"
    )

    cli.main(
        ["run", "--events", "my.csv", "--snapshot", "W", "--horizon", "3"]
    )
    assert captured == {"csv_path": "my.csv", "horizon": 3, "freq": "W"}


def test_score_no_event_data(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["score", "--user", "cc_1"]) == 1
    assert "no event data" in capsys.readouterr().err


def test_score_unknown_user(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["e1"],
            "user_id": ["cc_known"],
            "event_type": ["purchase"],
            "event_timestamp": [pd.Timestamp("2020-01-01")],
            "amount": [10.0],
            "session_id": [None],
        }
    ).to_parquet(raw / "events.parquet", index=False)

    assert cli.main(["score", "--user", "cc_missing"]) == 1
    err = capsys.readouterr().err
    assert "no events for user" in err
    assert "cc_known" in err  # the helpful example list


def test_score_no_models(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["e1"],
            "user_id": ["cc_known"],
            "event_type": ["purchase"],
            "event_timestamp": [pd.Timestamp("2020-01-01")],
            "amount": [10.0],
            "session_id": [None],
        }
    ).to_parquet(raw / "events.parquet", index=False)

    assert cli.main(["score", "--user", "cc_known"]) == 1
    assert "no trained models" in capsys.readouterr().err


def test_build_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run"])
    assert args.source == "sparkov"
    assert args.snapshot is None
    assert args.skip_report is False
