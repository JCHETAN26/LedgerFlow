"""
End-to-end integration test for the offline pipeline:
generate -> featurize -> split -> evaluate -> memo.

Runs entirely in a tmp directory on a small synthetic dataset, exercising the
same code paths the DVC stages call.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from LedgerFlow.data.splitter import split_main
from LedgerFlow.data.synthetic import generate_main
from LedgerFlow.evaluation.compare import evaluate_main
from LedgerFlow.evaluation.report import generate_memo
from LedgerFlow.pipeline import featurize_main


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("ledgerflow_run")
    raw = root / "raw"
    processed = root / "processed"
    splits = root / "splits"
    models = root / "models"
    reports = root / "reports"

    events_path, labels_path = generate_main(
        output_dir=str(raw), n_users=400, days=30, seed=11
    )
    featurize_main(
        events_path=events_path,
        labels_path=labels_path,
        output_path=str(processed / "features.parquet"),
        metrics_path=str(reports / "feature_runtime.json"),
    )
    split_main(
        features_path=str(processed / "features.parquet"),
        output_dir=str(splits),
    )
    eval_metrics = evaluate_main(
        splits_dir=str(splits),
        models_dir=str(models),
        reports_dir=str(reports),
    )
    memo_path = generate_memo(
        reports_dir=str(reports),
        output_path=str(reports / "recommendation_memo.md"),
    )
    return {
        "root": root,
        "reports": reports,
        "models": models,
        "processed": processed,
        "eval_metrics": eval_metrics,
        "memo_path": memo_path,
    }


def test_features_written_with_all_columns(pipeline_run):
    df = pd.read_parquet(pipeline_run["processed"] / "features.parquet")
    assert df.shape[0] == 400
    assert "label" in df.columns
    # 35 features + label + decision_time
    assert df.shape[1] == 37


def test_all_three_models_evaluated(pipeline_run):
    models = pipeline_run["eval_metrics"]["models"]
    assert set(models.keys()) == {"LogisticRegression", "XGBoost", "LightGBM"}
    for m in models.values():
        assert 0.0 <= m["auc_roc"] <= 1.0


def test_model_artifacts_saved(pipeline_run):
    for name in ["LogisticRegression", "XGBoost", "LightGBM"]:
        assert (pipeline_run["models"] / name / "model.pkl").exists()


def test_recommendation_is_a_real_model(pipeline_run):
    rec = pipeline_run["eval_metrics"]["recommended"]
    assert rec in {"LogisticRegression", "XGBoost", "LightGBM"}


def test_low_signal_artifact_written(pipeline_run):
    data = json.loads(
        (pipeline_run["reports"] / "low_signal_features.json").read_text()
    )
    assert "features" in data
    assert "threshold" in data


def test_memo_contains_recommendation(pipeline_run):
    memo = Path(pipeline_run["memo_path"]).read_text()
    assert "# Model Evaluation Report" in memo
    assert "Recommended" in memo
    assert pipeline_run["eval_metrics"]["recommended"] in memo
