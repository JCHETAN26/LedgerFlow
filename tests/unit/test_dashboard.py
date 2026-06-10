"""
Unit tests for the dashboard's artifact loaders (Streamlit-free).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the repo root importable regardless of install mode (dashboard/ is not
# part of the installed LedgerFlow package).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.loaders import Artifacts, find_project_root, metrics_table  # noqa: E402


@pytest.fixture
def fake_project(tmp_path):
    (tmp_path / "params.yaml").write_text("data: {}\n")
    reports = tmp_path / "reports"
    (reports / "curves").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "models" / "LightGBM").mkdir(parents=True)

    (reports / "eval_metrics.json").write_text(
        json.dumps(
            {
                "models": {
                    "LogisticRegression": {
                        "auc_roc": 0.82, "avg_precision": 0.80, "brier_score": 0.18,
                        "log_loss": 0.55, "precision_at_5pct_fpr": 0.90, "f1": 0.74,
                    },
                    "LightGBM": {
                        "auc_roc": 0.85, "avg_precision": 0.84, "brier_score": 0.16,
                        "log_loss": 0.49, "precision_at_5pct_fpr": 0.91, "f1": 0.75,
                    },
                },
                "recommended": "LightGBM",
                "recommendation_reason": "highest AUC",
            }
        )
    )
    (reports / "feature_importance.json").write_text(
        json.dumps(
            [
                {"feature": "purchase_count_30d", "mean_importance": 0.3, "LightGBM": 0.4},
                {"feature": "purchase_amount_std_1h", "mean_importance": 0.0005, "LightGBM": 0.0},
            ]
        )
    )
    (reports / "low_signal_features.json").write_text(
        json.dumps({"threshold": 0.001, "features": ["purchase_amount_std_1h"]})
    )
    (reports / "feature_runtime.json").write_text(
        json.dumps({"n_users": 2000, "n_features": 35, "runtime_seconds": 1.4})
    )
    (reports / "recommendation_memo.md").write_text("# Memo\nLightGBM recommended.")
    (reports / "curves" / "LightGBM_roc.json").write_text(
        json.dumps({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    )
    pd.DataFrame({"user_id": ["u1"], "purchase_count_24h": [2.0], "label": [1]}).to_parquet(
        tmp_path / "data" / "processed" / "features.parquet"
    )
    return tmp_path


def test_find_project_root(fake_project):
    nested = fake_project / "data" / "processed"
    assert find_project_root(nested) == fake_project


def test_available_reports_present(fake_project):
    art = Artifacts(fake_project)
    avail = art.available()
    assert avail["eval_metrics"] and avail["feature_importance"]
    assert avail["memo"] and avail["features"] and avail["curves"]
    assert not avail["events"]  # not created


def test_load_eval_metrics_and_table(fake_project):
    art = Artifacts(fake_project)
    ev = art.load_eval_metrics()
    assert ev["recommended"] == "LightGBM"
    table = metrics_table(ev)
    assert list(table.index) == ["LogisticRegression", "LightGBM"]
    assert table.loc["LightGBM", "auc_roc"] == pytest.approx(0.85)
    assert list(table.columns)[0] == "auc_roc"


def test_load_other_artifacts(fake_project):
    art = Artifacts(fake_project)
    assert len(art.load_feature_importance()) == 2
    assert art.load_low_signal()["features"] == ["purchase_amount_std_1h"]
    assert art.load_feature_runtime()["n_users"] == 2000
    assert "Memo" in art.load_memo()
    assert art.load_curve("LightGBM", "roc")["fpr"] == [0.0, 1.0]
    assert art.load_features().shape[0] == 1


def test_list_models_requires_pickle(fake_project):
    # The LightGBM dir exists but has no model.pkl yet.
    assert Artifacts(fake_project).list_models() == []
    (fake_project / "models" / "LightGBM" / "model.pkl").write_bytes(b"x")
    assert Artifacts(fake_project).list_models() == ["LightGBM"]
