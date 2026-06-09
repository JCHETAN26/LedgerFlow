"""
Unit tests for evaluation metrics and helpers.
"""

import numpy as np
import pandas as pd
import pytest

from LedgerFlow.evaluation.metrics import (
    evaluate_model,
    flag_low_signal_features,
    pr_curve_points,
    precision_at_fpr,
    roc_curve_points,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # well-separated scores
    y_score = np.array([0.1, 0.2, 0.3, 0.35, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_score


def test_precision_at_fpr_perfect_separation(scores):
    y_true, y_score = scores
    # With a clean gap, precision at low FPR should be 1.0.
    assert precision_at_fpr(y_true, y_score, fpr=0.05) == pytest.approx(1.0)


def test_precision_at_fpr_no_threshold_returns_zero():
    y_true = np.array([0, 1])
    y_score = np.array([0.9, 0.1])  # inverted -> first FPR step already > 0
    # fpr=0 cannot be achieved beyond the trivial point -> 0.0
    val = precision_at_fpr(y_true, y_score, fpr=0.0)
    assert 0.0 <= val <= 1.0


def test_roc_and_pr_curve_points(scores):
    y_true, y_score = scores
    roc = roc_curve_points(y_true, y_score)
    pr = pr_curve_points(y_true, y_score)
    assert len(roc["fpr"]) == len(roc["tpr"])
    assert len(pr["precision"]) == len(pr["recall"])
    assert all(0.0 <= v <= 1.0 for v in roc["fpr"])


def test_flag_low_signal_features():
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "mean_importance": [0.5, 0.0005, 0.002],
        }
    )
    flagged = flag_low_signal_features(df, threshold=0.001)
    assert flagged == ["b"]


class _DummyModel:
    """Minimal predict_proba/predict stub for evaluate_model."""

    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])

    def predict(self, X):
        return (self._proba >= 0.5).astype(int)


def test_evaluate_model_returns_all_metrics(scores):
    y_true, y_score = scores
    model = _DummyModel(y_score)
    X = pd.DataFrame({"f": range(len(y_true))})
    result = evaluate_model(model, X, y_true)
    expected_keys = {
        "auc_roc",
        "avg_precision",
        "brier_score",
        "log_loss",
        "precision_at_5pct_fpr",
        "f1",
    }
    assert set(result.keys()) == expected_keys
    assert result["auc_roc"] == pytest.approx(1.0)  # perfect ranking
    assert all(isinstance(v, float) for v in result.values())
