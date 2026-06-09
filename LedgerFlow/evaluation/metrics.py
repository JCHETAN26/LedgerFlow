"""
Evaluation metrics for LedgerFlow models.

Includes ranking quality (AUC-ROC, average precision), calibration (Brier score,
log loss — critical because the model outputs probabilities), an operational
metric (precision at a fixed false-positive rate), and helpers for curves and
low-signal feature flagging.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def precision_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr: float = 0.05
) -> float:
    """Precision at the operating point with false-positive rate <= ``fpr``.

    Picks the highest-recall threshold whose FPR does not exceed ``fpr`` and
    reports the precision there — i.e. "if we only tolerate 5% false positives,
    how precise are our alerts?"
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    mask = fprs <= fpr
    if not mask.any():
        return 0.0
    idx = np.where(mask)[0][-1]
    threshold = thresholds[idx]
    pred = (y_score >= threshold).astype(int)
    return float(precision_score(y_true, pred, zero_division=0))


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Compute the full metric suite for a fitted classifier on the test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "precision_at_5pct_fpr": precision_at_fpr(y_test, proba, fpr=0.05),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return ROC curve points as plain lists (a JSON-serialisable artifact)."""
    fprs, tprs, _ = roc_curve(y_true, y_score)
    return {"fpr": fprs.tolist(), "tpr": tprs.tolist()}


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return precision-recall curve points as plain lists."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def flag_low_signal_features(
    importance_df: pd.DataFrame, threshold: float = 0.001
) -> list[str]:
    """Return features whose mean importance across models is below ``threshold``.

    Args:
        importance_df: DataFrame with a ``feature`` column and a
            ``mean_importance`` column.
        threshold: Importance floor; features below it are removal candidates.
    """
    low = importance_df[importance_df["mean_importance"] < threshold]
    return [str(f) for f in low["feature"].tolist()]
