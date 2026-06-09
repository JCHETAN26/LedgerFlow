"""
Evaluation modules for model comparison and reporting.
"""

from .compare import evaluate_main
from .metrics import (
    evaluate_model,
    flag_low_signal_features,
    pr_curve_points,
    precision_at_fpr,
    roc_curve_points,
)
from .report import generate_memo

__all__ = [
    "evaluate_model",
    "precision_at_fpr",
    "flag_low_signal_features",
    "roc_curve_points",
    "pr_curve_points",
    "evaluate_main",
    "generate_memo",
]
