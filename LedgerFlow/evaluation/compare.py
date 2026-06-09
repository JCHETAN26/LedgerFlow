"""
Multi-model comparison runner (DVC ``evaluate`` stage).

Trains Logistic Regression, XGBoost, and LightGBM on the time-based splits,
evaluates them on the held-out test set, extracts feature importance (SHAP for
the tree models, standardised coefficients for LR), flags low-signal features,
and writes everything needed for the auto-generated recommendation memo.

The recommendation is data-driven: the model with the highest test AUC-ROC wins,
with calibration (Brier score) reported alongside.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..features.time_windows import ALL_FEATURES
from ..params import load_params
from .metrics import (
    evaluate_model,
    flag_low_signal_features,
    pr_curve_points,
    roc_curve_points,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f.name for f in ALL_FEATURES]
LOW_SIGNAL_THRESHOLD = 0.001


def _load_split(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    X = df[FEATURE_COLUMNS]
    y = df["label"].to_numpy()
    return X, y


def build_models(seed: int) -> dict:
    """Construct the three classifiers with reproducible settings."""
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            early_stopping_rounds=25,
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def _fit(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Any:
    """Fit a model, using the validation set for early stopping where supported."""
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(25, verbose=False), log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def _importance(name: str, model: Any, X_sample: pd.DataFrame) -> np.ndarray:
    """Per-feature importance, normalised to sum to 1 for cross-model comparison."""
    if name == "LogisticRegression":
        clf = model.named_steps["clf"]
        raw = np.abs(clf.coef_).ravel()
    else:
        raw = _shap_importance(model, X_sample)
    total = raw.sum()
    return np.asarray(raw / total if total > 0 else raw)


def _shap_importance(model: Any, X_sample: pd.DataFrame) -> np.ndarray:
    """Mean absolute SHAP value per feature (falls back to gain importance)."""
    try:
        import shap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_sample)
        if isinstance(values, list):  # older API: one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, -1]
        return np.asarray(np.abs(values).mean(axis=0))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("SHAP failed (%s); using model feature_importances_", exc)
        return np.asarray(model.feature_importances_, dtype=float)


def evaluate_main(
    splits_dir: str = "data/splits",
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> dict:
    """Train, evaluate, and persist all three models and their artifacts."""
    params = load_params()
    seed = params["evaluation"]["random_seed"]

    X_train, y_train = _load_split(f"{splits_dir}/train.parquet")
    X_val, y_val = _load_split(f"{splits_dir}/val.parquet")
    X_test, y_test = _load_split(f"{splits_dir}/test.parquet")

    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    curves_path = reports_path / "curves"
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    curves_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}
    importance_by_model: dict[str, np.ndarray] = {}

    for name, model in build_models(seed).items():
        logger.info("Training %s ...", name)
        model = _fit(name, model, X_train, y_train, X_val, y_val)

        metrics[name] = evaluate_model(model, X_test, y_test)
        importance_by_model[name] = _importance(name, model, X_test)

        model_dir = models_path / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.pkl")

        proba = model.predict_proba(X_test)[:, 1]
        (curves_path / f"{name}_roc.json").write_text(
            json.dumps(roc_curve_points(y_test, proba))
        )
        (curves_path / f"{name}_pr.json").write_text(
            json.dumps(pr_curve_points(y_test, proba))
        )
        logger.info(
            "%s: AUC=%.4f  AP=%.4f  Brier=%.4f",
            name,
            metrics[name]["auc_roc"],
            metrics[name]["avg_precision"],
            metrics[name]["brier_score"],
        )

    # Aggregate feature importance across models.
    importance_df = pd.DataFrame(importance_by_model, index=FEATURE_COLUMNS)
    importance_df["mean_importance"] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values("mean_importance", ascending=False)
    importance_df.index.name = "feature"

    low_signal = flag_low_signal_features(
        importance_df.reset_index(), threshold=LOW_SIGNAL_THRESHOLD
    )

    recommended = max(metrics, key=lambda m: metrics[m]["auc_roc"])

    # ---- write artifacts -------------------------------------------------- #
    eval_metrics = {
        "models": metrics,
        "recommended": recommended,
        "recommendation_reason": (
            f"{recommended} has the highest test AUC-ROC "
            f"({metrics[recommended]['auc_roc']:.4f}) with a Brier score of "
            f"{metrics[recommended]['brier_score']:.4f}."
        ),
    }
    (reports_path / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    importance_df.reset_index().to_json(
        str(reports_path / "feature_importance.json"), orient="records", indent=2
    )
    (reports_path / "low_signal_features.json").write_text(
        json.dumps(
            {"threshold": LOW_SIGNAL_THRESHOLD, "features": low_signal}, indent=2
        )
    )

    logger.info("Recommended model: %s", recommended)
    logger.info("Low-signal features flagged: %d", len(low_signal))
    return eval_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    evaluate_main()
