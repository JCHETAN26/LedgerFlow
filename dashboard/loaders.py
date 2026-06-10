"""
Artifact loaders for the LedgerFlow dashboard.

Pure, Streamlit-free functions that read the DVC-produced artifacts (metrics,
feature importance, curves, the feature matrix, raw events, trained models).
Kept separate from app.py so they can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Artifact locations relative to the project root.
ARTIFACTS = {
    "eval_metrics": "reports/eval_metrics.json",
    "feature_importance": "reports/feature_importance.json",
    "low_signal": "reports/low_signal_features.json",
    "feature_runtime": "reports/feature_runtime.json",
    "memo": "reports/recommendation_memo.md",
    "features": "data/processed/features.parquet",
    "events": "data/raw/events.parquet",
    "labels": "data/raw/labels.parquet",
    "curves": "reports/curves",
}


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk up from ``start`` until the directory containing params.yaml."""
    p = Path(start or Path.cwd()).resolve()
    while not (p / "params.yaml").exists() and p != p.parent:
        p = p.parent
    return p


class Artifacts:
    """Typed access to the pipeline's output artifacts under a project root."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else find_project_root()

    def path(self, key: str) -> Path:
        return self.root / ARTIFACTS[key]

    def available(self) -> dict[str, bool]:
        """Which artifacts currently exist on disk."""
        return {key: self.path(key).exists() for key in ARTIFACTS}

    # ---- JSON / text artifacts ------------------------------------------- #
    def load_eval_metrics(self) -> dict:
        return json.loads(self.path("eval_metrics").read_text())

    def load_feature_importance(self) -> pd.DataFrame:
        return pd.read_json(self.path("feature_importance"))

    def load_low_signal(self) -> dict:
        return json.loads(self.path("low_signal").read_text())

    def load_feature_runtime(self) -> dict:
        return json.loads(self.path("feature_runtime").read_text())

    def load_memo(self) -> str:
        return self.path("memo").read_text()

    # ---- tabular artifacts ----------------------------------------------- #
    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.path("features"))

    def load_events(self) -> pd.DataFrame:
        return pd.read_parquet(self.path("events"))

    def load_labels(self) -> pd.DataFrame:
        return pd.read_parquet(self.path("labels"))

    def load_curve(self, model: str, kind: str) -> dict:
        """Load a ROC ('roc') or PR ('pr') curve for a model."""
        return json.loads((self.path("curves") / f"{model}_{kind}.json").read_text())

    # ---- models ---------------------------------------------------------- #
    def list_models(self) -> list[str]:
        models_dir = self.root / "models"
        if not models_dir.exists():
            return []
        return sorted(
            p.name for p in models_dir.iterdir() if (p / "model.pkl").exists()
        )

    def load_model(self, name: str):
        import joblib

        return joblib.load(self.root / "models" / name / "model.pkl")


def metrics_table(eval_metrics: dict) -> pd.DataFrame:
    """Turn eval_metrics['models'] into a tidy DataFrame indexed by model."""
    df = pd.DataFrame(eval_metrics["models"]).T
    df.index.name = "model"
    ordered = [
        "auc_roc",
        "avg_precision",
        "brier_score",
        "log_loss",
        "precision_at_5pct_fpr",
        "f1",
    ]
    return df[[c for c in ordered if c in df.columns]]
