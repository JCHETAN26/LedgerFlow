#!/usr/bin/env python
"""
Generate the result charts embedded in the README from the pipeline artifacts.

Reads reports/eval_metrics.json, reports/curves/*.json, and
reports/feature_importance.json (produced by `dvc repro`) and writes PNGs into
assets/. Re-run after a new evaluation to refresh the figures.

    python scripts/make_readme_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"

# Consistent colour per model across every figure.
COLORS = {
    "LogisticRegression": "#6c757d",
    "XGBoost": "#1f77b4",
    "LightGBM": "#2ca02c",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def model_comparison(metrics: dict) -> None:
    models = list(metrics["models"])
    colors = [COLORS.get(m, "#888") for m in models]

    higher = ["auc_roc", "avg_precision", "precision_at_5pct_fpr", "f1"]
    lower = ["brier_score", "log_loss"]
    labels_h = ["AUC-ROC", "Avg precision", "Precision@5%FPR", "F1"]
    labels_l = ["Brier", "Log-loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    width = 0.25
    for i, m in enumerate(models):
        vals = [metrics["models"][m][k] for k in higher]
        x = [j + i * width for j in range(len(higher))]
        ax1.bar(x, vals, width, label=m, color=colors[i])
    ax1.set_xticks([j + width for j in range(len(higher))])
    ax1.set_xticklabels(labels_h, rotation=15)
    ax1.set_ylim(0, 1)
    ax1.set_title("Ranking & operating metrics (higher is better)")
    ax1.legend(fontsize=9)

    for i, m in enumerate(models):
        vals = [metrics["models"][m][k] for k in lower]
        x = [j + i * width for j in range(len(lower))]
        ax2.bar(x, vals, width, label=m, color=colors[i])
    ax2.set_xticks([j + width for j in range(len(lower))])
    ax2.set_xticklabels(labels_l)
    ax2.set_title("Calibration (lower is better)")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Model comparison — recommended: {metrics['recommended']}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def curves(metrics: dict, kind: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m in metrics["models"]:
        pts = json.loads((REPORTS / "curves" / f"{m}_{kind}.json").read_text())
        if kind == "roc":
            auc = metrics["models"][m]["auc_roc"]
            ax.plot(
                pts["fpr"],
                pts["tpr"],
                color=COLORS.get(m),
                label=f"{m} (AUC={auc:.3f})",
            )
        else:
            ap = metrics["models"][m]["avg_precision"]
            ax.plot(
                pts["recall"],
                pts["precision"],
                color=COLORS.get(m),
                label=f"{m} (AP={ap:.3f})",
            )
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left" if kind == "roc" else "best")
    fig.tight_layout()
    fig.savefig(ASSETS / filename, bbox_inches="tight")
    plt.close(fig)


def feature_importance(top_n: int = 15) -> None:
    imp = json.loads((REPORTS / "feature_importance.json").read_text())
    imp = sorted(imp, key=lambda r: r["mean_importance"], reverse=True)[:top_n]
    names = [r["feature"] for r in imp][::-1]
    vals = [r["mean_importance"] for r in imp][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="#2ca02c")
    ax.set_xlabel("Mean importance (across models)")
    ax.set_title(f"Top {top_n} features")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    metrics = _load("eval_metrics.json")
    model_comparison(metrics)
    curves(metrics, "roc", "roc_curves.png")
    curves(metrics, "pr", "pr_curves.png")
    feature_importance()
    print(f"Wrote charts to {ASSETS}")


if __name__ == "__main__":
    main()
