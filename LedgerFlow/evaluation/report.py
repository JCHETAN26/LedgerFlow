"""
Auto-generated recommendation memo (DVC ``report`` stage).

Reads the evaluation artifacts written by :mod:`LedgerFlow.evaluation.compare`
and renders ``reports/recommendation_memo.md`` from a Jinja2 template. The memo
is never written by hand — that is the whole point: the recommendation always
reflects the latest metrics.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "recommendation_memo.md.j2"


def generate_memo(
    reports_dir: str = "reports",
    output_path: str = "reports/recommendation_memo.md",
    top_n: int = 10,
    date: str | None = None,
) -> str:
    """Render the recommendation memo from evaluation artifacts.

    Args:
        reports_dir: Directory containing the evaluation JSON artifacts.
        output_path: Where to write the rendered Markdown memo.
        top_n: How many top features to list.
        date: Report date (defaults to today, ISO format).

    Returns:
        Path to the written memo.
    """
    reports = Path(reports_dir)
    eval_metrics = json.loads((reports / "eval_metrics.json").read_text())
    importance = json.loads((reports / "feature_importance.json").read_text())
    low_signal = json.loads((reports / "low_signal_features.json").read_text())

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)

    rendered = template.render(
        date=date or datetime.date.today().isoformat(),
        models=eval_metrics["models"],
        recommended=eval_metrics["recommended"],
        recommendation_reason=eval_metrics["recommendation_reason"],
        top_features=importance[:top_n],
        low_signal_features=low_signal["features"],
        low_signal_threshold=low_signal["threshold"],
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    logger.info("Wrote recommendation memo to %s", out)
    return str(out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    generate_memo()
