"""
Central loader for ``params.yaml``.

A single source of truth for configurable values (data columns, feature windows,
evaluation settings) so DVC stages and library code never drift apart.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# params.yaml lives at the repository root, one level above this package.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


@functools.cache
def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load and cache the parsed ``params.yaml``.

    Args:
        path: Path to the params file (defaults to the repo-root params.yaml).

    Returns:
        Nested dict of parameters.
    """
    with open(path) as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
