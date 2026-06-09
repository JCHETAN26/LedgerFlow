#!/usr/bin/env python
"""
Scaffold a new LedgerFlow feature in under 20 minutes.

Generates a ready-to-implement BaseFeature subclass plus a matching test file,
so a developer only has to fill in ``compute()`` and run pytest. This is the
mechanism behind the project's "new feature in < 20 minutes" goal.

Usage:
    python scripts/new_feature.py --name session_count_24h --window 24h \
        --description "Number of distinct sessions in the last 24h"

Run with no arguments for interactive prompts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "LedgerFlow" / "features" / "custom"
TESTS_DIR = REPO_ROOT / "tests" / "unit"

FEATURE_TEMPLATE = '''"""
{description}

Auto-scaffolded by scripts/new_feature.py — implement compute() below.
"""

import pandas as pd

from ..base import BaseFeature
from ...registry import register_feature


class {class_name}(BaseFeature):
    """{description}"""

    def __init__(self) -> None:
        self.name = "{name}"
        self.description = "{description}"
        self.output_dtype = "{dtype}"
        self.window = "{window}"
        self.nullable = {nullable}
        register_feature(self)

    def compute(
        self, df: pd.DataFrame, reference_time: pd.Timestamp
    ) -> pd.Series:
        # TODO: implement. Return a Series indexed by user_id, named self.name,
        # computed over events in (reference_time - window, reference_time].
        cutoff = reference_time - pd.Timedelta(self.window)
        window = df[
            (df["event_timestamp"] > cutoff)
            & (df["event_timestamp"] <= reference_time)
        ]
        raise NotImplementedError("Implement {class_name}.compute()")
'''

TEST_TEMPLATE = '''"""
Tests for {class_name} ({name}).
"""

import pandas as pd
import pytest

from LedgerFlow.features.custom.{module} import {class_name}

REFERENCE = pd.Timestamp("2024-01-15 12:00:00")


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {{
            "user_id": ["a", "a", "b"],
            "event_type": ["purchase", "login", "purchase"],
            "event_timestamp": [
                REFERENCE - pd.Timedelta("10min"),
                REFERENCE - pd.Timedelta("20min"),
                REFERENCE - pd.Timedelta("30min"),
            ],
            "amount": [10.0, None, 5.0],
        }}
    )


def test_happy_path():
    feature = {class_name}()
    result = feature.compute(_events(), REFERENCE)
    assert result.name == "{name}"
    # TODO: assert expected values.


def test_empty_window():
    feature = {class_name}()
    empty = _events().iloc[0:0]
    result = feature.compute(empty, REFERENCE)
    assert result.empty
'''


def to_class_name(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\W]+", name) if part)


def _prompt(arg: str | None, label: str, default: str = "") -> str:
    if arg is not None:
        return arg
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new LedgerFlow feature")
    parser.add_argument("--name", help="machine name, e.g. session_count_24h")
    parser.add_argument("--window", help="time window, e.g. 24h, 7d")
    parser.add_argument("--description", help="human-readable description")
    parser.add_argument(
        "--dtype", default="float", choices=["float", "int", "bool"]
    )
    parser.add_argument("--nullable", action="store_true", help="allow null output")
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files"
    )
    args = parser.parse_args(argv)

    name = _prompt(args.name, "Feature name (snake_case)")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        print(f"Invalid feature name: {name!r} (use snake_case)", file=sys.stderr)
        return 1
    window = _prompt(args.window, "Time window", "24h")
    description = _prompt(args.description, "Description", f"{name} feature")

    class_name = to_class_name(name)
    module = name

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    (FEATURES_DIR / "__init__.py").touch()

    feature_path = FEATURES_DIR / f"{module}.py"
    test_path = TESTS_DIR / f"test_{module}.py"

    for path in (feature_path, test_path):
        if path.exists() and not args.force:
            print(f"Refusing to overwrite {path} (use --force)", file=sys.stderr)
            return 1

    feature_path.write_text(
        FEATURE_TEMPLATE.format(
            class_name=class_name,
            name=name,
            description=description,
            dtype=args.dtype,
            window=window,
            nullable=args.nullable,
        )
    )
    test_path.write_text(
        TEST_TEMPLATE.format(
            class_name=class_name, name=name, module=module
        )
    )

    print(f"✅ Created {feature_path.relative_to(REPO_ROOT)}")
    print(f"✅ Created {test_path.relative_to(REPO_ROOT)}")
    print()
    print("Next steps:")
    print(f"  1. Implement {class_name}.compute() in {feature_path.name}")
    print(f"  2. Fill in the assertions in {test_path.name}")
    print(f"  3. Run: pytest {test_path.relative_to(REPO_ROOT)} -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
