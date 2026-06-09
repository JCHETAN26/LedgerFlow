"""
Feature registry and catalog for LedgerFlow.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import only for type checking to avoid a circular import:
    # features -> registry -> features.
    from .features.base import BaseFeature

# This will be populated dynamically as features are registered
FEATURE_REGISTRY: dict[str, dict[str, Any]] = {}


def register_feature(feature: BaseFeature) -> None:
    """Register a feature in the global registry.

    Args:
        feature: An instance of a BaseFeature subclass
    """
    FEATURE_REGISTRY[feature.name] = {
        "description": feature.description,
        "window": feature.window,
        "dtype": feature.output_dtype,
        "nullable": getattr(feature, "nullable", False),
        "added_by": getattr(feature, "added_by", "ledgerflow"),
        # Deterministic by default so the registry export is reproducible in CI.
        "added_date": getattr(
            feature, "added_date", datetime.date.today().isoformat()
        ),
    }


def get_feature_info(feature_name: str) -> dict[str, Any]:
    """Get information about a registered feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Dictionary with feature metadata

    Raises:
        KeyError: If feature is not registered
    """
    if feature_name not in FEATURE_REGISTRY:
        raise KeyError(f"Feature '{feature_name}' not found in registry")
    return FEATURE_REGISTRY[feature_name]


def list_features() -> dict[str, dict[str, Any]]:
    """List all registered features.

    Returns:
        Dictionary mapping feature names to their metadata
    """
    return FEATURE_REGISTRY.copy()


def validate_registry() -> bool:
    """Validate that all features in the registry are properly registered.

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if not FEATURE_REGISTRY:
        raise ValueError("Feature registry is empty")

    required_keys = {"description", "window", "dtype", "added_by", "added_date"}

    for feature_name, metadata in FEATURE_REGISTRY.items():
        missing_keys = required_keys - set(metadata.keys())
        if missing_keys:
            raise ValueError(
                f"Feature '{feature_name}' missing required metadata: {missing_keys}"
            )

    return True


def export_to_markdown() -> str:
    """Export feature registry as a Markdown table.

    Returns:
        Markdown string with feature catalog
    """
    if not FEATURE_REGISTRY:
        return "# Feature Registry\n\nNo features registered yet."

    headers = ["Feature", "Description", "Window", "Type", "Nullable", "Added"]
    rows = []

    for name, info in sorted(FEATURE_REGISTRY.items()):
        rows.append([
            f"`{name}`",
            info["description"],
            info["window"],
            info["dtype"],
            str(info.get("nullable", False)),
            info["added_date"][:10],  # Just the date part
        ])

    # Create markdown table
    md_lines = ["# Feature Registry\n"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)
