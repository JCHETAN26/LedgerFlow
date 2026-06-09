"""
Test to verify the basic project structure and imports.
"""

def test_import_ledgerflow():
    """Test that LedgerFlow can be imported."""
    import LedgerFlow
    assert LedgerFlow.__version__ == "0.1.0"


def test_project_structure():
    """Test that essential source directories exist.

    Only source directories are checked. Runtime artifact directories (data/,
    models/, reports/curves) are DVC-managed and absent on a clean checkout
    until `dvc repro` runs, so they are intentionally not required here.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    required_dirs = [
        "LedgerFlow",
        "LedgerFlow/data",
        "LedgerFlow/features",
        "LedgerFlow/evaluation",
        "tests",
        "tests/unit",
        "tests/integration",
    ]

    for dir_path in required_dirs:
        assert (root / dir_path).is_dir(), f"Directory {dir_path} does not exist"