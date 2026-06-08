"""
Test to verify the basic project structure and imports.
"""

def test_import_ledgerflow():
    """Test that LedgerFlow can be imported."""
    import LedgerFlow
    assert LedgerFlow.__version__ == "0.1.0"


def test_project_structure():
    """Test that essential directories exist."""
    import os
    
    required_dirs = [
        "LedgerFlow",
        "LedgerFlow/features", 
        "LedgerFlow/evaluation",
        "data",
        "tests",
        "tests/unit",
        "tests/integration",
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"