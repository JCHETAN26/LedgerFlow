#!/usr/bin/env python3
"""
Verify that the LedgerFlow project setup is correct.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_directory_structure():
    """Check that all required directories exist."""
    print("Checking directory structure...")
    
    required_dirs = [
        "LedgerFlow",
        "LedgerFlow/features",
        "LedgerFlow/evaluation",
        "data/raw",
        "data/processed", 
        "data/splits",
        "models",
        "reports",
        "tests/unit",
        "tests/integration",
        ".github/workflows",
        "notebooks",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"  ❌ Missing directory: {dir_path}")
            all_exist = False
        else:
            print(f"  ✓ Directory exists: {dir_path}")
    
    return all_exist


def check_required_files():
    """Check that all required files exist."""
    print("\nChecking required files...")
    
    required_files = [
        "pyproject.toml",
        "params.yaml",
        "README.md",
        ".gitignore",
        ".github/workflows/ci.yml",
        "LedgerFlow/__init__.py",
        "LedgerFlow/features/__init__.py",
        "LedgerFlow/features/base.py",
        "LedgerFlow/features/time_windows.py",
        "LedgerFlow/registry.py",
        "tests/conftest.py",
        "tests/test_structure.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"  ❌ Missing file: {file_path}")
            all_exist = False
        else:
            print(f"  ✓ File exists: {file_path}")
    
    return all_exist


def check_python_imports():
    """Check that Python modules can be imported."""
    print("\nChecking Python imports...")
    
    try:
        import LedgerFlow
        print(f"  ✓ Imported LedgerFlow (version: {LedgerFlow.__version__})")
        
        from LedgerFlow.features.base import BaseFeature
        print("  ✓ Imported BaseFeature")
        
        from LedgerFlow.features.time_windows import PurchaseCountWindow
        print("  ✓ Imported PurchaseCountWindow")
        
        from LedgerFlow.registry import FEATURE_REGISTRY, register_feature
        print("  ✓ Imported registry functions")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def check_feature_registry():
    """Check that features are properly registered."""
    print("\nChecking feature registry...")
    
    try:
        from LedgerFlow.features.time_windows import ALL_FEATURES
        from LedgerFlow.registry import FEATURE_REGISTRY, list_features
        
        print(f"  ✓ Created {len(ALL_FEATURES)} feature instances")
        print(f"  ✓ Registry contains {len(FEATURE_REGISTRY)} features")
        
        if FEATURE_REGISTRY:
            print("\n  Registered features:")
            for name, info in FEATURE_REGISTRY.items():
                print(f"    - {name}: {info['description']}")
        
        return len(FEATURE_REGISTRY) > 0
    except Exception as e:
        print(f"  ❌ Registry error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("LedgerFlow Project Setup Verification")
    print("=" * 60)
    
    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Python Imports", check_python_imports),
        ("Feature Registry", check_feature_registry),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n[{check_name}]")
        result = check_func()
        results.append((check_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! LedgerFlow setup is complete.")
        print("\nNext steps:")
        print("1. Create virtual environment: python -m venv .venv")
        print("2. Activate it: source .venv/bin/activate")
        print("3. Install dependencies: pip install -e '.[dev]'")
        print("4. Run tests: pytest tests/")
        print("5. Initialize DVC: dvc init")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()