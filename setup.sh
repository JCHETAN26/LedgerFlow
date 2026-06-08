#!/bin/bash
# LedgerFlow Setup Script
# Run this script to set up the development environment

set -e

echo "==========================================="
echo "LedgerFlow Development Environment Setup"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the LedgerFlow project root"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python3 --version | grep -q "Python 3\.10\|Python 3\.11\|Python 3\.12" || {
    echo "Error: Python 3.10+ is required"
    echo "Found: $(python3 --version)"
    exit 1
}

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists at .venv/"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Verify installation
echo "Verifying installation..."
python3 -c "import LedgerFlow; print(f'Successfully imported LedgerFlow {LedgerFlow.__version__}')"
python3 -c "from LedgerFlow.features.base import BaseFeature; print('Successfully imported BaseFeature')"

# Run verification script
echo "Running verification script..."
python3 verify_setup.py

echo ""
echo "==========================================="
echo "Setup Complete! 🎉"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Your virtual environment is activated at .venv/"
echo "2. To activate in a new shell: source .venv/bin/activate"
echo "3. Run tests: pytest tests/"
echo "4. Initialize DVC (when ready): dvc init"
echo ""
echo "Development commands:"
echo "  pytest tests/              # Run all tests"
echo "  pytest tests/ --cov        # Run tests with coverage"
echo "  python verify_setup.py     # Verify project setup"
echo "  pip install -e '.[dev]'    # Reinstall in dev mode"
echo ""