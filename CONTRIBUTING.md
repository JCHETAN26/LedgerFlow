# Contributing to LedgerFlow

Thank you for your interest in contributing to LedgerFlow! This document provides guidelines and instructions for contributing.

## Development Workflow

### Git Workflow

1. **No direct pushes to `main`** — all changes go through pull requests
2. Create a feature branch: `git checkout -b feat/description`
3. Make your changes and commit: `git commit -m "feat: description of changes"`
4. Push to your branch: `git push origin feat/description`
5. Open a pull request on GitHub

### Branch Naming Convention

- `feat/` - New features or enhancements
- `fix/` - Bug fixes
- `exp/` - Experimental changes
- `chore/` - Maintenance tasks, documentation, etc.
- `docs/` - Documentation updates

Examples:
- `feat/rolling-std-feature`
- `fix/null-handling-purchase-events`
- `docs/update-api-documentation`

### Commit Message Convention

We follow a simplified version of Conventional Commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add purchase count feature for 24h window`

## Code Standards

### Python Style Guide

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before committing:

```bash
ruff check LedgerFlow/ --fix
```

### Type Hints

All function signatures should include type hints:

```python
def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
    # ...
```

### Testing

- Write tests for all new features
- Aim for 90%+ code coverage
- Tests should be in `tests/unit/` or `tests/integration/`
- Run tests: `pytest tests/ --cov=LedgerFlow`

### Adding a New Feature

1. Create a new class in `LedgerFlow/features/` that inherits from `BaseFeature`
2. Implement the `compute()` method
3. Register the feature automatically (see examples in `time_windows.py`)
4. Write unit tests in `tests/unit/`
5. Ensure the feature appears in the registry

Example:

```python
from LedgerFlow.features.base import BaseFeature
from LedgerFlow.registry import register_feature

class MyFeature(BaseFeature):
    def __init__(self, window: str):
        self.window = window
        self.name = f"my_feature_{window}"
        self.description = f"My feature for {window} window"
        self.output_dtype = "float"
        self.nullable = False
        register_feature(self)  # Important!
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        # Your implementation
        pass
```

## Pull Request Process

1. Ensure all tests pass: `pytest tests/`
2. Run linting: `ruff check LedgerFlow/`
3. Update documentation if needed
4. Fill out the PR template with:
   - Description of changes
   - Testing performed
   - Any breaking changes
   - Related issues

## Development Setup

1. Clone the repository
2. Run setup script: `./setup.sh`
3. Activate virtual environment: `source .venv/bin/activate`
4. Install in development mode: `pip install -e ".[dev]"`

## DVC Guidelines

- All data artifacts must be tracked with DVC
- Never commit raw data files to Git
- Use `dvc add` for new data files
- Run `dvc repro --dry` before committing to check pipeline consistency

## Questions?

If you have questions about contributing, please:
1. Check the existing documentation
2. Look at existing examples in the codebase
3. Open an issue for discussion