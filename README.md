# LedgerFlow

A modular, tested, and versioned feature engineering library that transforms raw event logs into machine-learning-ready signals.

## Overview

LedgerFlow is a Python library that:
- Pulls raw event logs from PostgreSQL
- Validates the schema strictly with Pandera before any processing
- Generates 35 time-window aggregation features (5 windows × 7 aggregations)
- Runs offline model evaluation across Logistic Regression, XGBoost, and LightGBM
- Auto-generates a recommendation memo comparing models on calibration and precision-recall
- Versions every artifact — data, features, models, metrics — with DVC for full reproducibility

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JCHETAN26/LedgerFlow.git
cd LedgerFlow

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Project Structure

```
LedgerFlow/
├── LedgerFlow/              # Main Python package
│   ├── __init__.py
│   ├── registry.py         # Feature catalog + metadata
│   ├── validators.py       # Pandera input schemas
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py         # BaseFeature abstract class
│   │   ├── time_windows.py # Time-window aggregation features
│   │   └── utils.py        # Shared helpers
│   ├── pipeline.py         # Orchestrates feature generation
│   └── evaluation/
│       ├── metrics.py      # Calibration, PR, feature importance
│       ├── compare.py      # Multi-model comparison runner
│       └── report.py       # Auto-generate evaluation memo
├── data/
│   ├── raw/                # Raw event logs (DVC-tracked)
│   ├── processed/          # Parquet feature files (DVC-tracked)
│   └── splits/             # Train/val/test splits (DVC-tracked)
├── models/                 # Trained model artifacts (DVC-tracked)
├── reports/                # Auto-generated evaluation memos
├── tests/
│   ├── unit/               # One test file per feature module
│   ├── integration/        # Pipeline end-to-end tests
│   └── conftest.py
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # All configurable parameters
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
└── notebooks/
    ├── 01_eda.ipynb
    └── 02_feature_analysis.ipynb
```

## Development

### Adding a New Feature

1. Create a new class inheriting from `BaseFeature`:
```python
from LedgerFlow.features.base import BaseFeature

class MyNewFeature(BaseFeature):
    def __init__(self, window: str):
        self.window = window
        self.name = f"my_feature_{window}"
        self.description = f"My new feature for {window} window"
        self.output_dtype = "float"
        self.nullable = False
    
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        # Your implementation here
        pass
```

2. Register the feature (automatically done if using the factory pattern)
3. Write unit tests in `tests/unit/`
4. Run tests: `pytest tests/unit/test_my_feature.py`

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=LedgerFlow --cov-report=term-missing

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### DVC Pipeline

```bash
# Initialize DVC (first time only)
dvc init
dvc remote add -d myremote s3://LedgerFlow-artifacts/

# Run the full pipeline
dvc repro

# Check what would re-run
dvc repro --dry

# Sync artifacts with remote storage
dvc push
dvc pull
```

## Configuration

All configurable parameters are in `params.yaml`:

```yaml
data:
  raw_table: event_logs
  date_col: event_timestamp
  user_col: user_id

features:
  time_windows: [1h, 6h, 24h, 7d, 30d]
  aggregations: [count, sum, mean, std, min, max, last]

evaluation:
  test_size: 0.2
  val_size: 0.1
  cv_folds: 5
  random_seed: 42
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Tests**: Runs pytest with 90% coverage requirement
- **Linting**: Uses ruff for code style and mypy for type checking
- **DVC**: Validates pipeline consistency with `dvc repro --dry`
- **Feature Registry**: Ensures all features are properly registered

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.