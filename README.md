# LedgerFlow

A modular, tested, and versioned feature engineering library that transforms raw event logs into machine-learning-ready signals.

## Overview

LedgerFlow is a Python library that:
- Pulls raw event logs from PostgreSQL (with an offline synthetic generator for CI/dev)
- Validates the schema strictly with Pandera before any processing
- Generates 35 time-window aggregation features (5 windows × 7 aggregations)
- Computes features **point-in-time** — each user as of their own `decision_time`, so no feature leaks information from after the moment being predicted
- Computes batch (training) and single-user (inference) features through one code path — no training-serving skew
- Runs offline model evaluation across Logistic Regression, XGBoost, and LightGBM
- Auto-generates a recommendation memo comparing models on calibration and precision-recall
- Versions every artifact — data, features, models, metrics — with DVC for full reproducibility

> **Data source note:** the DVC `ingest` stage uses a deterministic synthetic
> event-log generator (`LedgerFlow.data.synthetic`) so `dvc repro` and CI run
> without a database. In production this stage is swapped for the PostgreSQL
> loader (`LedgerFlow.data.loader`); both emit the same Pandera-validated schema.

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
│   ├── params.py           # params.yaml loader
│   ├── registry.py         # Feature catalog + metadata
│   ├── pipeline.py         # FeaturePipeline (batch + single-user)
│   ├── data/
│   │   ├── loader.py       # PostgreSQL loader (production ingest)
│   │   ├── synthetic.py    # Offline synthetic event-log generator
│   │   ├── validators.py   # Pandera input schema + business rules
│   │   ├── ingest.py       # Postgres → Parquet orchestrator
│   │   └── splitter.py     # Time-based train/val/test split
│   ├── features/
│   │   ├── base.py         # BaseFeature abstract class
│   │   └── time_windows.py # 35 time-window aggregation features
│   └── evaluation/
│       ├── metrics.py      # Calibration, PR, precision@FPR
│       ├── compare.py      # Multi-model comparison runner
│       ├── report.py       # Jinja2 recommendation-memo generator
│       └── templates/      # Memo template
├── scripts/
│   ├── new_feature.py      # Scaffold a new feature in < 20 min
│   └── reproduce_experiment.sh
├── data/{raw,processed,splits}/   # DVC-tracked artifacts
├── models/                 # Trained model artifacts (DVC-tracked)
├── reports/                # Metrics JSON + auto-generated memo
├── tests/{unit,integration}/
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # All configurable parameters
└── .github/workflows/ci.yml
```

### Run the pipeline (offline, no database)

```bash
# Reproduce the full pipeline: ingest → featurize → split → evaluate → report
dvc repro

# View model metrics and the auto-generated recommendation
dvc metrics show
cat reports/recommendation_memo.md
```

## Development

### Adding a New Feature (< 20 minutes)

Use the scaffold to generate a ready-to-implement feature class and its test:

```bash
python scripts/new_feature.py \
  --name session_count_24h \
  --window 24h \
  --description "Number of distinct sessions in the last 24h"
```

This writes `LedgerFlow/features/custom/session_count_24h.py` (a `BaseFeature`
subclass that registers itself) and `tests/unit/test_session_count_24h.py`. Then:

1. Implement `compute()` — return a `pd.Series` indexed by `user_id`, named `self.name`.
2. Fill in the test assertions (happy path + edge case).
3. Run: `pytest tests/unit/test_session_count_24h.py -v`

Every feature registers itself on construction, so the catalog and the CI
registry check stay in sync automatically.

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=LedgerFlow --cov-report=term-missing

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### DVC Pipeline

The default remote is an S3 bucket (`s3://ledgerflow-artifacts-049000283645/`,
configured in `.dvc/config`). `dvc push`/`pull` require AWS credentials with
access to it.

```bash
# Run the full pipeline (self-contained — synthetic ingest, no DB needed)
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