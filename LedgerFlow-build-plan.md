# LedgerFlow — Build Plan

## Project Overview
A modular, tested, and versioned feature engineering library that transforms raw event
logs into machine-learning-ready signals. Generates 35 time-window aggregation features,
validates inputs with Pandera, evaluates models offline, and versions everything with DVC
for full experiment reproducibility.

---

## Tech Stack
| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Feature Engineering | scikit-learn, pandas, numpy |
| Data Validation | Pandera |
| Models | Logistic Regression (sklearn), XGBoost, LightGBM |
| Versioning | DVC |
| Storage | PostgreSQL (raw event logs), Apache Parquet (processed features) |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |
| Reporting | Jinja2 or Markdown (auto-generated evaluation memo) |
| Parallelism | joblib |

---

## Git Workflow
- **No direct pushes to `main`** — ever
- All changes go through a pull request
- Require at least 1 approving review before merge
- CI must pass (tests + DVC repro check) before merge is allowed
- Branch naming: `feat/`, `fix/`, `exp/`, `chore/`
- Example: `git checkout -b feat/rolling-std-feature`

---

## Phases

---

### Phase 0 — Environment & Repo Setup (Days 1–3)

**Goal:** Clean repo structure, DVC initialized, CI skeleton running before any feature code.

#### Tasks
- [ ] Create Python virtual environment (`Python 3.10+`)
- [ ] Install core dependencies:
  ```
  pandas numpy scikit-learn xgboost lightgbm dvc[s3]
  pandera pytest pytest-cov joblib sqlalchemy psycopg2
  pyarrow fastparquet jinja2
  ```
- [ ] Initialize Git repo and set branch protection on `main`:
  - No direct pushes to `main`
  - PRs required, 1 approving review minimum
  - CI must pass before merge
- [ ] Initialize DVC:
  ```bash
  dvc init
  dvc remote add -d myremote s3://LedgerFlow-artifacts/
  ```
- [ ] Create project repo structure:
  ```
  LedgerFlow/
  ├── LedgerFlow/
  │   ├── __init__.py
  │   ├── registry.py         # feature catalog + metadata
  │   ├── validators.py       # Pandera input schemas
  │   ├── features/
  │   │   ├── __init__.py
  │   │   ├── base.py         # BaseFeature abstract class
  │   │   ├── time_windows.py # all 35 time-window aggregations
  │   │   └── utils.py        # shared helpers
  │   ├── pipeline.py         # orchestrates feature generation
  │   └── evaluation/
  │       ├── __init__.py
  │       ├── metrics.py      # calibration, PR, feature importance
  │       ├── compare.py      # multi-model comparison runner
  │       └── report.py       # auto-generate evaluation memo
  ├── data/
  │   ├── raw/                # raw event logs (DVC-tracked)
  │   ├── processed/          # Parquet feature files (DVC-tracked)
  │   └── splits/             # train/val/test splits (DVC-tracked)
  ├── models/                 # trained model artifacts (DVC-tracked)
  ├── reports/                # auto-generated evaluation memos
  ├── tests/
  │   ├── unit/               # one test file per feature module
  │   ├── integration/        # pipeline end-to-end tests
  │   └── conftest.py
  ├── dvc.yaml                # DVC pipeline stages
  ├── params.yaml             # all configurable parameters
  ├── .github/workflows/
  │   └── ci.yml              # GitHub Actions CI
  └── notebooks/
      ├── 01_eda.ipynb
      └── 02_feature_analysis.ipynb
  ```
- [ ] Set up `params.yaml` with all configurable values:
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
- [ ] Create GitHub Actions CI skeleton (`.github/workflows/ci.yml`):
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v4
          with: { python-version: "3.10" }
        - run: pip install -e ".[dev]"
        - run: pytest tests/ --cov=LedgerFlow --cov-report=term-missing
        - run: dvc repro --dry  # verify pipeline stages are consistent
  ```

#### Success Criteria
- `pytest tests/` runs (0 tests, but no errors)
- `dvc status` shows clean state
- GitHub Actions CI triggers on a test PR and passes
- Branch protection blocks a direct push to `main`

---

### Phase 1 — Data Ingestion & Validation (Days 4–8)

**Goal:** Pull raw event logs from PostgreSQL, validate schema, write to Parquet.

#### 1.1 — PostgreSQL Connector
- [ ] Write `LedgerFlow/data/loader.py`:
  - Connect via SQLAlchemy + psycopg2
  - Query configurable date range and user subset
  - Return a typed pandas DataFrame
- [ ] Add connection config via environment variables (never hardcode credentials)
- [ ] Write integration test: `tests/integration/test_loader.py`

#### 1.2 — Input Schema Validation with Pandera
Define a strict schema for the raw event log:
```python
import pandera as pa

RawEventSchema = pa.DataFrameSchema({
    "event_id":        pa.Column(str,   nullable=False, unique=True),
    "user_id":         pa.Column(str,   nullable=False),
    "event_type":      pa.Column(str,   isin=["purchase", "login", "view", "click"]),
    "event_timestamp": pa.Column(pa.DateTime, nullable=False),
    "amount":          pa.Column(float, nullable=True,  checks=pa.Check.ge(0)),
    "session_id":      pa.Column(str,   nullable=True),
})
```
- [ ] Validate on every load — fail loudly, never silently
- [ ] Log validation errors to a report before raising
- [ ] Write unit tests for schema validation: missing cols, wrong types, out-of-range values

#### 1.3 — Raw → Parquet
- [ ] Write validated DataFrame to `data/raw/events.parquet` (partitioned by date)
- [ ] Track with DVC: `dvc add data/raw/events.parquet`
- [ ] Verify Parquet schema is stable across runs

#### Success Criteria
- Schema validation catches a manually injected bad row
- Parquet file written and tracked by DVC
- `dvc repro` re-runs ingestion stage cleanly

---

### Phase 2 — Feature Engineering Library (Days 9–22)

**Goal:** Build the modular library generating 35 time-window aggregations, with every
transformation independently testable and a developer adding a new feature in < 20 minutes.

#### 2.1 — BaseFeature Abstract Class
```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    name: str           # machine-readable name, e.g. "purchase_count_1h"
    description: str    # human-readable explanation
    output_dtype: str   # "float", "int", "bool"
    window: str         # e.g. "1h", "24h"

    @abstractmethod
    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        """Compute the feature for all users as of reference_time."""
        ...

    def validate_output(self, result: pd.Series) -> None:
        """Assert output has no nulls (unless nullable=True) and correct dtype."""
        ...
```
- [ ] Write `LedgerFlow/features/base.py`
- [ ] Any new feature is a class that inherits `BaseFeature` and implements `compute()`
- [ ] This is the 20-minute developer experience: subclass → implement → register → done

#### 2.2 — Time-Window Aggregation Features (35 total)
Build across 5 time windows × 7 aggregations:

**Time windows:** 1h, 6h, 24h, 7d, 30d
**Aggregations per window:**

| Aggregation | Feature Example |
|---|---|
| `count` | `purchase_count_24h` — number of events in window |
| `sum` | `purchase_amount_sum_24h` — total spend in window |
| `mean` | `purchase_amount_mean_24h` — average transaction value |
| `std` | `purchase_amount_std_24h` — volatility of spend |
| `min` | `purchase_amount_min_24h` — smallest transaction |
| `max` | `purchase_amount_max_24h` — largest transaction |
| `last` | `purchase_amount_last_24h` — most recent transaction value |

Implementation pattern:
```python
class PurchaseCountWindow(BaseFeature):
    def __init__(self, window: str):
        self.window = window
        self.name = f"purchase_count_{window}"
        self.description = f"Number of purchase events in the last {window}"
        self.output_dtype = "int"

    def compute(self, df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        cutoff = reference_time - pd.Timedelta(self.window)
        filtered = df[(df["event_type"] == "purchase") & (df["event_timestamp"] >= cutoff)]
        return filtered.groupby("user_id").size().rename(self.name)
```

- [ ] Implement all 35 features in `LedgerFlow/features/time_windows.py`
- [ ] Each feature is its own class (or parameterised factory — your choice)
- [ ] Write at least 2 unit tests per feature: happy path + edge case (empty window, single event)
- [ ] **60+ unit tests total** — one test file per feature group in `tests/unit/`

#### 2.3 — Feature Pipeline
```python
# LedgerFlow/pipeline.py
class FeaturePipeline:
    def __init__(self, features: list[BaseFeature]):
        self.features = features

    def run(self, df: pd.DataFrame, reference_time: pd.Timestamp,
            n_jobs: int = -1) -> pd.DataFrame:
        """Compute all features in parallel, return wide DataFrame."""
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(feat.compute)(df, reference_time)
            for feat in self.features
        )
        return pd.concat(results, axis=1).fillna(0)
```
- [ ] `n_jobs=-1` uses all available CPU cores
- [ ] Fill nulls with 0 (users with no events in a window had 0 activity)
- [ ] Output written to `data/processed/features.parquet`
- [ ] Benchmark: log runtime for N users to `params.yaml` / DVC metrics

#### 2.4 — Feature Registry / Catalog
```python
# LedgerFlow/registry.py
FEATURE_REGISTRY = {
    feat.name: {
        "description": feat.description,
        "window":       feat.window,
        "dtype":        feat.output_dtype,
        "added_by":     "...",
        "added_date":   "...",
    }
    for feat in ALL_FEATURES
}
```
- [ ] Auto-generated from class metadata — no manual documentation required
- [ ] Exportable to Markdown table for team review
- [ ] Write test: every registered feature has non-empty description

#### Success Criteria
- All 35 features compute without errors on the IEEE-CIS dataset or a synthetic event log
- 60+ unit tests pass with > 90% code coverage on `LedgerFlow/features/`
- Feature pipeline runtime benchmarked and logged
- A new feature can be added end-to-end in < 20 minutes (timed)

---

### Phase 3 — Training-Serving Consistency (Days 23–26)

**Goal:** Ensure features computed at training time are identical to features computed at inference.

#### 3.1 — The Problem
Training features are computed over historical windows. Inference features are computed
for a single user at a single point in time. If these use different code paths, you get
training-serving skew — the model sees different inputs in production than it trained on.

#### 3.2 — Solution: Single `transform()` Entry Point
```python
class FeaturePipeline:
    def transform_batch(self, df, reference_time) -> pd.DataFrame:
        """For training: compute features for all users."""
        ...

    def transform_single(self, user_history: pd.DataFrame,
                          reference_time: pd.Timestamp) -> dict:
        """For inference: compute features for one user."""
        result = self.run(user_history, reference_time, n_jobs=1)
        return result.to_dict(orient="records")[0]
```

- [ ] Both methods call the same `BaseFeature.compute()` — no duplicated logic
- [ ] Write a parity test: compute features for 10 users via `transform_batch`, then
  via `transform_single` one at a time — assert outputs are identical

#### Success Criteria
- Parity test passes: batch and single-user outputs match to 6 decimal places
- No feature has a separate "online" vs "offline" implementation

---

### Phase 4 — DVC Pipeline & Versioning (Days 27–31)

**Goal:** Every artifact — raw data, features, models, eval results — is versioned and
reproducible from a single `dvc repro` command.

#### 4.1 — DVC Pipeline Stages (`dvc.yaml`)
```yaml
stages:
  ingest:
    cmd: python -m LedgerFlow.data.loader
    deps: [LedgerFlow/data/loader.py, params.yaml]
    outs: [data/raw/events.parquet]

  featurize:
    cmd: python -m LedgerFlow.pipeline
    deps: [LedgerFlow/pipeline.py, LedgerFlow/features/, data/raw/events.parquet]
    outs: [data/processed/features.parquet]
    metrics: [reports/feature_runtime.json]

  split:
    cmd: python -m LedgerFlow.data.splitter
    deps: [data/processed/features.parquet, params.yaml]
    outs: [data/splits/train.parquet, data/splits/val.parquet, data/splits/test.parquet]

  evaluate:
    cmd: python -m LedgerFlow.evaluation.compare
    deps: [LedgerFlow/evaluation/, data/splits/]
    outs: [models/]
    metrics: [reports/eval_metrics.json]

  report:
    cmd: python -m LedgerFlow.evaluation.report
    deps: [reports/eval_metrics.json]
    outs: [reports/recommendation_memo.md]
```

- [ ] `dvc repro` runs all stages in dependency order
- [ ] Stages only re-run when their inputs change (hash-based caching)
- [ ] All outputs tracked: `dvc push` sends artifacts to S3
- [ ] `dvc pull` restores full experiment state on a new machine

#### 4.2 — Experiment Lineage
- [ ] Tag each DVC experiment with a Git commit hash
- [ ] `dvc params diff` shows what changed between runs
- [ ] `dvc metrics diff` shows how metrics changed
- [ ] Write a `scripts/reproduce_experiment.sh`:
  ```bash
  git checkout <commit-hash>
  dvc pull
  dvc repro
  ```

#### Success Criteria
- Full pipeline runs end-to-end with `dvc repro` from a clean checkout
- `dvc metrics show` displays eval metrics for the current run
- `dvc params diff HEAD~1` correctly shows any parameter changes

---

### Phase 5 — Offline Model Evaluation (Days 32–40)

**Goal:** Compare Logistic Regression, XGBoost, and LightGBM across calibration,
precision-recall, and feature importance — then auto-generate the recommendation memo.

#### 5.1 — Model Training
For each of 3 models:
- [ ] Train on `data/splits/train.parquet`
- [ ] Validate on `data/splits/val.parquet` (for early stopping in XGBoost/LightGBM)
- [ ] Save model artifacts to `models/<model_name>/`
- [ ] Track all artifacts with DVC

#### 5.2 — Evaluation Metrics
For each model, compute on the held-out test set:

```python
def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred  = model.predict(X_test)
    return {
        "auc_roc":            roc_auc_score(y_test, proba),
        "avg_precision":      average_precision_score(y_test, proba),
        "brier_score":        brier_score_loss(y_test, proba),  # calibration
        "log_loss":           log_loss(y_test, proba),          # calibration
        "precision_at_5pct_fpr": precision_at_fpr(y_test, proba, fpr=0.05),
        "f1":                 f1_score(y_test, pred),
    }
```

- [ ] Brier score and log loss measure calibration — critical for probability outputs
- [ ] Precision-recall curve saved as artifact for each model
- [ ] ROC curve saved as artifact
- [ ] Feature importance extracted: SHAP for XGBoost/LightGBM, coefficients for LR
- [ ] All metrics written to `reports/eval_metrics.json` (DVC-tracked)

#### 5.3 — Feature Importance Feedback Loop
After evaluation, automatically flag weak features:
```python
def flag_low_signal_features(importance_df, threshold=0.001) -> list[str]:
    """Return features with near-zero importance across all models."""
    return importance_df[importance_df["mean_importance"] < threshold]["feature"].tolist()
```
- [ ] Log flagged features to `reports/low_signal_features.json`
- [ ] These are candidates for removal in the next feature iteration

#### 5.4 — Auto-Generated Recommendation Memo
Replace the manual memo with a generated one:
```
reports/recommendation_memo.md (auto-generated)

# Model Evaluation Report — {date}

## Summary
| Model | AUC-ROC | Avg Precision | Brier Score | Recommended |
|---|---|---|---|---|
| LogisticRegression | 0.81 | 0.74 | 0.12 | ❌ |
| XGBoost            | 0.89 | 0.83 | 0.09 | ✅ |
| LightGBM           | 0.88 | 0.82 | 0.09 | — |

## Recommendation
XGBoost is recommended for production A/B test based on highest AUC-ROC (0.89)
and best calibration (Brier score: 0.09).

## Top 10 Features (XGBoost)
...

## Low-Signal Features (candidates for removal)
...
```
- [ ] Write `LedgerFlow/evaluation/report.py` using Jinja2 template
- [ ] Memo committed to `reports/` and DVC-tracked
- [ ] CI fails if memo is outdated relative to eval metrics

#### Success Criteria
- All 3 models evaluated and compared in a single `dvc repro` run
- Recommendation memo generated automatically — zero manual writing
- Low-signal feature list produced after every eval run
- `dvc metrics diff` shows metric changes between experiment runs

---

### Phase 6 — CI Polish & Developer Experience (Days 41–45)

**Goal:** Make contributing to the library fast and safe.

#### 6.1 — Full CI Pipeline
```yaml
# .github/workflows/ci.yml
jobs:
  test:
    steps:
      - run: pytest tests/unit/ --cov=LedgerFlow --cov-fail-under=90
      - run: pytest tests/integration/
      - run: dvc repro --dry

  lint:
    steps:
      - run: ruff check LedgerFlow/
      - run: mypy LedgerFlow/ --strict

  feature-registry:
    steps:
      - run: python -c "from LedgerFlow.registry import validate_registry; validate_registry()"
```
- [ ] CI blocks merge if test coverage drops below 90%
- [ ] CI blocks merge if any feature in the pipeline is missing from the registry
- [ ] Lint and type checks run on every PR

#### 6.2 — Developer Onboarding Script
Write `scripts/new_feature.py` — a CLI scaffold that:
- Prompts for feature name, window, aggregation type
- Generates the boilerplate class file
- Generates the boilerplate test file
- Registers the feature in the registry
- Prints "Now implement `compute()` and run `pytest`"

This is what keeps new feature development at < 20 minutes.

#### 6.3 — Runtime Benchmark as a CI Check
- [ ] Run feature pipeline on a fixed 10K-row synthetic dataset in CI
- [ ] Assert total runtime < 30 seconds on that dataset
- [ ] Track runtime trend in DVC metrics over time

#### Success Criteria
- CI passes end-to-end on a fresh PR with a new dummy feature
- New developer can add a real feature in < 20 minutes using the scaffold script
- Test coverage ≥ 90% enforced by CI

---

## Overall Timeline Summary

| Phase | Duration | Cumulative |
|---|---|---|
| 0 — Setup | 3 days | Day 3 |
| 1 — Data & Validation | 5 days | Day 8 |
| 2 — Feature Library | 14 days | Day 22 |
| 3 — Training-Serving Consistency | 4 days | Day 26 |
| 4 — DVC Pipeline | 5 days | Day 31 |
| 5 — Offline Evaluation | 9 days | Day 40 |
| 6 — CI & Developer Experience | 5 days | Day 45 |

**Total: ~6.5 weeks solo at mid-level pace**

---

## Key Metrics to Track

| Metric | Target |
|---|---|
| Unit tests | ≥ 60, all passing |
| Code coverage | ≥ 90% on `LedgerFlow/` |
| Feature count | 35 time-window aggregations |
| New feature dev time | < 20 minutes |
| Pipeline runtime (10K rows) | < 30 seconds |
| Batch/single parity test | Outputs match to 6 decimal places |
| DVC repro from clean checkout | Must succeed without manual steps |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Training-serving skew | Single `compute()` path, enforced by parity test in CI |
| Feature pipeline too slow | joblib parallelism; benchmark tracked in DVC metrics |
| Upstream schema changes break features | Pandera validation fails loudly at ingestion, not silently downstream |
| DVC pipeline gets out of sync | `dvc repro --dry` in CI catches stale stages before merge |
| Low-quality features accumulate | Automatic feature importance flagging after every eval run |
| Manual memo gets stale | Report auto-generated by `dvc repro` — no human step |
