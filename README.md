# MLBV1 — MLB Spread Prediction Pipeline

> **v1.0.0** — Production-ready MLB prediction system.

MLBV1 is a multi-market MLB prediction pipeline using modular data ingestion, 27-feature engineering, multi-output regression ensembles, and Poisson/Skellam probability derivation. Includes full test suite, Docker, CI/CD, and Azure Container Apps deployment.

## Features

- **Data loaders** — Odds API, BetsAPI, MLB Stats API (free), CSV/JSON, synthetic
- **Weather enrichment** — Visual Crossing integration for game-time conditions
- **27 engineered features** — rolling stats, pitcher metrics, rest days, Elo, weather, devigged odds
- **4 model types** — RandomForest, Ridge, XGBoost, LightGBM (multi-output regression)
- **7 markets** — Moneyline, Spread, Total, F5 Moneyline, F5 Spread, F5 Total, Team Totals
- **Poisson/Skellam** probability derivation for all markets
- **Kelly criterion** bet sizing (quarter-Kelly)
- **Ensemble** voting with optional weighting
- **Evaluation** — accuracy, ROI, Sharpe ratio
- **SQLite tracking** — predictions, settlement, bankroll, model registry
- **Canonical slate SSOT** — validated slate rows are atomically published to SQLite (`published_slate`)
- **Data quality gates + audit manifest** — run-level checks and `artifacts/data_audit_YYYY-MM-DD.json`
- **Alerts** — Discord webhooks, email (SMTP)
- **Docker + GitHub Actions CI**
- **Azure deployment** via Bicep / `azd up`

## Requirements

- Python 3.12+

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# . .venv/bin/activate     # macOS/Linux
pip install -e ".[dev]"
```

## Usage

### Train

```bash
python scripts/train.py --loader synthetic --model all
```

### Predict

```bash
python scripts/predict.py --model-path artifacts/models/random_forest.pkl --loader synthetic
```

### Daily Run

```bash
python scripts/daily_run.py --loader odds_api
```

## Configuration

Configuration via JSON file or environment variables. Copy `.env.example` to `.env` and fill in your keys.

```bash
python scripts/train.py --config config.json
```

Example JSON:

```json
{
  "data": {
    "loader": "odds_api",
    "input_path": "data/games.csv"
  },
  "model": {
    "type": "all",
    "random_forest": { "n_estimators": 300, "max_depth": 8 },
    "ridge_regression": { "C": 1.0 }
  }
}
```

## Testing

```bash
pytest
```

## Linting and Type Checking

```bash
ruff check src tests
mypy src
```

## Docker

```bash
docker build -t mlbv1:latest .
docker run --env-file .env --rm mlbv1:latest
```

## Azure Deployment

The Bicep template is in [infra/main.bicep](infra/main.bicep). Deploy using:

```bash
azd up
```

Or manually:

```bash
az group create -n mlb-gbsv-v1-az-rg -l eastus
az deployment group create -g mlb-gbsv-v1-az-rg -f infra/main.bicep
```

## Project Layout

```
mlb_gbsv_v1.0_git/
  src/mlbv1/        # Core package
  tests/            # 68 tests
  scripts/          # CLI scripts (train, predict, daily_run, backtest, tune)
  infra/            # Azure Bicep templates
  artifacts/        # Trained models
```

## License

MIT
