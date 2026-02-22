# MLBV1 - MLB Spread Prediction

MLBV1 is a production-ready pipeline for MLB spread prediction using modular data ingestion, feature engineering, and scikit-learn models. It includes a full testing suite, Dockerization, CI/CD, and Azure infrastructure templates.

## Features
- Pluggable data loaders (Odds API, Action Network, BetsAPI, CSV/JSON, synthetic)
- Feature engineering (team stats, trends, rest days, weather)
- Dual model training (RandomForest and LogisticRegression)
- Evaluation metrics (accuracy, ROI, Sharpe ratio)
- Docker + GitHub Actions CI
- Azure deployment via Bicep

## Requirements
- Python 3.11+

## Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Train

```bash
python scripts/train.py --loader synthetic --model both
```

### Predict

```bash
python scripts/predict.py --model-path artifacts/models/random_forest.pkl --loader synthetic
```

## Configuration
Configuration can be supplied via JSON file or environment variables.

```bash
python scripts/train.py --config config.json
```

Example JSON:

```json
{
  "data": {
    "loader": "synthetic",
    "input_path": "data/games.csv"
  },
  "model": {
    "type": "both",
    "random_forest": {"n_estimators": 300, "max_depth": 8},
    "logistic_regression": {"C": 1.0}
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
docker run --rm mlbv1:latest
```

## Azure Deployment
The Bicep template is in [infra/main.bicep](infra/main.bicep). Deploy using:

```bash
az group create -n mlb-gbsv-v1-az-rg -l eastus
az deployment group create -g mlb-gbsv-v1-az-rg -f infra/main.bicep
```

## Project Layout

```
mlb_gbsv_local_v1.0/
  src/mlbv1/
  tests/
  scripts/
  infra/
```

## License
MIT
