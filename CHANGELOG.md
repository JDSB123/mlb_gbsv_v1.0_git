# Changelog

## [1.0.1] — 2026-04-04

### Changed

- Migrated from Python 3.12 to Python 3.14
- Relaxed dependency pins from exact (`==`) to minimum floors (`>=`) for 3.14 compatibility
- Upgraded key dependencies: numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, lightgbm 4.6.0, scipy 1.17.1
- Updated Dockerfile, devcontainer, GitHub Actions, and CI/CD to Python 3.14
- Removed hardcoded personal paths from PS1 scripts; fallback now uses `artifacts/`
- Fixed `v2` resource name defaults in `deploy.ps1` and `setup_teams_webhook.ps1` to `v1`
- Removed stale `setuptools<82` constraint (pkg_resources not used)
- Auto-fixed ruff UP017/UP043 lint findings for 3.14 target

## [1.0.0] — 2026-03-31

### Added

- F5 (first-five-innings) linescore extraction from MLB Stats API
- F5 market settlement in tracking database
- Devigged moneyline implied probability feature
- Weighted voting support in ensemble predictions
- Per-inning Poisson scoring in synthetic history generator
- Feature engineering defaults logging
- Weather enrichment via Visual Crossing API
- `.env` / `.env.example` infrastructure for API keys
- BetsAPI loader support
- CHANGELOG.md

### Changed

- **BREAKING**: Renamed `LogisticRegressionConfig` → `RidgeRegressionConfig` (backward alias retained)
- **BREAKING**: Renamed model config field `logistic_regression` → `ridge_regression` (old JSON key still accepted)
- **BREAKING**: Renamed `train_logistic_regression()` → `train_ridge_regression()` (backward alias retained)
- Bumped version from 0.2.0 → 1.0.0
- Bumped Python requirement from 3.11+ → 3.12+ (superseded by 1.0.1 → 3.14)
- F5 training fallback changed from `÷ 2` to `× 0.58` (MLB historical average)
- Widened probability clipping from [0.01, 0.99] to [0.005, 0.995]
- Added `scipy>=1.11.0` as explicit dependency
- Regenerated `requirements-lock.txt` from clean venv

### Removed

- Dead `_generate_score()` method in synthetic history
- Duplicate `pytest.ini` (settings in `pyproject.toml`)

### Fixed

- `nhl-gbsv` cross-project contamination in lock file

## [0.2.0] — 2025-12-01

- Initial tagged release with full pipeline, Azure deployment, Docker, CI/CD
