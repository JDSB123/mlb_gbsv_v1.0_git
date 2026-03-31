# Changelog

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
- Bumped Python requirement from 3.11+ → 3.12+
- F5 training fallback changed from `÷ 2` to `× 0.58` (MLB historical average)
- Widened probability clipping from [0.01, 0.99] to [0.005, 0.995]
- Pinned `setuptools<82` (v82 removed `pkg_resources`)
- Added `scipy>=1.11.0` as explicit dependency
- Regenerated `requirements-lock.txt` from clean venv

### Removed
- Dead `_generate_score()` method in synthetic history
- Duplicate `pytest.ini` (settings in `pyproject.toml`)

### Fixed
- `nhl-gbsv` cross-project contamination in lock file

## [0.2.0] — 2025-12-01

- Initial tagged release with full pipeline, Azure deployment, Docker, CI/CD
