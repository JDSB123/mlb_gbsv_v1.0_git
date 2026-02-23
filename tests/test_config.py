"""Tests for configuration loading, serialization, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

from mlbv1.config import AppConfig, FeatureConfig, ModelConfig


class TestAppConfig:
    def test_default_config(self) -> None:
        cfg = AppConfig.load()
        assert cfg.features.rolling_window_short > 0
        assert cfg.features.rolling_window_long > cfg.features.rolling_window_short
        assert cfg.model.type in (
            "all",
            "both",
            "random_forest",
            "logistic_regression",
            "xgboost",
            "lightgbm",
        )

    def test_roundtrip_dict(self) -> None:
        cfg = AppConfig.load()
        d = cfg.to_dict()
        cfg2 = AppConfig._from_dict(d)
        assert cfg2.features.rolling_window_short == cfg.features.rolling_window_short
        assert cfg2.model.type == cfg.model.type

    def test_from_json_file(self, tmp_path: Path) -> None:
        config_data = {
            "features": {
                "rolling_window_short": 7,
                "rolling_window_long": 25,
            },
            "model": {
                "type": "random_forest",
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        cfg = AppConfig.load(config_path=str(config_path))
        assert cfg.features.rolling_window_short == 7
        assert cfg.features.rolling_window_long == 25

    def test_serializes_to_json(self) -> None:
        cfg = AppConfig.load()
        d = cfg.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert "features" in parsed
        assert "model" in parsed


class TestFeatureConfig:
    def test_defaults(self) -> None:
        fc = FeatureConfig()
        assert fc.rolling_window_short == 5
        assert fc.rolling_window_long == 20


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig()
        assert mc.type == "all"
        assert mc.random_forest.n_estimators == 300
        assert mc.logistic_regression.C == 1.0
