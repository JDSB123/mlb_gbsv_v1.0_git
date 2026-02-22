"""Configuration management for MLBV1."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RandomForestConfig:
    """RandomForest model parameters."""

    n_estimators: int = 300
    max_depth: Optional[int] = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """LogisticRegression model parameters."""

    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    type: str = "both"
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    logistic_regression: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)


@dataclass(frozen=True)
class DataConfig:
    """Data ingestion configuration."""

    loader: str = "synthetic"
    input_path: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    rolling_window_short: int = 5
    rolling_window_long: int = 20


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @staticmethod
    def _from_dict(payload: Dict[str, Any]) -> "AppConfig":
        data_cfg = DataConfig(**payload.get("data", {}))
        rf_cfg = RandomForestConfig(**payload.get("model", {}).get("random_forest", {}))
        lr_cfg = LogisticRegressionConfig(**payload.get("model", {}).get("logistic_regression", {}))
        model_cfg = ModelConfig(
            type=payload.get("model", {}).get("type", "both"),
            random_forest=rf_cfg,
            logistic_regression=lr_cfg,
        )
        feature_cfg = FeatureConfig(**payload.get("features", {}))
        return AppConfig(data=data_cfg, model=model_cfg, features=feature_cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "data": {
                "loader": self.data.loader,
                "input_path": self.data.input_path,
                "api_key": self.data.api_key,
                "api_base_url": self.data.api_base_url,
            },
            "model": {
                "type": self.model.type,
                "random_forest": self.model.random_forest.__dict__,
                "logistic_regression": self.model.logistic_regression.__dict__,
            },
            "features": {
                "rolling_window_short": self.features.rolling_window_short,
                "rolling_window_long": self.features.rolling_window_long,
            },
        }

    def override(
        self, data: Optional[Dict[str, Any]] = None, model: Optional[Dict[str, Any]] = None
    ) -> "AppConfig":
        """Return a new config with overrides applied."""
        payload = self.to_dict()
        if data:
            payload["data"].update(data)
        if model:
            payload["model"].update(model)
        return AppConfig._from_dict(payload)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load configuration from JSON file or environment variables."""
        if config_path:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return cls._from_dict(payload)

        env_payload: Dict[str, Any] = {
            "data": {
                "loader": os.getenv("MLB_LOADER", "synthetic"),
                "input_path": os.getenv("MLB_INPUT_PATH"),
                "api_key": os.getenv("MLB_API_KEY"),
                "api_base_url": os.getenv("MLB_API_BASE_URL"),
            },
            "model": {
                "type": os.getenv("MLB_MODEL_TYPE", "both"),
            },
        }
        return cls._from_dict(env_payload)
