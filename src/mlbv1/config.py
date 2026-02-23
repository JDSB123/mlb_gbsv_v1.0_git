"""Configuration management for MLBV1."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv(dotenv_path: str | None = None) -> None:
    """Read a .env file and inject into os.environ (no dependency needed)."""
    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


# Auto-load .env on import so every module sees the vars.
_load_dotenv()


# Map loader names to their env-var key names.
_LOADER_API_KEY_MAP: dict[str, str] = {
    "odds_api": "ODDS_API_KEY",
    "action_network": "ACTION_NETWORK_PASSWORD",
    "bets_api": "BETS_API_KEY",
}

_LOADER_BASE_URL_MAP: dict[str, str] = {
    "odds_api": "https://api.the-odds-api.com/v4",
    "action_network": "https://api.actionnetwork.com/v1",
    "bets_api": "https://api.betsapi.com",
}


@dataclass(frozen=True)
class RandomForestConfig:
    """RandomForest model parameters."""

    n_estimators: int = 300
    max_depth: int | None = 8
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
class XGBoostConfig:
    """XGBoost model parameters."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.01
    reg_lambda: float = 1.0
    random_state: int = 42
    use_label_encoder: bool = False
    eval_metric: str = "logloss"


@dataclass(frozen=True)
class LightGBMConfig:
    """LightGBM model parameters."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.01
    reg_lambda: float = 1.0
    num_leaves: int = 31
    random_state: int = 42
    verbose: int = -1


@dataclass(frozen=True)
class TuningConfig:
    """Hyperparameter tuning settings."""

    enabled: bool = False
    cv_folds: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1


@dataclass(frozen=True)
class AlertConfig:
    """Alert / notification settings (populated from env vars)."""

    discord_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""

    @classmethod
    def from_env(cls) -> AlertConfig:
        return cls(
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_from=os.getenv("SMTP_FROM", ""),
            smtp_to=os.getenv("SMTP_TO", ""),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    type: str = "all"
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


@dataclass(frozen=True)
class DataConfig:
    """Data ingestion configuration."""

    loader: str = "synthetic"
    input_path: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None


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
    alerts: AlertConfig = field(default_factory=AlertConfig)

    @staticmethod
    def _from_dict(payload: dict[str, Any]) -> AppConfig:
        data_cfg = DataConfig(**payload.get("data", {}))
        rf_cfg = RandomForestConfig(**payload.get("model", {}).get("random_forest", {}))
        lr_cfg = LogisticRegressionConfig(
            **payload.get("model", {}).get("logistic_regression", {})
        )
        xgb_raw = payload.get("model", {}).get("xgboost", {})
        xgb_cfg = XGBoostConfig(**xgb_raw) if xgb_raw else XGBoostConfig()
        lgbm_raw = payload.get("model", {}).get("lightgbm", {})
        lgbm_cfg = LightGBMConfig(**lgbm_raw) if lgbm_raw else LightGBMConfig()
        tuning_raw = payload.get("model", {}).get("tuning", {})
        tuning_cfg = TuningConfig(**tuning_raw) if tuning_raw else TuningConfig()
        model_cfg = ModelConfig(
            type=payload.get("model", {}).get("type", "all"),
            random_forest=rf_cfg,
            logistic_regression=lr_cfg,
            xgboost=xgb_cfg,
            lightgbm=lgbm_cfg,
            tuning=tuning_cfg,
        )
        feature_cfg = FeatureConfig(**payload.get("features", {}))
        alert_cfg = AlertConfig.from_env()
        return AppConfig(
            data=data_cfg, model=model_cfg, features=feature_cfg, alerts=alert_cfg
        )

    def to_dict(self) -> dict[str, Any]:
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
                "xgboost": self.model.xgboost.__dict__,
                "lightgbm": self.model.lightgbm.__dict__,
                "tuning": self.model.tuning.__dict__,
            },
            "features": {
                "rolling_window_short": self.features.rolling_window_short,
                "rolling_window_long": self.features.rolling_window_long,
            },
        }

    def override(
        self, data: dict[str, Any] | None = None, model: dict[str, Any] | None = None
    ) -> AppConfig:
        """Return a new config with overrides applied."""
        payload = self.to_dict()
        if data:
            payload["data"].update(data)
        if model:
            payload["model"].update(model)
        return AppConfig._from_dict(payload)

    @classmethod
    def load(cls, config_path: str | None = None) -> AppConfig:
        """Load configuration from JSON file or environment variables."""
        if config_path:
            with open(config_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return cls._from_dict(payload)

        loader = os.getenv("MLB_LOADER", "synthetic")
        # Resolve the correct API key for the selected loader.
        api_key = os.getenv("MLB_API_KEY") or os.getenv(
            _LOADER_API_KEY_MAP.get(loader, ""), ""
        )
        api_base_url = os.getenv("MLB_API_BASE_URL") or _LOADER_BASE_URL_MAP.get(
            loader, ""
        )

        env_payload: dict[str, Any] = {
            "data": {
                "loader": loader,
                "input_path": os.getenv("MLB_INPUT_PATH"),
                "api_key": api_key,
                "api_base_url": api_base_url,
            },
            "model": {
                "type": os.getenv("MLB_MODEL_TYPE", "all"),
            },
        }
        return cls._from_dict(env_payload)
