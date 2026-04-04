"""Configuration management for MLBV1."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from mlbv1.azure_secrets import get_secret
from mlbv1.environment import bootstrap_environment, env_bool, env_float, env_int

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


def _env_secret(secret_name: str, env_name: str) -> str:
    return get_secret(secret_name, env_name)


def _redact(value: str | None) -> str:
    return "***REDACTED***" if value else ""


@dataclass(frozen=True)
class RandomForestConfig:
    """RandomForest model parameters."""

    n_estimators: int = 300
    max_depth: int | None = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42


@dataclass(frozen=True)
class RidgeRegressionConfig:
    """Ridge regression model parameters."""

    C: float = 1.0  # converted to alpha = 1/C in trainer
    max_iter: int = 2000
    random_state: int = 42


# Backward-compatible alias
LogisticRegressionConfig = RidgeRegressionConfig


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
    eval_metric: str = "rmse"


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
    """Alert / notification settings."""

    discord_webhook_url: str = ""
    teams_webhook_url: str = ""
    teams_group_id: str = ""
    teams_channel_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""

    @property
    def smtp_login(self) -> str:
        return self.smtp_user or self.smtp_from

    @property
    def smtp_sender(self) -> str:
        return self.smtp_from or self.smtp_user

    @property
    def smtp_recipient(self) -> str:
        return self.smtp_to or self.smtp_sender

    @classmethod
    def from_env(cls) -> AlertConfig:
        smtp_user = os.getenv("SMTP_USER", "") or os.getenv("SMTP_EMAIL", "")
        smtp_from = os.getenv("SMTP_FROM", "") or smtp_user
        smtp_to = os.getenv("SMTP_TO", "") or os.getenv("ALERT_RECIPIENT", "")
        return cls(
            discord_webhook_url=_env_secret(
                "discord-webhook-url", "DISCORD_WEBHOOK_URL"
            ),
            teams_webhook_url=_env_secret("teams-webhook-url", "TEAMS_WEBHOOK_URL"),
            teams_group_id=os.getenv("TEAMS_GROUP_ID", ""),
            teams_channel_id=os.getenv("TEAMS_CHANNEL_ID", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=env_int("SMTP_PORT", 587),
            smtp_user=smtp_user,
            smtp_password=_env_secret("smtp-password", "SMTP_PASSWORD"),
            smtp_from=smtp_from,
            smtp_to=smtp_to,
        )


@dataclass(frozen=True)
class ServicesConfig:
    """External service credentials and secrets."""

    odds_api_key: str = ""
    bets_api_key: str = ""
    action_network_email: str = ""
    action_network_password: str = ""
    visual_crossing_api_key: str = ""

    @classmethod
    def from_env(cls) -> ServicesConfig:
        return cls(
            odds_api_key=_env_secret("odds-api-key", "ODDS_API_KEY"),
            bets_api_key=_env_secret("bets-api-key", "BETS_API_KEY"),
            action_network_email=_env_secret(
                "action-network-email", "ACTION_NETWORK_EMAIL"
            ),
            action_network_password=_env_secret(
                "action-network-password", "ACTION_NETWORK_PASSWORD"
            ),
            visual_crossing_api_key=_env_secret(
                "visual-crossing-api-key", "VISUAL_CROSSING_API_KEY"
            ),
        )

    def loader_api_key(self, loader: str) -> str:
        return {
            "odds_api": self.odds_api_key,
            "bets_api": self.bets_api_key,
            "action_network": self.action_network_password,
        }.get(loader, "")

    def loader_email(self, loader: str) -> str:
        if loader == "action_network":
            return self.action_network_email
        return ""


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings shared across local, CI, and ACA."""

    environment: str = "local"
    tracking_db_path: str = "artifacts/tracking.db"
    allow_synthetic_fallback: bool = False
    slate_timezone: str = "America/Chicago"
    live_context_days: int = 120
    trigger_api_key: str = ""
    allow_unauth_trigger: bool = False
    trigger_min_interval_seconds: float = 30.0
    applicationinsights_connection_string: str = ""
    azure_key_vault_name: str = ""

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        state = bootstrap_environment()
        return cls(
            environment=state.environment,
            tracking_db_path=os.getenv("TRACKING_DB_PATH", "artifacts/tracking.db"),
            allow_synthetic_fallback=env_bool("ALLOW_SYNTHETIC_FALLBACK", False),
            slate_timezone=os.getenv("SLATE_TIMEZONE", "America/Chicago"),
            live_context_days=env_int("LIVE_CONTEXT_DAYS", 120),
            trigger_api_key=_env_secret("trigger-api-key", "TRIGGER_API_KEY"),
            allow_unauth_trigger=env_bool("ALLOW_UNAUTH_TRIGGER", False),
            trigger_min_interval_seconds=env_float(
                "TRIGGER_MIN_INTERVAL_SECONDS", 30.0
            ),
            applicationinsights_connection_string=os.getenv(
                "APPLICATIONINSIGHTS_CONNECTION_STRING", ""
            ),
            azure_key_vault_name=os.getenv("AZURE_KEY_VAULT_NAME", ""),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    type: str = "all"
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    ridge_regression: RidgeRegressionConfig = field(
        default_factory=RidgeRegressionConfig
    )
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    @property
    def logistic_regression(self) -> RidgeRegressionConfig:
        """Backward-compatible alias."""
        return self.ridge_regression


@dataclass(frozen=True)
class DataConfig:
    """Data ingestion configuration."""

    loader: str = "synthetic"
    input_path: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    email: str | None = None


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
    services: ServicesConfig = field(default_factory=ServicesConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @staticmethod
    def _from_dict(payload: dict[str, Any]) -> AppConfig:
        data_cfg = DataConfig(**payload.get("data", {}))
        rf_cfg = RandomForestConfig(**payload.get("model", {}).get("random_forest", {}))
        rr_raw = payload.get("model", {}).get(
            "ridge_regression", payload.get("model", {}).get("logistic_regression", {})
        )
        rr_cfg = RidgeRegressionConfig(**rr_raw) if rr_raw else RidgeRegressionConfig()
        xgb_raw = payload.get("model", {}).get("xgboost", {})
        xgb_cfg = XGBoostConfig(**xgb_raw) if xgb_raw else XGBoostConfig()
        lgbm_raw = payload.get("model", {}).get("lightgbm", {})
        lgbm_cfg = LightGBMConfig(**lgbm_raw) if lgbm_raw else LightGBMConfig()
        tuning_raw = payload.get("model", {}).get("tuning", {})
        tuning_cfg = TuningConfig(**tuning_raw) if tuning_raw else TuningConfig()
        model_cfg = ModelConfig(
            type=payload.get("model", {}).get("type", "all"),
            random_forest=rf_cfg,
            ridge_regression=rr_cfg,
            xgboost=xgb_cfg,
            lightgbm=lgbm_cfg,
            tuning=tuning_cfg,
        )
        feature_cfg = FeatureConfig(**payload.get("features", {}))
        alert_cfg = AlertConfig.from_env()
        services_cfg = ServicesConfig.from_env()
        runtime_defaults = RuntimeConfig.from_env()
        runtime_raw = payload.get("runtime", {})
        runtime_cfg = RuntimeConfig(
            environment=runtime_raw.get("environment", runtime_defaults.environment),
            tracking_db_path=runtime_raw.get(
                "tracking_db_path", runtime_defaults.tracking_db_path
            ),
            allow_synthetic_fallback=runtime_raw.get(
                "allow_synthetic_fallback",
                runtime_defaults.allow_synthetic_fallback,
            ),
            slate_timezone=runtime_raw.get(
                "slate_timezone", runtime_defaults.slate_timezone
            ),
            live_context_days=runtime_raw.get(
                "live_context_days", runtime_defaults.live_context_days
            ),
            trigger_api_key=runtime_raw.get(
                "trigger_api_key", runtime_defaults.trigger_api_key
            ),
            allow_unauth_trigger=runtime_raw.get(
                "allow_unauth_trigger", runtime_defaults.allow_unauth_trigger
            ),
            trigger_min_interval_seconds=runtime_raw.get(
                "trigger_min_interval_seconds",
                runtime_defaults.trigger_min_interval_seconds,
            ),
            applicationinsights_connection_string=runtime_raw.get(
                "applicationinsights_connection_string",
                runtime_defaults.applicationinsights_connection_string,
            ),
            azure_key_vault_name=runtime_raw.get(
                "azure_key_vault_name", runtime_defaults.azure_key_vault_name
            ),
        )
        return AppConfig(
            data=data_cfg,
            model=model_cfg,
            features=feature_cfg,
            alerts=alert_cfg,
            services=services_cfg,
            runtime=runtime_cfg,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "data": {
                "loader": self.data.loader,
                "input_path": self.data.input_path,
                "api_key": self.data.api_key,
                "api_base_url": self.data.api_base_url,
                "email": self.data.email,
            },
            "model": {
                "type": self.model.type,
                "random_forest": self.model.random_forest.__dict__,
                "ridge_regression": self.model.ridge_regression.__dict__,
                "xgboost": self.model.xgboost.__dict__,
                "lightgbm": self.model.lightgbm.__dict__,
                "tuning": self.model.tuning.__dict__,
            },
            "features": {
                "rolling_window_short": self.features.rolling_window_short,
                "rolling_window_long": self.features.rolling_window_long,
            },
            "alerts": {
                "teams_group_id": self.alerts.teams_group_id,
                "teams_channel_id": self.alerts.teams_channel_id,
                "smtp_host": self.alerts.smtp_host,
                "smtp_port": self.alerts.smtp_port,
                "smtp_user": self.alerts.smtp_user,
                "smtp_from": self.alerts.smtp_from,
                "smtp_to": self.alerts.smtp_to,
            },
            "services": {
                "odds_api_key": self.services.odds_api_key,
                "bets_api_key": self.services.bets_api_key,
                "action_network_email": self.services.action_network_email,
                "action_network_password": self.services.action_network_password,
                "visual_crossing_api_key": self.services.visual_crossing_api_key,
            },
            "runtime": self.runtime.__dict__,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialize config with secrets redacted for safe storage/logging."""
        payload = self.to_dict()
        payload["data"]["api_key"] = _redact(payload["data"].get("api_key"))
        payload["data"]["email"] = _redact(payload["data"].get("email"))
        payload["alerts"]["discord_webhook_url"] = _redact(
            self.alerts.discord_webhook_url
        )
        payload["alerts"]["teams_webhook_url"] = _redact(self.alerts.teams_webhook_url)
        payload["alerts"]["smtp_password"] = _redact(self.alerts.smtp_password)
        payload["services"] = {
            "odds_api_key": _redact(self.services.odds_api_key),
            "bets_api_key": _redact(self.services.bets_api_key),
            "action_network_email": _redact(self.services.action_network_email),
            "action_network_password": _redact(
                self.services.action_network_password
            ),
            "visual_crossing_api_key": _redact(
                self.services.visual_crossing_api_key
            ),
        }
        payload["runtime"]["trigger_api_key"] = _redact(self.runtime.trigger_api_key)
        payload["runtime"]["applicationinsights_connection_string"] = _redact(
            self.runtime.applicationinsights_connection_string
        )
        return payload

    def override(
        self, data: dict[str, Any] | None = None, model: dict[str, Any] | None = None
    ) -> AppConfig:
        """Return a new config with overrides applied."""
        payload = self.to_dict()
        if data:
            payload["data"].update(data)
            loader = payload["data"].get("loader")
            if loader:
                if "api_key" not in data:
                    payload["data"]["api_key"] = self.services.loader_api_key(
                        str(loader)
                    ) or os.getenv("MLB_API_KEY", "")
                if "api_base_url" not in data:
                    payload["data"]["api_base_url"] = os.getenv(
                        "MLB_API_BASE_URL",
                        _LOADER_BASE_URL_MAP.get(str(loader), ""),
                    )
                if "email" not in data:
                    payload["data"]["email"] = os.getenv("MLB_EMAIL") or (
                        self.services.loader_email(str(loader))
                    )
        if model:
            payload["model"].update(model)
        return AppConfig._from_dict(payload)

    @classmethod
    def load(cls, config_path: str | None = None) -> AppConfig:
        """Load configuration from JSON file or environment variables."""
        bootstrap_environment()

        if config_path:
            with open(config_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return cls._from_dict(payload)

        services = ServicesConfig.from_env()
        runtime = RuntimeConfig.from_env()
        loader = os.getenv("MLB_LOADER", "synthetic")
        api_key = services.loader_api_key(loader) or os.getenv("MLB_API_KEY", "")
        api_base_url = os.getenv("MLB_API_BASE_URL") or _LOADER_BASE_URL_MAP.get(
            loader, ""
        )
        email = os.getenv("MLB_EMAIL") or services.loader_email(loader)

        env_payload: dict[str, Any] = {
            "data": {
                "loader": loader,
                "input_path": os.getenv("MLB_INPUT_PATH"),
                "api_key": api_key,
                "api_base_url": api_base_url,
                "email": email,
            },
            "model": {
                "type": os.getenv("MLB_MODEL_TYPE", "all"),
            },
            "runtime": runtime.__dict__,
        }
        return cls._from_dict(env_payload)
