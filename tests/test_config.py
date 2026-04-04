"""Tests for configuration loading, serialization, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

from mlbv1.config import AlertConfig, AppConfig, FeatureConfig, ModelConfig


class TestAppConfig:
    def test_default_config(self) -> None:
        cfg = AppConfig.load()
        assert cfg.features.rolling_window_short > 0
        assert cfg.features.rolling_window_long > cfg.features.rolling_window_short
        assert cfg.model.type in (
            "all",
            "both",
            "random_forest",
            "ridge_regression",
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

    def test_override_action_network_email_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("MLB_LOADER", "odds_api")
        monkeypatch.setenv("ACTION_NETWORK_EMAIL", "an@example.com")
        monkeypatch.delenv("MLB_EMAIL", raising=False)

        cfg = AppConfig.load()
        overridden = cfg.override(data={"loader": "action_network"})
        assert overridden.data.email == "an@example.com"

    def test_runtime_values_load_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TRACKING_DB_PATH", "custom/tracking.db")
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("LIVE_CONTEXT_DAYS", "45")
        monkeypatch.setenv("SLATE_TIMEZONE", "America/New_York")

        cfg = AppConfig.load()

        assert cfg.runtime.tracking_db_path == "custom/tracking.db"
        assert cfg.runtime.allow_synthetic_fallback is True
        assert cfg.runtime.live_context_days == 45
        assert cfg.runtime.slate_timezone == "America/New_York"


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
        assert mc.ridge_regression.C == 1.0


class TestAlertConfig:
    def test_supports_current_smtp_names(self, monkeypatch) -> None:
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_PORT", "2525")
        monkeypatch.setenv("SMTP_USER", "login@example.com")
        monkeypatch.setenv("SMTP_FROM", "alerts@example.com")
        monkeypatch.setenv("SMTP_TO", "ops@example.com")

        cfg = AlertConfig.from_env()

        assert cfg.smtp_host == "smtp.example.com"
        assert cfg.smtp_port == 2525
        assert cfg.smtp_login == "login@example.com"
        assert cfg.smtp_sender == "alerts@example.com"
        assert cfg.smtp_recipient == "ops@example.com"

    def test_supports_legacy_smtp_names(self, monkeypatch) -> None:
        monkeypatch.setenv("SMTP_EMAIL", "legacy@example.com")
        monkeypatch.setenv("ALERT_RECIPIENT", "recipient@example.com")
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_FROM", raising=False)
        monkeypatch.delenv("SMTP_TO", raising=False)

        cfg = AlertConfig.from_env()

        assert cfg.smtp_login == "legacy@example.com"
        assert cfg.smtp_sender == "legacy@example.com"
        assert cfg.smtp_recipient == "recipient@example.com"
