"""OpenTelemetry and Azure Monitor observability setup."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def configure_telemetry() -> None:
    """Configure OpenTelemetry with Azure Monitor for production observability.

    This should be called once at application startup before any other code runs.
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")

    if not connection_string:
        logger.info(
            "Application Insights not configured (APPLICATIONINSIGHTS_CONNECTION_STRING not set). "
            "Telemetry disabled for local development."
        )
        return

    try:
        from azure.monitor.opentelemetry import (
                configure_azure_monitor,
        )
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider

        # Configure Azure Monitor with automatic instrumentation
        configure_azure_monitor(
            connection_string=connection_string,
            enable_live_metrics=True,
            logger_name="mlbv1",
        )

        # Get tracer and meter for custom instrumentation
        tracer_provider = trace.get_tracer_provider()
        meter_provider = metrics.get_meter_provider()

        if isinstance(tracer_provider, TracerProvider):
            logger.info("OpenTelemetry tracing configured with Azure Monitor")

        if isinstance(meter_provider, MeterProvider):
            logger.info("OpenTelemetry metrics configured with Azure Monitor")

        # Store global instances for use throughout the application
        _configure_custom_metrics()

    except ImportError:
        logger.warning(
            "Azure Monitor OpenTelemetry SDK not available. "
            "Install: pip install azure-monitor-opentelemetry"
        )
    except Exception as exc:
        logger.warning("Failed to configure Application Insights: %s", exc)


def _configure_custom_metrics() -> None:
    """Set up custom metrics for the MLB prediction application."""
    try:
        from opentelemetry import metrics

        meter = metrics.get_meter(__name__)

        # Create custom metrics
        global _prediction_counter
        global _accuracy_gauge
        global _api_call_counter
        global _model_load_duration

        _prediction_counter = meter.create_counter(
            name="mlb.predictions.count",
            description="Total number of predictions made",
            unit="1",
        )

        _accuracy_gauge = meter.create_up_down_counter(
            name="mlb.model.accuracy",
            description="Model accuracy metric",
            unit="1",
        )

        _api_call_counter = meter.create_counter(
            name="mlb.api.calls",
            description="External API calls made by data loaders",
            unit="1",
        )

        _model_load_duration = meter.create_histogram(
            name="mlb.model.load.duration",
            description="Time taken to load model from disk",
            unit="ms",
        )

        logger.info("Custom metrics configured")

    except Exception as exc:
        logger.warning("Failed to configure custom metrics: %s", exc)


# Global metric instances (set by _configure_custom_metrics)
_prediction_counter: Any = None
_accuracy_gauge: Any = None
_api_call_counter: Any = None
_model_load_duration: Any = None


def track_prediction(model_name: str, market: str, count: int = 1) -> None:
    """Record a prediction event.

    Args:
        model_name: Name of the model used (e.g., 'random_forest', 'ensemble')
        market: Market type (e.g., 'spread', 'moneyline')
        count: Number of predictions (default 1)
    """
    if _prediction_counter:
        _prediction_counter.add(count, {"model": model_name, "market": market})


def track_accuracy(model_name: str, accuracy: float) -> None:
    """Record model accuracy metric.

    Args:
        model_name: Name of the model
        accuracy: Accuracy value between 0.0 and 1.0
    """
    if _accuracy_gauge:
        _accuracy_gauge.add(int(accuracy * 100), {"model": model_name})


def track_api_call(loader_name: str, status: str) -> None:
    """Record an external API call.

    Args:
        loader_name: Name of the data loader (e.g., 'odds_api', 'bets_api')
        status: Status of the call ('success', 'error', 'timeout')
    """
    if _api_call_counter:
        _api_call_counter.add(1, {"loader": loader_name, "status": status})


def track_model_load_time(model_name: str, duration_ms: float) -> None:
    """Record model loading time.

    Args:
        model_name: Name of the model
        duration_ms: Time taken to load in milliseconds
    """
    if _model_load_duration:
        _model_load_duration.record(duration_ms, {"model": model_name})


def get_tracer(name: str) -> Any:
    """Get an OpenTelemetry tracer for custom spans.

    Args:
        name: Name for the tracer (typically module name)

    Returns:
        Tracer instance or no-op tracer if not configured
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        logger.debug("OpenTelemetry not available, returning no-op tracer")
        return _NoOpTracer()


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not configured."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """No-op context manager."""
        import contextlib

        return contextlib.nullcontext()
