"""HTTP server for Azure Container Apps with health checks and scheduled pipeline execution.

This server provides:
- Health endpoint for Container Apps liveness/readiness probes
- Manual trigger endpoint for daily pipeline
- Graceful shutdown handling with SIGTERM
- OpenTelemetry integration for observability
"""

from __future__ import annotations

import hmac
import logging
import os
import signal
import sys
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from mlbv1.observability import configure_telemetry
from mlbv1.tracking.database import TrackingDB

# Add project root to Python path for importing scripts
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure telemetry BEFORE any other imports
configure_telemetry()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
_shutdown_event = threading.Event()
_pipeline_lock = threading.Lock()
_last_trigger_monotonic = 0.0
_tracking_db = TrackingDB(os.getenv("TRACKING_DB_PATH", "artifacts/tracking.db"))


def _is_trigger_authorized() -> tuple[bool, str | None]:
    """Validate trigger request auth based on environment configuration."""
    expected_key = os.getenv("TRIGGER_API_KEY", "").strip()
    allow_unauth = os.getenv("ALLOW_UNAUTH_TRIGGER", "false").lower() == "true"

    if not expected_key:
        if allow_unauth:
            return True, None
        return False, "Trigger authentication is not configured"

    provided_key = request.headers.get("X-Trigger-Key", "").strip()
    if not provided_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided_key = auth_header[7:].strip()

    if hmac.compare_digest(provided_key, expected_key):
        return True, None

    return False, "Invalid trigger key"


@app.route("/health", methods=["GET"])
def health() -> ResponseReturnValue:
    """Health check endpoint for Container Apps probes."""
    pipeline = _tracking_db.get_pipeline_status()

    status: dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "last_run": pipeline.get("last_run"),
        "last_run_status": pipeline.get("last_run_status", "never_run"),
    }
    return jsonify(status), 200


@app.route("/", methods=["GET"])
def root() -> ResponseReturnValue:
    """Root endpoint with service info."""
    info: dict[str, Any] = {
        "service": "MLB Prediction Model v1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "trigger": "/trigger (POST)",
        },
    }
    return jsonify(info), 200


@app.route("/trigger", methods=["POST"])
def trigger_pipeline() -> ResponseReturnValue:
    """Manual trigger for the daily pipeline (for testing/manual runs)."""
    global _last_trigger_monotonic

    logger.info("Pipeline triggered via HTTP POST")

    authorized, auth_error = _is_trigger_authorized()
    if not authorized:
        status = 503 if auth_error and "not configured" in auth_error.lower() else 401
        logger.warning("Rejected pipeline trigger: %s", auth_error)
        return jsonify({"status": "error", "message": auth_error}), status

    min_interval = float(os.getenv("TRIGGER_MIN_INTERVAL_SECONDS", "30"))
    now_mono = time.monotonic()

    with _pipeline_lock:
        if (now_mono - _last_trigger_monotonic) < min_interval:
            wait_for = int(min_interval - (now_mono - _last_trigger_monotonic))
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Trigger rate limited. Retry in {wait_for}s",
                    }
                ),
                429,
            )

        _last_trigger_monotonic = now_mono

    if not _tracking_db.try_start_pipeline():
        return (
            jsonify({"status": "error", "message": "Pipeline is already running"}),
            409,
        )

    try:
        # Import daily_run from scripts directory
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "daily_run", PROJECT_ROOT / "scripts" / "daily_run.py"
        )
        if spec and spec.loader:
            daily_run_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(daily_run_module)
            run_pipeline = daily_run_module.main
        else:
            raise ImportError("Could not load daily_run module")

        # Run in background thread to avoid blocking the health endpoint
        def _run():
            try:
                old_argv = sys.argv[:]
                sys.argv = [str(PROJECT_ROOT / "scripts" / "daily_run.py")]
                run_pipeline()
                _tracking_db.finish_pipeline_run("success")
            except BaseException as exc:
                logger.exception("Pipeline failed: %s", exc)
                _tracking_db.finish_pipeline_run("error")
            finally:
                with suppress(Exception):
                    sys.argv = old_argv

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return (
            jsonify(
                {"status": "triggered", "message": "Pipeline started in background"}
            ),
            202,
        )

    except Exception as exc:
        _tracking_db.finish_pipeline_run("error")
        logger.error("Failed to trigger pipeline: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


def _shutdown_handler(signum: int, frame: Any) -> None:  # noqa: ANN401
    """Handle graceful shutdown on SIGTERM."""
    logger.info("Received shutdown signal %d, initiating graceful shutdown...", signum)
    _shutdown_event.set()
    sys.exit(0)


def main() -> None:
    """Start the Flask server with graceful shutdown support."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    logger.info("Starting MLB prediction server on port 8000")
    logger.info("Health endpoint: http://localhost:8000/health")

    # Check configuration
    appinsights_conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if appinsights_conn:
        logger.info("Application Insights configured ✓")
    else:
        logger.info("Application Insights not configured (local development mode)")

    kv_name = os.getenv("AZURE_KEY_VAULT_NAME", "")
    if kv_name:
        logger.info("Azure Key Vault configured: %s", kv_name)
    else:
        logger.info("Azure Key Vault not configured (using environment variables)")

    # Start Flask server
    # In production, use a proper WSGI server like gunicorn
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
