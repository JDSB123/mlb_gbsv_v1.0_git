"""Azure Key Vault secrets provider for production deployments."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class AzureSecretsProvider:
    """Fetch secrets from Azure Key Vault using managed identity or DefaultAzureCredential."""

    def __init__(self, vault_name: str | None = None) -> None:
        """Initialize the secrets provider.

        Args:
            vault_name: Name of the Azure Key Vault (without .vault.azure.net suffix).
                       If None, reads from AZURE_KEY_VAULT_NAME environment variable.
        """
        self.vault_name = vault_name or os.getenv("AZURE_KEY_VAULT_NAME", "")
        self._client: Any = None
        self._enabled = bool(self.vault_name)

        if self._enabled:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient

                vault_url = f"https://{self.vault_name}.vault.azure.net"
                credential = DefaultAzureCredential()
                self._client = SecretClient(vault_url=vault_url, credential=credential)
                logger.info("Azure Key Vault provider initialized: %s", vault_url)
            except ImportError:
                logger.warning(
                    "Azure SDK not available. Install: pip install azure-identity azure-keyvault-secrets"
                )
                self._enabled = False
            except Exception as exc:
                logger.warning("Failed to initialize Azure Key Vault client: %s", exc)
                self._enabled = False

    def get_secret(self, secret_name: str, fallback_env_var: str | None = None) -> str:
        """Get a secret from Key Vault or fall back to environment variable.

        Args:
            secret_name: Name of the secret in Key Vault (kebab-case, e.g., 'odds-api-key').
            fallback_env_var: Environment variable to check if Key Vault is unavailable.

        Returns:
            The secret value, or empty string if not found.
        """
        # Try Key Vault first if enabled
        if self._enabled and self._client:
            try:
                secret = self._client.get_secret(secret_name)
                logger.debug("Retrieved secret '%s' from Key Vault", secret_name)
                return secret.value
            except Exception as exc:
                logger.warning(
                    "Failed to retrieve secret '%s' from Key Vault: %s",
                    secret_name,
                    exc,
                )

        # Fall back to environment variable
        if fallback_env_var:
            value = os.getenv(fallback_env_var, "")
            if value:
                logger.debug(
                    "Using environment variable '%s' for secret '%s'",
                    fallback_env_var,
                    secret_name,
                )
                return value

        logger.warning("Secret '%s' not found in Key Vault or environment", secret_name)
        return ""

    @property
    def is_enabled(self) -> bool:
        """Check if Azure Key Vault integration is active."""
        return self._enabled


# Global singleton instance
_secrets_provider: AzureSecretsProvider | None = None


def get_secrets_provider() -> AzureSecretsProvider:
    """Get or create the global secrets provider instance."""
    global _secrets_provider
    if _secrets_provider is None:
        _secrets_provider = AzureSecretsProvider()
    return _secrets_provider


def get_secret(secret_name: str, fallback_env_var: str | None = None) -> str:
    """Convenience function to get a secret from the global provider.

    Args:
        secret_name: Name of the secret in Key Vault (kebab-case).
        fallback_env_var: Environment variable to check if Key Vault is unavailable.

    Returns:
        The secret value, or empty string if not found.
    """
    provider = get_secrets_provider()
    return provider.get_secret(secret_name, fallback_env_var)
