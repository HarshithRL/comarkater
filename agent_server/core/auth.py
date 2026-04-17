"""Multi-client Service Principal authentication.

Three responsibilities:
1. SecretsLoader — reads SP credentials from Databricks secrets scope at runtime
2. resolve_client() — maps SP ID to client info + resolved SP credentials
3. SecureTokenProvider — OAuth2 client_credentials flow with token caching

Secrets are read via WorkspaceClient().secrets.get_secret() + base64 decode,
matching the pattern from the legacy model serving agent.
"""

import base64
import logging
import time
from typing import Optional

import httpx
from mlflow.types.responses import ResponsesAgentRequest
from core.config import settings, CLIENT_REGISTRY

logger = logging.getLogger(__name__)


class SecretsLoader:
    """Reads SP credentials from Databricks secrets scope. Caches after first read."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._client = None
        self._init_failed = False

    def _get_client(self):
        """Lazy-init WorkspaceClient."""
        if self._client is not None:
            return self._client
        if self._init_failed:
            return None
        try:
            from databricks.sdk import WorkspaceClient
            self._client = WorkspaceClient()
            logger.info("SECRETS: WorkspaceClient initialized")
            return self._client
        except Exception as e:
            logger.warning(f"SECRETS: Failed to init WorkspaceClient: {e}")
            self._init_failed = True
            return None

    def get_secret(self, key: str) -> str:
        """Read a secret from the configured scope. Cached after first read.

        Returns:
            Secret value string, or empty string on failure.
        """
        if key in self._cache:
            return self._cache[key]

        client = self._get_client()
        if not client:
            return ""

        try:
            resp = client.secrets.get_secret(scope=settings.SECRETS_SCOPE, key=key)
            value = base64.b64decode(resp.value).decode("utf-8")
            self._cache[key] = value
            logger.info(f"SECRETS: Loaded key={key} from scope={settings.SECRETS_SCOPE}")
            return value
        except Exception as e:
            logger.warning(f"SECRETS: Failed to read key={key}: {e}")
            return ""


# Singleton — initialized once at module load
secrets_loader = SecretsLoader()


def resolve_client(request: ResponsesAgentRequest) -> dict:
    """Resolve client info from the request.

    Reads sp_id from custom_inputs, looks up CLIENT_REGISTRY.
    Resolves per-client SP credentials from Databricks secrets scope.

    Returns:
        dict with keys: client_id, client_name, display_name, sp_id,
                        sp_client_id, sp_client_secret
    """
    custom = request.custom_inputs or {}
    sp_id = custom.get("sp_id", "default")

    client_info = CLIENT_REGISTRY.get(sp_id, CLIENT_REGISTRY["default"]).copy()
    client_info["sp_id"] = sp_id

    # Read per-client SP credentials from secrets scope
    cid_key = client_info.pop("secret_client_id_key", "default-sp-client-id")
    csecret_key = client_info.pop("secret_client_secret_key", "default-sp-client-secret")

    client_info["sp_client_id"] = secrets_loader.get_secret(cid_key)
    client_info["sp_client_secret"] = secrets_loader.get_secret(csecret_key)

    # Fall back to default SP creds if client-specific ones are empty
    if not client_info["sp_client_id"]:
        client_info["sp_client_id"] = secrets_loader.get_secret("default-sp-client-id")
    if not client_info["sp_client_secret"]:
        client_info["sp_client_secret"] = secrets_loader.get_secret("default-sp-client-secret")

    logger.info(
        f"AUTH.resolve_client: sp_id={sp_id} | client_name={client_info['client_name']} "
        f"| has_sp_creds={bool(client_info['sp_client_id'])}"
    )
    return client_info


class SecureTokenProvider:
    """SP OAuth2 client_credentials flow with token caching.

    Caches tokens keyed by client_id prefix.
    Refreshes automatically 120s before expiry.
    """

    def __init__(self) -> None:
        self._token_cache: dict[str, dict] = {}
        self._buffer_seconds: int = 120

    def get_token(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> str:
        """Get OAuth token via client_credentials flow.

        Uses provided creds. Caches tokens. Refreshes 120s before expiry.

        Returns:
            Bearer token string.

        Raises:
            ValueError: If no credentials available.
            RuntimeError: If token request fails.
        """
        cid = client_id or ""
        csecret = client_secret or ""

        if not cid or not csecret:
            raise ValueError("AUTH.get_token: No SP credentials available")

        cache_key = cid[:8]

        # Check cache
        cached = self._token_cache.get(cache_key)
        if cached and cached["expires_at"] > time.time() + self._buffer_seconds:
            logger.info(f"AUTH.get_token: Cache hit | key={cache_key}")
            return cached["token"]

        # Request new token
        token_url = f"{settings.DATABRICKS_HOST}/oidc/v1/token"
        logger.info(f"AUTH.get_token: Requesting new token | key={cache_key}")

        try:
            resp = httpx.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": cid,
                    "client_secret": csecret,
                    "scope": "all-apis",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"AUTH.get_token: HTTP {e.response.status_code} from token endpoint")
            raise RuntimeError(f"Token request failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"AUTH.get_token: Request error: {e}")
            raise RuntimeError(f"Token request failed: {e}") from e

        data = resp.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", 3600)

        self._token_cache[cache_key] = {
            "token": token,
            "expires_at": time.time() + expires_in,
        }

        logger.info(f"AUTH.get_token: New token cached | key={cache_key} | expires_in={expires_in}s")
        return token


# Singleton
token_provider = SecureTokenProvider()
