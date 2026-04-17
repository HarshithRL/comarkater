"""All configuration, constants, and settings. LEAF module — no internal imports."""

import os
from dataclasses import dataclass


def _ensure_https(url: str) -> str:
    """Ensure URL has https:// prefix. Databricks App runtime sets DATABRICKS_HOST without protocol."""
    if url and not url.startswith("http"):
        return f"https://{url}"
    return url


@dataclass(frozen=True)
class Settings:
    """Application-wide settings, read from environment variables."""

    # Databricks
    DATABRICKS_HOST: str = _ensure_https(os.environ.get("DATABRICKS_HOST", "https://dbc-540c0e05-7c19.cloud.databricks.com"))

    # AI Gateway (for LLM calls via SP token)
    AI_GATEWAY_URL: str = os.environ.get(
        "AI_GATEWAY_URL",
        "https://2276245144129479.ai-gateway.cloud.databricks.com/mlflow/v1",
    )

    # Model endpoints
    LLM_ENDPOINT_NAME: str = os.environ.get("LLM_ENDPOINT_NAME", "databricks-gpt-5-2")

    # Genie API
    GENIE_SPACE_ID: str = os.environ.get("GENIE_SPACE_ID", "01f1369ea0a11cd9a386669de0083fd8")
    GENIE_POLL_TIMEOUT: int = int(os.environ.get("GENIE_POLL_TIMEOUT", "120"))

    # SP auth
    SECRETS_SCOPE: str = os.environ.get("SECRETS_SCOPE", "agent-sp-credentials")
    REQUIRE_PER_REQUEST_SP: bool = os.environ.get("REQUIRE_PER_REQUEST_SP", "false").lower() == "true"

    # Trace table (Unity Catalog)
    TRACE_CATALOG: str = os.environ.get("TRACE_CATALOG", "channel")
    TRACE_SCHEMA: str = os.environ.get("TRACE_SCHEMA", "gold_channel")
    TRACE_TABLE: str = os.environ.get("TRACE_TABLE", "comarketer_traces")

    # SQL Warehouse (for Delta table writes)
    SQL_WAREHOUSE_ID: str = os.environ.get("SQL_WAREHOUSE_ID", "")

    # Memory (Lakebase + embeddings)
    LAKEBASE_INSTANCE_NAME: str = os.environ.get("LAKEBASE_INSTANCE_NAME", "agentmemory")
    LAKEBASE_MODE: str = os.environ.get("LAKEBASE_MODE", "autoscaling")  # "autoscaling" or "provisioned"
    LAKEBASE_PROJECT: str = os.environ.get("LAKEBASE_PROJECT", "agentmemory")  # Autoscaling project name
    LAKEBASE_BRANCH: str = os.environ.get("LAKEBASE_BRANCH", "production")  # Branch name (not UID)
    # Note: autoscaling_endpoint expects an endpoint NAME (e.g. "ep-damp-mouse-d2rgetbb"),
    # NOT a full URL. Use project+branch for auto-resolution instead.
    EMBEDDING_ENDPOINT: str = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    EMBEDDING_DIMS: int = int(os.environ.get("EMBEDDING_DIMS", "1024"))

    # Environment (use COMARKETER_ENV — AGENT_ENV is reserved/injected by Databricks App runtime as "production")
    ENV: str = os.environ.get("COMARKETER_ENV", "development")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()


# ── Multi-client SP registry ──
# Maps sp_id → client info + secret key names in the Databricks secrets scope.
# Secrets are read at runtime via WorkspaceClient().secrets.get_secret()
CLIENT_REGISTRY: dict[str, dict] = {
    "sp-igp-001": {
        "client_id": "72994",
        "client_name": "IGP",
        "display_name": "IGP Team",
        "secret_client_id_key": "igp-sp-client-id",
        "secret_client_secret_key": "igp-sp-client-secret",
    },
    "sp-pepe-002": {
        "client_id": "81001",
        "client_name": "Pepe Jeans",
        "display_name": "Pepe Jeans Team",
        "secret_client_id_key": "pepe-sp-client-id",
        "secret_client_secret_key": "pepe-sp-client-secret",
    },
    "sp-crocs-003": {
        "client_id": "65432",
        "client_name": "Crocs",
        "display_name": "Crocs Team",
        "secret_client_id_key": "shopcrocsmt-sp-client-id",
        "secret_client_secret_key": "shopcrocsmt-sp-client-secret",
    },
    "default": {
        "client_id": "00000",
        "client_name": "Demo",
        "display_name": "Demo User",
        "secret_client_id_key": "default-sp-client-id",
        "secret_client_secret_key": "default-sp-client-secret",
    },
}