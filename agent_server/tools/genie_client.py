"""Sync Genie API client — submits queries, polls status, fetches results.

Three-step flow:
1. start_conversation() — POST to start a Genie conversation
2. check_status() — GET message status (caller implements poll loop)
3. fetch_result() — GET the SQL query result data

Uses sync httpx.Client — safe as module-level singleton (no event loop issues).
Token passed per-call from config["configurable"]["sp_token"].
"""

import logging

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

# Genie API status values
STATUS_COMPLETED = "COMPLETED"
STATUS_EXECUTING = "EXECUTING_QUERY"
STATUS_FAILED = "FAILED"
STATUS_FEEDBACK = "FEEDBACK_NEEDED"
STATUS_FILTERING = "FILTERING_CONTEXT"
STATUS_ASKING_AI = "ASKING_AI"
STATUS_PENDING = "PENDING_WAREHOUSE"


class GenieClient:
    """Sync Databricks Genie API client. Module-level singleton safe."""

    def __init__(self):
        self.client = httpx.Client(timeout=60)
        logger.info("GENIE_CLIENT: initialized | host=%s", settings.DATABRICKS_HOST)

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def start_conversation(self, question: str, space_id: str, token: str) -> tuple[str, str]:
        """Submit a question to Genie and start a new conversation.

        Returns:
            Tuple of (conversation_id, message_id).
        """
        url = f"{settings.DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}/start-conversation"
        logger.info(
            "GENIE_CLIENT.start: submitting | space=%s | question='%s'",
            space_id, question[:120],
        )

        try:
            resp = self.client.post(url, headers=self._headers(token), json={"content": question})
            resp.raise_for_status()
            data = resp.json()

            conv_id = data["conversation"]["id"]
            msg_id = data["message"]["id"]
            logger.info("GENIE_CLIENT.start: conv_id=%s | msg_id=%s", conv_id, msg_id)
            return conv_id, msg_id

        except httpx.HTTPStatusError as e:
            logger.error("GENIE_CLIENT.start: HTTP %d | body=%s", e.response.status_code, e.response.text[:300])
            raise
        except Exception:
            logger.exception("GENIE_CLIENT.start: failed")
            raise

    def check_status(self, space_id: str, conv_id: str, msg_id: str, token: str) -> dict:
        """Check Genie message status (single poll — caller implements loop).

        Returns:
            Dict with keys: status, sql, statement_id, raw.
        """
        url = f"{settings.DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}"

        try:
            resp = self.client.get(url, headers=self._headers(token))
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")
            logger.info("GENIE_CLIENT.poll: status=%s | conv_id=%s", status, conv_id)

            sql, stmt = None, None
            for att in data.get("attachments", []):
                if "query" in att:
                    sql = att["query"].get("query")
                    stmt = att["query"].get("statement_id")

            if sql:
                logger.debug("GENIE_CLIENT.poll: SQL found | %s", sql[:200])

            return {"status": status, "sql": sql, "statement_id": stmt, "raw": data}

        except httpx.HTTPStatusError as e:
            logger.error("GENIE_CLIENT.poll: HTTP %d | conv_id=%s", e.response.status_code, conv_id)
            raise
        except Exception:
            logger.exception("GENIE_CLIENT.poll: failed | conv_id=%s", conv_id)
            raise

    def fetch_result(
        self, space_id: str, conv_id: str, msg_id: str, attachment_id: str, token: str,
    ) -> dict:
        """Fetch the query result data for a completed Genie attachment.

        Returns:
            Raw result dict containing statement_response with columns and data_array.
        """
        url = (
            f"{settings.DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}"
            f"/conversations/{conv_id}/messages/{msg_id}"
            f"/attachments/{attachment_id}/query-result"
        )
        logger.info("GENIE_CLIENT.fetch: attachment_id=%s | conv_id=%s", attachment_id, conv_id)

        try:
            resp = self.client.get(url, headers=self._headers(token))
            resp.raise_for_status()

            result = resp.json()
            if "statement_response" in result:
                rows = len(result["statement_response"].get("result", {}).get("data_array", []))
                logger.info("GENIE_CLIENT.fetch: %d rows | conv_id=%s", rows, conv_id)

            return result

        except httpx.HTTPStatusError as e:
            logger.error("GENIE_CLIENT.fetch: HTTP %d | conv_id=%s", e.response.status_code, conv_id)
            raise
        except Exception:
            logger.exception("GENIE_CLIENT.fetch: failed | conv_id=%s", conv_id)
            raise


# Module-level singleton — safe with sync httpx.Client (no event loop issues)
genie_client = GenieClient()
