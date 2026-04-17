"""Genie REST tool handler returning structured ``GenieResponse`` objects."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from agents.campaign_insight.contracts import GenieResponse

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "FEEDBACK_NEEDED", "CANCELLED"}
_POLL_TIMEOUT_S = 120.0
_HTTP_TIMEOUT_S = 60.0

# Detect SQL leaking into the NL query. Matches the common SQL keywords at
# clause boundaries; one hit is enough to reject the query.
_SQL_GUARD = re.compile(
    r"(?is)(?:^|\b)(?:select\s+[\w*`\"'(]"
    r"|with\s+\w+\s+as\s*\("
    r"\bfrom\s+[\w`\"']"
    r"|\bwhere\s+\w+\s*[=<>]"
    r"|\bgroup\s+by\b"
    r"|\border\s+by\b"
    r"|\bjoin\s+\w+"
    r"|\bunion\s+(?:all\s+)?select\b"
    r"|\blimit\s+\d+"
    r")"
)


def _looks_like_sql(text: str) -> bool:
    """Return True if the NL query appears to contain SQL."""
    if not text:
        return False
    # Quick rejection on obvious statement terminators
    stripped = text.strip()
    if ";" in stripped or "`" in stripped:
        return True
    return bool(_SQL_GUARD.search(stripped))


class ToolHandler:
    """Async Genie REST client that returns ``GenieResponse`` dataclasses.

    Wraps ``start-conversation`` + poll + ``query-result`` into a single
    awaitable. Never formats results as text — callers are expected to run
    the payload through ``TableAnalyzer`` / ``TableBuilder``.
    """

    def __init__(
        self,
        genie_space_id: str,
        sp_token: str,
        databricks_host: str,
    ) -> None:
        """Initialize the handler.

        Args:
            genie_space_id: Target Genie space identifier.
            sp_token: Bearer token (service principal or OBO user token).
            databricks_host: Databricks workspace host (``https://...``).
        """
        self.genie_space_id = genie_space_id
        self.sp_token = sp_token
        self.databricks_host = databricks_host.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {sp_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def execute_query(self, nl_query: str) -> GenieResponse:
        """Run a single natural-language query against the Genie space.

        Args:
            nl_query: The NL question to dispatch.

        Returns:
            A ``GenieResponse`` with ``status`` set to ``"success"``,
            ``"error"``, or ``"feedback_needed"``.
        """
        preview = (nl_query or "")[:120].replace("\n", " ")
        start = time.monotonic()

        # Guard: reject anything that looks like SQL. Genie must receive NL only.
        if _looks_like_sql(nl_query):
            logger.error(
                "genie.execute_query REJECTED sql-like query=%r", preview,
            )
            return GenieResponse(
                status="error",
                error_message=(
                    "SQL detected in query — Genie requires natural-language "
                    "questions only. Rewrite as plain English."
                ),
            )

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                conv_id, msg_id = await self._start_conversation(client, nl_query)
                final_msg = await self._poll_message(client, conv_id, msg_id)
                response = await self._finalize(client, conv_id, msg_id, final_msg)
        except httpx.HTTPStatusError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "genie.execute_query http_error query=%r status=%s latency_ms=%d",
                preview, exc.response.status_code, latency_ms,
            )
            return GenieResponse(
                status="error",
                error_message=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "genie.execute_query timeout query=%r latency_ms=%d err=%s",
                preview, latency_ms, exc,
            )
            return GenieResponse(status="error", error_message=f"timeout: {exc}")
        except Exception as exc:  # noqa: BLE001 — defensive: never crash node
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.exception(
                "genie.execute_query unexpected_error query=%r latency_ms=%d",
                preview, latency_ms,
            )
            return GenieResponse(status="error", error_message=str(exc))

        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "genie.execute_query query=%r status=%s rows=%d latency_ms=%d",
            preview, response.status, response.row_count, latency_ms,
        )
        return response

    async def execute_query_with_retry(
        self, nl_query: str, max_retries: int = 2
    ) -> GenieResponse:
        """Run a query with retry on transient failures only.

        Retries only on HTTP 5xx / timeouts (``status == "error"`` with a
        retryable error_message). Does NOT retry on ``"feedback_needed"`` or
        authoritative Genie ``FAILED`` responses — those are terminal.

        Args:
            nl_query: NL question.
            max_retries: Additional attempts on top of the initial try.

        Returns:
            Final ``GenieResponse``.
        """
        last: GenieResponse | None = None
        for attempt in range(max_retries + 1):
            try:
                from langgraph.config import get_stream_writer
                get_stream_writer()({
                    "event_type": "genie_status",
                    "phase": "start",
                    "message": (
                        "Fetching data..." if attempt == 0
                        else f"Retrying data fetch (attempt {attempt + 1})..."
                    ),
                })
            except Exception:
                pass
            resp = await self.execute_query(nl_query)
            try:
                from langgraph.config import get_stream_writer
                get_stream_writer()({
                    "event_type": "genie_status",
                    "phase": "done",
                    "rows": getattr(resp, "row_count", None),
                    "status": resp.status,
                })
            except Exception:
                pass
            last = resp
            if resp.status in ("success", "feedback_needed"):
                return resp
            if not self._is_retryable(resp.error_message):
                return resp
            if attempt < max_retries:
                backoff = 2.0 * (2 ** attempt)
                logger.info(
                    "genie.retry attempt=%d/%d backoff=%.1fs err=%s",
                    attempt + 1, max_retries, backoff, resp.error_message[:120],
                )
                await asyncio.sleep(backoff)
        return last or GenieResponse(status="error", error_message="no attempts")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _is_retryable(error_message: str) -> bool:
        msg = (error_message or "").lower()
        if "timeout" in msg:
            return True
        if msg.startswith("http 5"):
            return True
        return False

    async def _start_conversation(
        self, client: httpx.AsyncClient, nl_query: str
    ) -> tuple[str, str]:
        url = (
            f"{self.databricks_host}/api/2.0/genie/spaces/"
            f"{self.genie_space_id}/start-conversation"
        )
        resp = await client.post(url, headers=self._headers, json={"content": nl_query})
        resp.raise_for_status()
        body = resp.json()
        conv_id = body["conversation"]["id"]
        msg_id = body["message"]["id"]
        return conv_id, msg_id

    async def _poll_message(
        self, client: httpx.AsyncClient, conv_id: str, msg_id: str
    ) -> dict[str, Any]:
        url = (
            f"{self.databricks_host}/api/2.0/genie/spaces/"
            f"{self.genie_space_id}/conversations/{conv_id}/messages/{msg_id}"
        )
        deadline = time.monotonic() + _POLL_TIMEOUT_S
        delay = 2.0
        while True:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status", "")
            if status in _TERMINAL_STATUSES:
                return body
            if time.monotonic() >= deadline:
                raise asyncio.TimeoutError(
                    f"genie polling exceeded {_POLL_TIMEOUT_S}s (last status={status})"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

    async def _finalize(
        self,
        client: httpx.AsyncClient,
        conv_id: str,
        msg_id: str,
        message: dict[str, Any],
    ) -> GenieResponse:
        trace_id = f"{conv_id}/{msg_id}"
        status = message.get("status", "")
        attachments = message.get("attachments", []) or []

        if status == "FAILED":
            err = message.get("error", {}) or {}
            err_text = err.get("error") or err.get("message") or "Genie FAILED"
            return GenieResponse(
                status="error",
                error_message=str(err_text),
                genie_trace_id=trace_id,
            )

        if status == "FEEDBACK_NEEDED":
            feedback = ""
            for att in attachments:
                text = att.get("text") or {}
                content = text.get("content")
                if content:
                    feedback = content
                    break
            return GenieResponse(
                status="feedback_needed",
                error_message=feedback or "Genie requested clarification",
                genie_trace_id=trace_id,
            )

        if status != "COMPLETED":
            return GenieResponse(
                status="error",
                error_message=f"unexpected terminal status: {status}",
                genie_trace_id=trace_id,
            )

        # COMPLETED — find the query attachment
        for att in attachments:
            if "query" not in att:
                continue
            attachment_id = att.get("attachment_id") or att.get("id")
            query_block = att.get("query", {}) or {}
            sql = query_block.get("query", "") or ""
            if not attachment_id:
                continue
            result_url = (
                f"{self.databricks_host}/api/2.0/genie/spaces/"
                f"{self.genie_space_id}/conversations/{conv_id}/messages/"
                f"{msg_id}/attachments/{attachment_id}/query-result"
            )
            r = await client.get(result_url, headers=self._headers)
            r.raise_for_status()
            payload = r.json()
            stmt = payload.get("statement_response", {}) or {}
            manifest = stmt.get("manifest", {}) or {}
            schema = manifest.get("schema", {}) or {}
            columns = list(schema.get("columns", []) or [])
            result = stmt.get("result", {}) or {}
            data_array = list(result.get("data_array", []) or [])
            return GenieResponse(
                columns=columns,
                data_array=data_array,
                row_count=len(data_array),
                sql=sql,
                status="success",
                genie_trace_id=trace_id,
            )

        # COMPLETED but no query attachment — treat as text-only (no data)
        return GenieResponse(
            status="error",
            error_message="Genie completed without a query attachment",
            genie_trace_id=trace_id,
        )
