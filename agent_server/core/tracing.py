"""Custom trace logger — writes structured trace records to a Delta table.

Two logging paths:
1. MLflow trace tags — attached to the autolog trace for the experiment UI
2. Delta table — via Databricks SQL Statement API (requires SQL_WAREHOUSE_ID)

If SQL_WAREHOUSE_ID is not set, Delta writes are skipped (logged as warning once).
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import mlflow
from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """One trace record per agent request."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    client_id: str = ""
    sp_id: str = ""
    user_query: str = ""
    intent: str = ""
    agent_route: str = ""
    response_text: str = ""
    llm_call_count: int = 0
    latency_ms: int = 0
    status: str = "success"
    error_message: str = ""
    conversation_id: str = ""
    conversation: str = "{}"
    mlflow_trace_id: str = ""
    environment: str = settings.ENV

    def to_dict(self) -> dict:
        """Convert to dict for logging/storage."""
        return asdict(self)


def _escape_sql(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    if value is None:
        return ""
    return str(value).replace("'", "''")


class TraceLogger:
    """Logs trace records to MLflow tags and Delta table."""

    def __init__(self) -> None:
        self._start_times: dict[str, float] = {}
        self._sql_client = None
        self._sql_warned = False
        self._table = f"{settings.TRACE_CATALOG}.{settings.TRACE_SCHEMA}.{settings.TRACE_TABLE}"

    def _get_sql_client(self):
        """Lazy-init Databricks SQL statement execution client."""
        if self._sql_client is not None:
            return self._sql_client

        if not settings.SQL_WAREHOUSE_ID:
            if not self._sql_warned:
                logger.warning("TRACE: SQL_WAREHOUSE_ID not set — Delta table writes disabled")
                self._sql_warned = True
            return None

        try:
            from databricks.sdk import WorkspaceClient
            self._sql_client = WorkspaceClient()
            logger.info(f"TRACE: SQL client initialized, warehouse={settings.SQL_WAREHOUSE_ID}")
            return self._sql_client
        except Exception as e:
            logger.warning(f"TRACE: Failed to init SQL client: {e}")
            return None

    def _ensure_table(self) -> None:
        """Create the trace table if it doesn't exist."""
        client = self._get_sql_client()
        if not client:
            return

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table} (
            trace_id        STRING      NOT NULL,
            request_id      STRING      NOT NULL,
            timestamp       STRING      NOT NULL,
            client_id       STRING      NOT NULL,
            sp_id           STRING      NOT NULL,
            user_query      STRING,
            intent          STRING,
            agent_route     STRING,
            response_text   STRING,
            llm_call_count  INT,
            latency_ms      INT,
            status          STRING      NOT NULL,
            error_message   STRING,
            conversation_id STRING,
            conversation    STRING,
            mlflow_trace_id STRING,
            environment     STRING
        )
        USING DELTA
        COMMENT 'CoMarketer agent trace logs'
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
        """
        try:
            client.statement_execution.execute_statement(
                warehouse_id=settings.SQL_WAREHOUSE_ID,
                statement=create_sql,
                wait_timeout="30s",
            )
            logger.info(f"TRACE: Table {self._table} ensured")
        except Exception as e:
            logger.warning(f"TRACE: Failed to create table: {e}")

    def _write_to_delta(self, record: TraceRecord) -> None:
        """Write a trace record to the Delta table via SQL."""
        client = self._get_sql_client()
        if not client:
            return

        r = record
        insert_sql = f"""
        INSERT INTO {self._table} VALUES (
            '{_escape_sql(r.trace_id)}',
            '{_escape_sql(r.request_id)}',
            '{_escape_sql(r.timestamp)}',
            '{_escape_sql(r.client_id)}',
            '{_escape_sql(r.sp_id)}',
            '{_escape_sql(r.user_query)}',
            '{_escape_sql(r.intent)}',
            '{_escape_sql(r.agent_route)}',
            '{_escape_sql(r.response_text)}',
            {r.llm_call_count},
            {r.latency_ms},
            '{_escape_sql(r.status)}',
            '{_escape_sql(r.error_message)}',
            '{_escape_sql(r.conversation_id)}',
            '{_escape_sql(r.conversation)}',
            '{_escape_sql(r.mlflow_trace_id)}',
            '{_escape_sql(r.environment)}'
        )
        """
        try:
            client.statement_execution.execute_statement(
                warehouse_id=settings.SQL_WAREHOUSE_ID,
                statement=insert_sql,
                wait_timeout="10s",
            )
            logger.info(f"TRACE.delta: Wrote trace {r.trace_id} to {self._table}")
        except Exception as e:
            logger.warning(f"TRACE.delta: Failed to write: {e}")

    def start_trace(self, request_id: str) -> str:
        """Mark the start of a request. Returns trace_id."""
        trace_id = str(uuid.uuid4())
        self._start_times[request_id] = time.time()
        logger.info(f"TRACE.start: request_id={request_id} trace_id={trace_id}")
        return trace_id

    def end_trace(self, record: TraceRecord) -> None:
        """Finalize and log the trace record."""
        # Calculate latency
        start = self._start_times.pop(record.request_id, None)
        if start:
            record.latency_ms = int((time.time() - start) * 1000)

        # Truncate response for storage
        if len(record.response_text) > 2000:
            record.response_text = record.response_text[:2000] + "...[truncated]"

        # 1. Write to Delta table
        self._write_to_delta(record)

        # 3. Log structured record to stdout (Databricks log ingestion)
        logger.info(f"TRACE.record: {json.dumps(record.to_dict())}")

        logger.info(
            f"TRACE.end: request_id={record.request_id} "
            f"client={record.client_id} intent={record.intent} "
            f"latency={record.latency_ms}ms status={record.status}"
        )

    def init_table(self) -> None:
        """Call once at startup to ensure the Delta table exists."""
        self._ensure_table()


# Singleton
trace_logger = TraceLogger()


def _fmt_flow_value(value) -> str:
    """Stringify a flow-log field; truncate long values to 2000 chars."""
    s = str(value) if not isinstance(value, str) else value
    return s if len(s) <= 2000 else s[:2000] + "...[truncated]"


def flow_log(request_id: str, phase: str, **fields) -> None:
    """Emit one INFO-level structured flow line. Always on.

    Grep stdout for ``FLOW[<first 8 of request_id>]`` to get the full trail
    for a given request. Fields with value ``None`` are skipped.
    """
    rid = (request_id or "unknown")[:8]
    kv = " ".join(f"{k}={_fmt_flow_value(v)}" for k, v in fields.items() if v is not None)
    logger.info(f"FLOW[{rid}] {phase} | {kv}")


def flow_debug(request_id: str, phase: str, **fields) -> None:
    """DEBUG-level line carrying LLM outputs / verbose payloads.

    Activated by setting ``LOG_LEVEL=DEBUG``. Values are wrapped in ``repr``
    so multi-line JSON stays readable on a single log line.
    """
    rid = (request_id or "unknown")[:8]
    kv = " ".join(f"{k}={_fmt_flow_value(v)!r}" for k, v in fields.items() if v is not None)
    logger.debug(f"FLOW[{rid}] {phase}.debug | {kv}")
