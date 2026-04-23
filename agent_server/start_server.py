"""Entry point — AgentServer (API) + Flask UI (internal), one Databricks App."""

import asyncio
import os
import logging
logging.basicConfig(level="INFO", format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
_log = logging.getLogger("start_server")

SSE_KEEPALIVE_INTERVAL_S = 5.0
SSE_KEEPALIVE_PAYLOAD = b": keepalive\n\n"

_log.info("START_SERVER: ── booting ──")
_log.info(f"START_SERVER: MLFLOW_TRACKING_URI (before setdefault)={os.environ.get('MLFLOW_TRACKING_URI', 'NOT SET')}")
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
_log.info(f"START_SERVER: MLFLOW_TRACKING_URI (after setdefault)={os.environ.get('MLFLOW_TRACKING_URI')}")
_log.info(f"START_SERVER: DATABRICKS_HOST={os.environ.get('DATABRICKS_HOST', 'NOT SET')}")
_log.info(f"START_SERVER: MLFLOW_EXPERIMENT_NAME={os.environ.get('MLFLOW_EXPERIMENT_NAME', 'NOT SET')}")
_log.info(f"START_SERVER: MLFLOW_EXPERIMENT_ID={os.environ.get('MLFLOW_EXPERIMENT_ID', 'NOT SET')}")
_log.info(f"START_SERVER: AGENT_ENV={os.environ.get('AGENT_ENV', 'NOT SET')} (system-injected by Databricks, always 'production')")
_log.info(f"START_SERVER: COMARKETER_ENV={os.environ.get('COMARKETER_ENV', 'NOT SET')}")

import agent  # noqa: F401 — registers @invoke/@stream
_log.info("START_SERVER: agent module imported OK")

from mlflow.genai.agent_server import setup_mlflow_git_based_version_tracking

# Long-running agent server: survives the Databricks App proxy 120s ceiling by
# running the agent as a background asyncio task with events persisted to
# Lakebase. Clients POST /invocations with {"background": true, "stream": true}
# and tail events via the returned SSE (or reconnect via GET /responses/{id}
# ?stream=true&starting_after=<seq>). task_timeout_seconds is our hard ceiling.
from databricks_ai_bridge.long_running.server import LongRunningAgentServer


class _ResponseIdInjector(LongRunningAgentServer):
    """Inject response_id into every stream event so the client can reconnect.

    The base class creates response_id internally but does not expose it to the
    client — clients cannot reconnect to a specific run without it. Overriding
    transform_stream_event attaches it to every persisted frame; the client
    reads the first frame that carries it and stores it for GET /responses/{id}.
    """

    def transform_stream_event(self, event: dict, response_id: str) -> dict:
        if isinstance(event, dict):
            return {**event, "response_id": response_id}
        return event


agent_server = _ResponseIdInjector(
    agent_type="ResponsesAgent",
    db_instance_name=os.environ.get("LAKEBASE_INSTANCE_NAME", "agentmemory"),
    task_timeout_seconds=900.0,    # 15 min hard ceiling for any single run
    poll_interval_seconds=0.5,     # SSE tail latency from the DB
)
app = agent_server.app

# ── STM + LTM lifecycle hooks — open Lakebase pools at startup, close on shutdown ──
from agent import stm_startup, stm_shutdown, ltm_startup, ltm_shutdown  # noqa: E402
app.router.on_startup.append(stm_startup)
app.router.on_startup.append(ltm_startup)
# LIFO shutdown order: LTM closes before STM (reverse of startup).
app.router.on_shutdown.append(ltm_shutdown)
app.router.on_shutdown.append(stm_shutdown)
_log.info("START_SERVER: STM + LTM lifecycle hooks registered")

# ── SSE anti-buffering middleware (raw ASGI — does NOT buffer response body) ──
# BaseHTTPMiddleware buffers the entire response before passing it through,
# which causes the Databricks App proxy to time out on long SSE streams (~60s+).
# Raw ASGI middleware intercepts response.start to inject headers AND spawns a
# wire-level keepalive task that writes `: keepalive\n\n` comment frames every
# 5s directly via the ASGI send. Sitting at the ASGI layer ensures bytes reach
# the socket even if upstream frameworks buffer app-generated frames.
class SSENoBufferMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # State for this single request/response cycle.
        is_sse = False
        keepalive_task: asyncio.Task | None = None
        send_lock = asyncio.Lock()

        # Capture inbound x-request-id (set by the Databricks App edge proxy)
        # so server logs correlate with browser DevTools headers.
        inbound_req_id = ""
        try:
            for k, v in scope.get("headers", []):
                if k.lower() == b"x-request-id":
                    inbound_req_id = v.decode("latin-1", errors="replace")[:80]
                    break
        except Exception:
            pass

        async def locked_send(message):
            # Serialize all writes to the ASGI send callable so keepalive
            # comment frames cannot interleave with real body frames on the
            # same HTTP/2 stream.
            async with send_lock:
                await send(message)

        async def keepalive_loop():
            try:
                while True:
                    await asyncio.sleep(SSE_KEEPALIVE_INTERVAL_S)
                    try:
                        await locked_send({
                            "type": "http.response.body",
                            "body": SSE_KEEPALIVE_PAYLOAD,
                            "more_body": True,
                        })
                    except Exception:
                        # Client gone or send closed — stop trying.
                        return
            except asyncio.CancelledError:
                return

        async def wrapped_send(message):
            nonlocal is_sse, keepalive_task
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                content_type = next(
                    (v.decode() for k, v in headers if k.lower() == b"content-type"),
                    "",
                )
                if "text/event-stream" in content_type:
                    is_sse = True
                    headers.append((b"x-accel-buffering", b"no"))
                    headers.append((b"cache-control", b"no-cache, no-transform"))
                    headers.append((b"connection", b"keep-alive"))
                    message = {**message, "headers": headers}
                    _log.info(
                        "SSE.open path=%s x_request_id=%s",
                        scope.get("path", ""), inbound_req_id or "none",
                    )
                await locked_send(message)
                if is_sse and keepalive_task is None:
                    keepalive_task = asyncio.create_task(keepalive_loop())
                return
            if message["type"] == "http.response.body":
                await locked_send(message)
                if is_sse and not message.get("more_body", False) and keepalive_task is not None:
                    keepalive_task.cancel()
                return
            await locked_send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            if keepalive_task is not None and not keepalive_task.done():
                keepalive_task.cancel()

app.add_middleware(SSENoBufferMiddleware)
_log.info("START_SERVER: SSE anti-buffering middleware added")

# ── Mount Flask UI on /ui/ ──
try:
    from asgiref.wsgi import WsgiToAsgi
    from ui import create_ui_app

    flask_app = create_ui_app()
    app.mount("/ui", WsgiToAsgi(flask_app))
    print("✅ UI mounted at /ui/")
except Exception as e:
    print(f"⚠️ UI not mounted: {e}. API still works.")

try:
    setup_mlflow_git_based_version_tracking()
except Exception as e:
    print(f"⚠️ Git-based version tracking not available: {e}. Skipping.")


def main():
    """Start the agent server."""
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
