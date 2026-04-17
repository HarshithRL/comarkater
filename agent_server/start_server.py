"""Entry point — AgentServer (API) + Flask UI (internal), one Databricks App."""

import os
import logging
logging.basicConfig(level="INFO", format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
_log = logging.getLogger("start_server")

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

from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# ── SSE anti-buffering middleware (raw ASGI — does NOT buffer response body) ──
# BaseHTTPMiddleware buffers the entire response before passing it through,
# which causes the Databricks App proxy to time out on long SSE streams (~60s+).
# Raw ASGI middleware intercepts only the response.start message to inject headers
# and passes all body chunks (http.response.body) directly through without buffering.
class SSENoBufferMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_sse_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                content_type = next(
                    (v.decode() for k, v in headers if k.lower() == b"content-type"),
                    "",
                )
                if "text/event-stream" in content_type:
                    headers.append((b"x-accel-buffering", b"no"))
                    headers.append((b"cache-control", b"no-cache, no-transform"))
                    headers.append((b"connection", b"keep-alive"))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_sse_headers)

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
