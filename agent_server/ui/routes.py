"""UI routes — chat interface, health, and debug diagnostics."""

import json
import logging
import os
import uuid

from flask import Blueprint, render_template, jsonify, request, Response

logger = logging.getLogger(__name__)
ui_bp = Blueprint("ui", __name__)


CLIENTS = [
    {"id": "sp-igp-001", "name": "IGP"},
    {"id": "sp-pepe-002", "name": "Pepe Jeans"},
    {"id": "sp-crocs-003", "name": "Crocs"},
]


@ui_bp.route("/")
def index():
    """Render the chat UI."""
    return render_template("index.html", clients=CLIENTS)


@ui_bp.route("/debug-stream")
def debug_stream():
    """Render the raw SSE stream debug page."""
    return render_template("debug_stream.html", clients=CLIENTS)


def _build_agent_request(query: str, sp_id: str, conversation_id: str = ""):
    """Build a ResponsesAgentRequest from UI chat params."""
    from mlflow.types.responses import ResponsesAgentRequest
    return ResponsesAgentRequest(
        input=[{"role": "user", "content": query}],
        custom_inputs={
            "sp_id": sp_id,
            "user_name": "UI User",
            "user_id": "ui-user",
            "conversation_id": conversation_id or f"ui-{uuid.uuid4().hex[:8]}",
            "task_type": "analytics",
        },
    )


ALLOWED_SP_IDS = {"sp-igp-001", "sp-pepe-002", "sp-crocs-003", "default"}
MAX_QUERY_LENGTH = 5000


@ui_bp.route("/chat", methods=["POST"])
def chat():
    """Handle chat request — supports both streaming (SSE) and non-streaming (JSON)."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

    query = data.get("query", "").strip()
    sp_id = data.get("sp_id", "sp-igp-001")
    use_stream = data.get("stream", False)

    if not query:
        return jsonify({"status": "error", "message": "Please enter a question"}), 400
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({"status": "error", "message": f"Query too long (max {MAX_QUERY_LENGTH} characters)"}), 400
    if sp_id not in ALLOWED_SP_IDS:
        return jsonify({"status": "error", "message": "Invalid client selected"}), 400

    conversation_id = data.get("conversation_id", "")
    agent_request = _build_agent_request(query, sp_id, conversation_id)

    if use_stream:
        return _handle_stream_chat(agent_request)
    else:
        return _handle_invoke_chat(agent_request)


def _handle_invoke_chat(agent_request):
    """Non-streaming: call predict(), return JSON with mlflow_trace_id."""
    try:
        import mlflow
        from agent import agent as comarketer_agent

        response = comarketer_agent.predict(agent_request)

        # Capture MLflow trace_id after predict() returns (trace span has closed)
        mlflow_trace_id = ""
        try:
            mlflow_trace_id = mlflow.get_last_active_trace_id() or ""
        except Exception:
            pass

        output_list = []
        for item in (response.output or []):
            if hasattr(item, "to_dict"):
                output_list.append(item.to_dict())
            elif hasattr(item, "__dict__"):
                output_list.append(item.__dict__)
            elif isinstance(item, dict):
                output_list.append(item)
            else:
                output_list.append({"content": [{"text": str(item)}]})

        custom_outputs = dict(response.custom_outputs or {})
        if mlflow_trace_id:
            custom_outputs["mlflow_trace_id"] = mlflow_trace_id

        return jsonify({
            "output": output_list,
            "custom_outputs": custom_outputs,
        })

    except Exception as e:
        logger.error(f"UI.chat: invoke error | error={e}", exc_info=True)
        return jsonify({
            "output": [{"content": [{"text": "Something went wrong. Please try again."}]}],
            "custom_outputs": {"status": "error"},
        }), 500


def _handle_stream_chat(agent_request):
    """Streaming: call predict_stream(), return SSE events."""

    def generate():
        try:
            from agent import agent as comarketer_agent

            for event in comarketer_agent.predict_stream(agent_request):
                if hasattr(event, "model_dump"):
                    event_dict = event.model_dump(exclude_none=True)
                elif isinstance(event, dict):
                    event_dict = event
                else:
                    event_dict = {"raw": str(event)}
                yield f"data: {json.dumps(event_dict)}\n\n"

        except Exception as e:
            logger.error(f"UI.chat: stream error | error={e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@ui_bp.route("/feedback", methods=["POST"])
def feedback():
    """Log user feedback (like/dislike + optional comment) to MLflow trace."""
    data = request.get_json() or {}
    trace_id = (data.get("trace_id") or "").strip()
    is_helpful = data.get("is_helpful")
    comment = (data.get("comment") or "").strip() or None
    user_id = (data.get("user_id") or "ui-user").strip()

    if not trace_id:
        return jsonify({"error": "trace_id is required"}), 400
    if is_helpful is None:
        return jsonify({"error": "is_helpful is required"}), 400

    try:
        from feedback import log_user_feedback
        success = log_user_feedback(
            trace_id=trace_id,
            is_helpful=bool(is_helpful),
            comment=comment,
            user_id=user_id,
        )
        if success:
            return jsonify({"status": "recorded"})
        return jsonify({"error": "Failed to log feedback to MLflow"}), 500
    except Exception as e:
        logger.error(f"UI.feedback: error | error={e}", exc_info=True)
        return jsonify({"status": "error", "message": "Failed to record feedback"}), 500


@ui_bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "agent": "comarketer", "stage": "2"})


@ui_bp.route("/debug")
def debug():
    """Debug endpoint — shows env vars, auth status, tracing, and errors."""
    info = {
        "env": {
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "NOT SET"),
            "MLFLOW_EXPERIMENT_ID": os.environ.get("MLFLOW_EXPERIMENT_ID", "NOT SET"),
            "SQL_WAREHOUSE_ID": os.environ.get("SQL_WAREHOUSE_ID", "NOT SET"),
            "AGENT_ENV": os.environ.get("AGENT_ENV", "NOT SET"),
            "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", "NOT SET"),
            "SECRETS_SCOPE": os.environ.get("SECRETS_SCOPE", "NOT SET"),
            "AI_GATEWAY_URL": os.environ.get("AI_GATEWAY_URL", "NOT SET"),
            "LLM_ENDPOINT_NAME": os.environ.get("LLM_ENDPOINT_NAME", "NOT SET"),
            "AGENTBRICKS_ENDPOINT_NAME": os.environ.get("AGENTBRICKS_ENDPOINT_NAME", "NOT SET"),
        },
        "auth": {},
        "tracing": {},
        "mlflow": {},
    }

    # ── Auth diagnostics ──
    try:
        from core.auth import secrets_loader
        from core.config import settings

        ws_client = secrets_loader._get_client()
        info["auth"]["workspace_client_initialized"] = ws_client is not None
        info["auth"]["workspace_client_init_failed"] = secrets_loader._init_failed
        info["auth"]["secrets_scope"] = settings.SECRETS_SCOPE
        info["auth"]["secrets_cache_keys"] = list(secrets_loader._cache.keys())

        if ws_client:
            try:
                test_val = secrets_loader.get_secret("default-sp-client-id")
                info["auth"]["default_sp_client_id_loaded"] = bool(test_val)
                info["auth"]["default_sp_client_id_len"] = len(test_val) if test_val else 0
            except Exception as e:
                info["auth"]["default_sp_client_id_error"] = str(e)

            try:
                test_val = secrets_loader.get_secret("default-sp-client-secret")
                info["auth"]["default_sp_client_secret_loaded"] = bool(test_val)
                info["auth"]["default_sp_client_secret_len"] = len(test_val) if test_val else 0
            except Exception as e:
                info["auth"]["default_sp_client_secret_error"] = str(e)

            try:
                from core.auth import token_provider
                cid = secrets_loader.get_secret("default-sp-client-id")
                csecret = secrets_loader.get_secret("default-sp-client-secret")
                if cid and csecret:
                    token = token_provider.get_token(cid, csecret)
                    info["auth"]["token_obtained"] = bool(token)
                    info["auth"]["token_len"] = len(token) if token else 0
                else:
                    info["auth"]["token_obtained"] = False
                    info["auth"]["token_reason"] = "no credentials loaded"
            except Exception as e:
                info["auth"]["token_error"] = str(e)
    except Exception as e:
        info["auth"]["error"] = str(e)

    # ── Tracing diagnostics ──
    try:
        from core.tracing import trace_logger
        info["tracing"]["sql_client_initialized"] = trace_logger._sql_client is not None
        info["tracing"]["sql_warned"] = trace_logger._sql_warned
        info["tracing"]["table"] = trace_logger._table

        client = trace_logger._get_sql_client()
        info["tracing"]["client_after_init"] = client is not None
        if client:
            info["tracing"]["client_type"] = str(type(client))
    except Exception as e:
        info["tracing"]["error"] = str(e)

    # ── MLflow diagnostics ──
    try:
        import mlflow
        info["mlflow"]["tracking_uri"] = mlflow.get_tracking_uri()
        exp = mlflow.get_experiment(os.environ.get("MLFLOW_EXPERIMENT_ID", "0"))
        if exp:
            info["mlflow"]["experiment_name"] = exp.name
            info["mlflow"]["experiment_id"] = exp.experiment_id
    except Exception as e:
        info["mlflow"]["error"] = str(e)

    return jsonify(info)
