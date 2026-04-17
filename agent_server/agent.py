"""Agent entry point — ResponsesAgent subclass registered via set_model().

Stage 2 rewrite:
  1. Extracts CustomInputs from request
  2. Greeting short-circuit (zero cost, no graph)
  3. Uses cached graph via get_compiled_graph(), token in config
  4. Acknowledgment system (LLM-based "working on it" before graph)
  5. Formats response from supervisor_json
  6. Backward-compatible custom_outputs with new user context fields
"""

import json
import logging
import os
import uuid

import mlflow
import mlflow.langchain
from langchain_openai import ChatOpenAI
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from langchain_core.messages import HumanMessage
import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from core.config import settings, CLIENT_REGISTRY
from core.auth import resolve_client, token_provider
from core.graph import get_compiled_graph  # noqa: E402
from core.models import CustomInputs
from core.tracing import trace_logger, TraceRecord
from supervisor.intent_classifier import _is_greeting
from agents.greeting import greeting_node
from agents.acknowledgment import create_acknowledgment
from agents.followup import generate_follow_ups
from parsers.formatters import build_custom_outputs
from parsers.filters import should_filter_message
from memory import (
    LTMManager,
    extract_entities_from_query,
    extract_insights_from_response,
    format_ltm_context,
)
from memory.context_formatter import format_greeting_context
from memory.stm_trimmer import get_trim_removals

logger = logging.getLogger(__name__)

# ── STM checkpointer (conversation history — persistent via Lakebase, fallback to InMemorySaver) ──
_stm_checkpointer = None
_stm_lock: "asyncio.Lock | None" = None


async def _get_stm_checkpointer():
    """Lazy-init STM checkpointer. Tries Lakebase AsyncCheckpointSaver, falls back to InMemorySaver.

    InMemorySaver implements both sync and async methods, so it's a safe fallback
    when called from async contexts via graph.ainvoke/astream.
    """
    global _stm_checkpointer, _stm_lock
    if _stm_checkpointer is not None:
        return _stm_checkpointer
    if _stm_lock is None:
        _stm_lock = asyncio.Lock()
    async with _stm_lock:
        if _stm_checkpointer is not None:
            return _stm_checkpointer
        try:
            from databricks_langchain import AsyncCheckpointSaver
            logger.info(
                f"STM: Attempting AsyncCheckpointSaver | "
                f"project={settings.LAKEBASE_PROJECT} branch={settings.LAKEBASE_BRANCH}"
            )
            cp = AsyncCheckpointSaver(
                project=settings.LAKEBASE_PROJECT,
                branch=settings.LAKEBASE_BRANCH,
            )
            logger.info("STM: AsyncCheckpointSaver created, calling setup()...")
            setup_result = cp.setup()
            if asyncio.iscoroutine(setup_result):
                await setup_result
            _stm_checkpointer = cp
            logger.info("STM: AsyncCheckpointSaver (Lakebase) initialized — persistent across restarts")
        except Exception as e:
            logger.error(
                f"STM: AsyncCheckpointSaver FAILED — falling back to InMemorySaver (ephemeral)\n"
                f"  project={settings.LAKEBASE_PROJECT}, branch={settings.LAKEBASE_BRANCH}\n"
                f"  error_type={type(e).__name__}\n"
                f"  error={e}",
                exc_info=True,
            )
            _stm_checkpointer = InMemorySaver()
    return _stm_checkpointer
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ── Setup MLflow experiment ──
logger.info("AGENT.startup: ── MLflow setup starting ──")
logger.info(f"AGENT.startup: MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI', 'NOT SET')}")
logger.info(f"AGENT.startup: MLFLOW_EXPERIMENT_NAME={os.environ.get('MLFLOW_EXPERIMENT_NAME', 'NOT SET')}")
logger.info(f"AGENT.startup: MLFLOW_EXPERIMENT_ID={os.environ.get('MLFLOW_EXPERIMENT_ID', 'NOT SET')}")
logger.info(f"AGENT.startup: DATABRICKS_HOST={os.environ.get('DATABRICKS_HOST', 'NOT SET')}")

try:
    mlflow.langchain.autolog()
    logger.info("AGENT.startup: mlflow.langchain.autolog() OK")
except Exception as e:
    logger.error(f"AGENT.startup: mlflow.langchain.autolog() FAILED: {e}")

EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
)
logger.info(f"AGENT.startup: Setting experiment name={EXPERIMENT_NAME}")
try:
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"AGENT.startup: Experiment set OK | id={exp.experiment_id} | name={exp.name} | artifact_location={exp.artifact_location}")
except Exception as e:
    logger.warning(f"AGENT.startup: Failed to set experiment by name: {e}")
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        try:
            exp = mlflow.set_experiment(experiment_id=experiment_id)
            logger.info(f"AGENT.startup: Experiment set by ID OK | id={exp.experiment_id} | name={exp.name}")
        except Exception as e2:
            logger.error(f"AGENT.startup: FAILED to set experiment by ID={experiment_id}: {e2}")
    else:
        logger.error("AGENT.startup: No MLFLOW_EXPERIMENT_ID fallback available — traces will NOT be stored!")

logger.info(f"AGENT.startup: mlflow.get_tracking_uri()={mlflow.get_tracking_uri()}")
logger.info("AGENT.startup: ── MLflow setup complete ──")

# ── Ensure Delta trace table exists ──
try:
    trace_logger.init_table()
except Exception as e:
    logger.warning(f"AGENT: Failed to init trace table: {e}")


def _extract_query(request: ResponsesAgentRequest) -> str:
    """Get the last user message text."""
    if request.input:
        last = request.input[-1]
        if hasattr(last, "content"):
            return str(last.content)
        if isinstance(last, dict):
            return last.get("content", "")
    return ""


def _extract_custom_inputs(request: ResponsesAgentRequest) -> CustomInputs:
    """Extract and validate CustomInputs from request.custom_inputs."""
    raw = request.custom_inputs or {}

    # Derive client_scope from sp_id via CLIENT_REGISTRY if not explicitly sent
    # Priority: explicit client_scope > client_name from registry (lowercase) > client_id
    sp_id = raw.get("sp_id", "default")
    registry_info = CLIENT_REGISTRY.get(sp_id, CLIENT_REGISTRY.get("default", {}))
    client_id = raw.get("client_id", "") or registry_info.get("client_id", "")
    client_scope = (
        raw.get("client_scope", "")
        or registry_info.get("client_name", "").lower().replace(" ", "")
        or client_id
    )

    # Build thread_id with client_scope for checkpoint isolation
    thread_id = raw.get("thread_id")
    if not thread_id:
        session_id = raw.get("conversation_id", str(uuid.uuid4()))
        if client_scope:
            thread_id = f"client-{client_scope}:session-{session_id}"
        else:
            thread_id = f"session-{session_id}"

    return CustomInputs(
        client_scope=client_scope,
        client_id=client_id,
        user_name=raw.get("user_name", "User"),
        user_id=raw.get("user_id", ""),
        conversation_id=raw.get("conversation_id", ""),
        task_type=raw.get("task_type", "general"),
        thread_id=thread_id,
        conversation_history=raw.get("conversation_history"),
        ltm_context=raw.get("ltm_context"),
    )


def _get_sp_token(client_info: dict) -> str:
    """Get SP OAuth token for the resolved client."""
    sp_client_id = client_info.get("sp_client_id")
    sp_client_secret = client_info.get("sp_client_secret")
    try:
        return token_provider.get_token(sp_client_id, sp_client_secret)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"AGENT.get_token: No token available: {e}")
        return ""


def _build_llm(sp_token: str) -> ChatOpenAI:
    """Build a ChatOpenAI LLM instance with SP token."""
    return ChatOpenAI(
        model=settings.LLM_ENDPOINT_NAME,
        api_key=sp_token,
        base_url=settings.AI_GATEWAY_URL,
        temperature=0.0,
    )


def _build_custom_outputs(
    client_info: dict,
    request_id: str,
    trace_id: str,
    intent: str,
    agent_route: str,
    custom_inputs: CustomInputs,
) -> dict:
    """Build backward-compatible custom_outputs with new user context fields."""
    return {
        # Existing (backward-compatible)
        "client_id": client_info["client_id"],
        "client_name": client_info["client_name"],
        "request_id": request_id,
        "trace_id": trace_id,
        "intent": intent,
        "agent_route": agent_route,
        # New user context
        "user_name": custom_inputs.user_name,
        "user_id": custom_inputs.user_id,
        "thread_id": custom_inputs.thread_id or "",
        "conversation_id": custom_inputs.conversation_id,
        "task_type": custom_inputs.task_type,
    }


# ── Status event labels for Genie poll statuses ──
GENIE_STATUS_LABELS = {
    "FILTERING_CONTEXT": "Understanding your question...",
    "ASKING_AI": "Formulating the query...",
    "PENDING_WAREHOUSE": "Querying the database...",
    "EXECUTING_QUERY": "Running the query...",
}


def _build_status_event(message: str, custom_inputs, agent_id: str = "VEF9O1SFFR", request_id: str = "") -> ResponsesAgentStreamEvent:
    """Build a RATIONALE SSE event (type='text', shown in collapsible thought process in UI).

    Uses RATIONALE type (thought process, like ACK) — not observation (actual content).
    agent_id follows the convention: VEF9O1SFFR for supervisor/coordinator, ZECDLGGP3J for genie/insight.
    """
    s_id = f"status_{uuid.uuid4().hex[:12]}"
    status_items = [{"type": "text", "id": str(uuid.uuid4()), "value": message}]
    status_co = build_custom_outputs(custom_inputs, agent_id, "RATIONALE")
    return ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item_id=s_id,
        output_index=0,
        item={
            "type": "message",
            "id": s_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps({"items": [status_items], "custom_outputs": status_co})}],
        },
        custom_outputs=status_co,
    )


def _build_heartbeat_event() -> ResponsesAgentStreamEvent:
    """Build a lightweight SSE keepalive event (type='HEARTBEAT', ignored by UI).

    Prevents Databricks App reverse proxy from killing idle SSE connections (~60s timeout).
    """
    hb_id = f"hb_{uuid.uuid4().hex[:12]}"
    hb_co = {"type": "HEARTBEAT"}
    return ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item_id=hb_id,
        output_index=0,
        item={
            "type": "message",
            "id": hb_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps({"items": [], "custom_outputs": hb_co})}],
        },
        custom_outputs=hb_co,
    )


class CoMarketerAgent(ResponsesAgent):
    """ResponsesAgent subclass — routes /invocations to predict() or predict_stream()."""

    def __init__(self):
        super().__init__()
        # LTM manager (initialized once, reused across all requests)
        self.ltm = LTMManager()

    @mlflow.trace(name="comarketer_invoke")
    async def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming handler. Resolves auth, runs graph, returns response."""
        request_id = str(uuid.uuid4())
        logger.info(f"AGENT.invoke: ── REQUEST START | request_id={request_id} ──")
        logger.info(f"AGENT.invoke: tracking_uri={mlflow.get_tracking_uri()} | experiment={mlflow.get_experiment(mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id).name if mlflow.get_experiment_by_name(EXPERIMENT_NAME) else 'NOT FOUND'}")

        # Check if @mlflow.trace created an active span
        active_span = mlflow.get_current_active_span()
        logger.info(f"AGENT.invoke: active_span={active_span} | span_id={getattr(active_span, 'span_id', None)}")

        client_info = resolve_client(request)
        user_query = _extract_query(request)
        custom_inputs = _extract_custom_inputs(request)
        logger.info(f"AGENT.invoke: client={client_info.get('client_name')} | sp_id={client_info.get('sp_id')} | query={user_query[:80]}")

        sp_token = _get_sp_token(client_info)
        logger.info(f"AGENT.invoke: sp_token={'present' if sp_token else 'MISSING'} | request_id={request_id}")

        # Tag the trace early — client identity + context
        # NOTE: ONLY tags={} is safe on this Databricks MLflow version.
        # BROKEN params (corrupt/drop tags): session_id=, metadata=, client_request_id=,
        # request_preview=, response_preview= — all cause partial or full trace data loss.
        logger.info(f"AGENT.invoke: calling mlflow.update_current_trace() | request_id={request_id}")
        try:
            mlflow.update_current_trace(
                tags={
                    "client_id": client_info["client_id"],
                    "client_name": client_info["client_name"],
                    "sp_id": client_info["sp_id"],
                    "sp_mode": "per_client" if client_info.get("sp_client_id") else "none",
                    "request_id": request_id,
                    "user_name": custom_inputs.user_name or "unknown",
                    "user_id": custom_inputs.user_id or "unknown",
                    "task_type": custom_inputs.task_type or "general",
                    "conversation_id": custom_inputs.conversation_id or request_id,
                    "environment": settings.ENV,
                    "user_query_preview": user_query[:60],
                    "user_question": user_query[:250],
                    "session_id": (custom_inputs.conversation_id or request_id)[:250],
                },
            )
            logger.info(f"AGENT.invoke: update_current_trace() early tags OK | request_id={request_id}")
        except Exception as e:
            logger.error(f"AGENT.invoke: update_current_trace() FAILED: {e} | request_id={request_id}", exc_info=True)

        if settings.REQUIRE_PER_REQUEST_SP and not sp_token:
            return ResponsesAgentResponse(
                output=[{
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Authentication required. Please provide SP credentials."}],
                }],
                custom_outputs={"error": "missing_sp_credentials", "request_id": request_id},
            )

        trace_id = trace_logger.start_trace(request_id)
        record = TraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            client_id=client_info["client_id"],
            sp_id=client_info["sp_id"],
            user_query=user_query[:500],
            conversation_id=custom_inputs.conversation_id or request_id,
            environment=settings.ENV,
        )

        client_scope = custom_inputs.client_scope
        is_greeting = _is_greeting(user_query)

        result = None  # Graph result — set if data_query path runs
        try:
            logger.info(f"AGENT.invoke: entering graph/greeting logic | request_id={request_id}")

            # Greeting short-circuit (with LTM personalization)
            if is_greeting:
                logger.info(f"AGENT.invoke: GREETING detected | request_id={request_id}")

                greeting_parts = [f"Hi {custom_inputs.user_name}! \U0001f44b"]

                # Personalize from LTM profile if client_scope is present
                if client_scope:
                    try:
                        ltm_profile = await self.ltm.get_client_profile(client_scope)
                        greeting_ctx = format_greeting_context(ltm_profile)

                        if greeting_ctx["top_channels"]:
                            channels_str = " & ".join(greeting_ctx["top_channels"])
                            greeting_parts.append(
                                f"Pick up where you left off? Your recent focus: {channels_str}."
                            )
                        if greeting_ctx["recent_query"]:
                            greeting_parts.append(
                                f'Last time: "{greeting_ctx["recent_query"][:80]}..."'
                            )
                    except Exception:
                        pass  # Fall through to generic greeting

                if len(greeting_parts) == 1:
                    # No LTM data — use generic greeting
                    greeting_parts.append(
                        "What would you like to explore today? You can ask me about "
                        "campaign performance, channel comparisons, revenue trends, "
                        "engagement metrics, or conversion analysis across any time period."
                    )

                greeting_text = "\n\n".join(greeting_parts)

                record.intent = "greeting"
                record.agent_route = "greeting"
                record.response_text = greeting_text
                record.status = "success"

                custom_outputs = _build_custom_outputs(
                    client_info, request_id, trace_id, "greeting", "greeting", custom_inputs
                )
                trace_logger.end_trace(record)

                return ResponsesAgentResponse(
                    output=[{
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:12]}",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": greeting_text}],
                    }],
                    custom_outputs=custom_outputs,
                )

            # ═══ LTM READ (before graph invoke) ═══
            if client_scope:
                try:
                    with mlflow.start_span(name="ltm_gated_read") as ltm_span:
                        ltm_profile = await self.ltm.get_client_profile(client_scope)

                        ltm_episodes = []
                        query_lower = user_query.lower()
                        needs_episodes = any(kw in query_lower for kw in [
                            "improve", "optimize", "reduce", "increase", "boost",
                            "compare", "trend", "breakdown", "comprehensive",
                            "analyze", "analysis", "recommend", "suggestion",
                        ])
                        if needs_episodes:
                            ltm_episodes = await self.ltm.search_past_episodes(client_scope, user_query)

                        ltm_text = format_ltm_context(ltm_profile, ltm_episodes)
                        if ltm_text:
                            custom_inputs.ltm_context = ltm_text

                        ltm_span.set_attributes({
                            "ltm.client_scope": client_scope,
                            "ltm.profile_loaded": bool(ltm_profile.get("total_queries")),
                            "ltm.episodes_loaded": len(ltm_episodes),
                            "ltm.context_length": len(ltm_text) if ltm_text else 0,
                        })
                        logger.info(
                            f"LTM READ: scope={client_scope} "
                            f"profile={'yes' if ltm_profile.get('total_queries') else 'new'} "
                            f"episodes={len(ltm_episodes)}"
                        )
                except Exception as e:
                    logger.error(f"LTM READ ERROR (non-fatal): {e}")

            # Data query — cached graph, token in config (never in state)
            with mlflow.start_span(name="graph_invoke") as span:
                span.set_attributes({
                    "request_id": request_id,
                    "client_id": client_info["client_id"],
                    "user_query": user_query[:200],
                    "has_sp_token": bool(sp_token),
                    "user_name": custom_inputs.user_name,
                })

                graph = get_compiled_graph(checkpointer=await _get_stm_checkpointer())

                thread_id = custom_inputs.thread_id or custom_inputs.conversation_id or request_id
                invoke_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "sp_token": sp_token,
                        "sp_identity": client_info.get("sp_id", "default"),
                    },
                }

                # STM: load checkpoint + trim old messages
                existing_msgs = []
                trim_ops = []
                with mlflow.start_span(name="stm_checkpoint_load") as stm_span:
                    stm_span.set_attributes({"stm.thread_id": thread_id})
                    try:
                        state_snapshot = await graph.aget_state(invoke_config)
                        existing_msgs = state_snapshot.values.get("messages", []) if state_snapshot and state_snapshot.values else []
                        trim_ops = get_trim_removals(existing_msgs)
                        stm_span.set_attributes({
                            "stm.status": "loaded",
                            "stm.history_messages": len(existing_msgs),
                            "stm.trimmed_count": len(trim_ops),
                        })
                    except Exception as e:
                        stm_span.set_attributes({"stm.status": "error", "stm.error": str(e)[:200]})
                        logger.warning(f"STM: checkpoint load failed (non-fatal): {e}")

                initial_state = {
                    "messages": trim_ops + [HumanMessage(content=user_query)],
                    "client_id": client_info["client_id"],
                    "client_name": client_info["display_name"],
                    "sp_id": client_info["sp_id"],
                    "request_id": request_id,
                    "conversation_id": custom_inputs.conversation_id or request_id,
                    "original_question": user_query,
                    "rewritten_question": "",
                    "intent": "",
                    "response_text": "",
                    "response_items": [],
                    "genie_summary": "",
                    "genie_sql": "",
                    "genie_table": "",
                    "genie_tables": [],
                    "genie_columns": [],
                    "genie_data_array": [],
                    "genie_text_content": "",
                    "genie_insights": "",
                    "genie_query_description": "",
                    "follow_up_suggestions": [],
                    "plan": [],
                    "plan_count": 0,
                    "worker_results": [],
                    "llm_call_count": 0,
                    "genie_trace_id": "",
                    "genie_retry_count": 0,
                    "error": None,
                }

                logger.info(f"AGENT.invoke: calling graph.ainvoke() | request_id={request_id} | thread_id={thread_id} | history_msgs={len(existing_msgs)}")
                result = await graph.ainvoke(initial_state, config=invoke_config)
                logger.info(f"AGENT.invoke: graph.invoke() returned | intent={result.get('intent')} | has_response={bool(result.get('response_text'))} | error={result.get('error')} | request_id={request_id}")

            # Extract response from graph state
            response_text = result.get("response_text", "")
            response_items = result.get("response_items", [])

            if not response_text:
                response_text = "No response generated. Please try rephrasing your question."

            record.intent = result.get("intent", "data_query")
            record.agent_route = result.get("intent", "data_query")
            record.response_text = response_text[:2000]
            record.llm_call_count = result.get("llm_call_count", 1)
            record.status = "error" if result.get("error") else "success"
            if result.get("error"):
                record.error_message = str(result["error"])[:500]

        except Exception as e:
            logger.error(f"AGENT.invoke: Error | request_id={request_id} | error={e}", exc_info=True)
            error_detail = f"{type(e).__name__}: {e}"
            response_text = f"I encountered an error. Please try again.\n\nDebug: {error_detail[:300]}"
            response_items = []
            record.status = "error"
            record.error_message = str(e)[:500]

        trace_logger.end_trace(record)

        # Update trace with results learned during execution
        try:
            result_tags = {
                "intent": record.intent or "unknown",
                "agent_route": record.agent_route or "unknown",
                "status": record.status or "unknown",
                "llm_call_count": str(record.llm_call_count or 0),
                "latency_ms": str(record.latency_ms or 0),
                "conversation_id": custom_inputs.conversation_id or request_id,
            }

            # Genie agent output (only for data_query requests)
            if result and result.get("intent") == "data_query":
                genie_trace_id = result.get("genie_trace_id", "")
                if genie_trace_id:
                    result_tags["genie_trace_id"] = str(genie_trace_id)
                # NOTE: MLflow hard limit is 255 bytes per tag value — truncate all to 250
                rewritten_q = result.get("rewritten_question", "")
                if rewritten_q:
                    result_tags["genie_rewritten_question"] = str(rewritten_q)[:250]
                genie_summary = result.get("genie_summary", "")
                if genie_summary:
                    result_tags["genie_summary"] = str(genie_summary)[:250]
                genie_sql = result.get("genie_sql", "")
                if genie_sql:
                    result_tags["genie_sql"] = str(genie_sql)[:250]
                genie_query_desc = result.get("genie_query_description", "")
                if genie_query_desc:
                    result_tags["genie_query_description"] = str(genie_query_desc)[:250]
                follow_ups = result.get("follow_up_suggestions", [])
                if follow_ups:
                    result_tags["genie_follow_ups"] = json.dumps(follow_ups)[:250]
                if result.get("genie_table"):
                    result_tags["genie_has_table"] = "true"
                if result.get("genie_insights"):
                    result_tags["genie_has_insights"] = "true"
                if result.get("error"):
                    result_tags["genie_error"] = str(result["error"])[:250]

            result_tags["response_preview"] = (response_text or "")[:250]
            logger.info(f"AGENT.invoke: calling update_current_trace() result tags | count={len(result_tags)} | request_id={request_id}")
            logger.info(f"AGENT.invoke: result_tags keys={list(result_tags.keys())}")
            active_span_after = mlflow.get_current_active_span()
            logger.info(f"AGENT.invoke: active_span at result tagging={active_span_after}")
            mlflow.update_current_trace(tags=result_tags)
            logger.info(f"AGENT.invoke: update_current_trace() result tags OK | request_id={request_id}")
        except Exception as e:
            logger.error(f"AGENT.invoke: update_current_trace() result FAILED: {e} | request_id={request_id}", exc_info=True)

        custom_outputs = _build_custom_outputs(
            client_info, request_id, trace_id,
            record.intent or "data_query", record.agent_route or "supervisor",
            custom_inputs,
        )

        if record.error_message:
            custom_outputs["error"] = record.error_message

        if response_items:
            custom_outputs["response_items"] = response_items

        # ═══ LTM Write (after graph, before return) ═══
        if client_scope and not is_greeting and user_query:
            try:
                with mlflow.start_span(name="ltm_post_invoke_write") as span:
                    entities = extract_entities_from_query(user_query)

                    profile_updated = await self.ltm.update_client_profile(
                        client_scope=client_scope,
                        entities=entities,
                        user_query=user_query,
                    )

                    episodes_saved = 0
                    if result:
                        last_valid_content = ""
                        sup_json = result.get("supervisor_json")
                        if sup_json:
                            try:
                                last_valid_content = json.dumps(sup_json)
                            except (TypeError, ValueError):
                                pass

                        if last_valid_content:
                            insights = extract_insights_from_response(last_valid_content)
                            for insight in insights[:2]:
                                saved = await self.ltm.save_episode(
                                    client_scope=client_scope,
                                    query=user_query,
                                    agent_route=result.get("intent", "insight_agent"),
                                    finding=insight.get("finding", ""),
                                    channels_involved=entities.get("channels", []),
                                    metrics_involved=entities.get("metrics", []),
                                    success=True,
                                )
                                if saved:
                                    episodes_saved += 1

                    span.set_attributes({
                        "ltm.client_scope": client_scope,
                        "ltm.profile_updated": profile_updated,
                        "ltm.episodes_saved": episodes_saved,
                        "ltm.channels_extracted": len(entities.get("channels", [])),
                        "ltm.metrics_extracted": len(entities.get("metrics", [])),
                    })
                    logger.info(
                        f"LTM WRITE: scope={client_scope} "
                        f"profile={'ok' if profile_updated else 'fail'} "
                        f"episodes={episodes_saved}"
                    )
            except Exception as e:
                logger.error(f"LTM WRITE ERROR (non-fatal): {e}")

        return ResponsesAgentResponse(
            output=[{
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_text}],
            }],
            custom_outputs=custom_outputs,
        )

    async def predict_stream(self, request: ResponsesAgentRequest):
        """Streaming handler — yields structured SSE events with MLflow tracing.

        No manual @mlflow.trace needed — ResponsesAgent.__init_subclass__ auto-applies
        mlflow.trace(span_type=AGENT, output_reducer=...) to predict_stream.

        Yields:
          1. Acknowledgment event (before graph, RATIONALE custom_outputs)
          2. Response event (after graph, observation custom_outputs, {items:[...]} JSON body)
        """
        request_id = str(uuid.uuid4())
        logger.info(f"AGENT.stream: ── REQUEST START | request_id={request_id} ──")

        # Capture MLflow trace_id from the auto-applied @mlflow.trace span.
        # ResponsesAgent.__init_subclass__ wraps predict_stream with a trace span,
        # so get_current_active_span() returns that root span here.
        mlflow_trace_id = ""
        try:
            _active_span = mlflow.get_current_active_span()
            if _active_span:
                # request_id on the span IS the trace's ID in Databricks MLflow
                mlflow_trace_id = str(
                    getattr(_active_span, "request_id", "")
                    or getattr(_active_span, "_trace_id", "")
                    or ""
                )
        except Exception:
            pass
        logger.info(f"AGENT.stream: mlflow_trace_id={mlflow_trace_id!r} | request_id={request_id}")

        client_info = resolve_client(request)
        user_query = _extract_query(request)
        custom_inputs = _extract_custom_inputs(request)
        sp_token = _get_sp_token(client_info)

        logger.info(f"AGENT.stream: client={client_info.get('client_name')} | query={user_query[:80]} | request_id={request_id}")

        # ── Early MLflow tags (same keys as predict) ──
        try:
            mlflow.update_current_trace(tags={
                "client_id": client_info["client_id"],
                "client_name": client_info["client_name"],
                "sp_id": client_info["sp_id"],
                "sp_mode": "per_client" if client_info.get("sp_client_id") else "none",
                "request_id": request_id,
                "user_name": custom_inputs.user_name or "unknown",
                "user_id": custom_inputs.user_id or "unknown",
                "task_type": custom_inputs.task_type or "general",
                "conversation_id": custom_inputs.conversation_id or request_id,
                "environment": settings.ENV,
                "user_question": user_query[:250],
                "session_id": (custom_inputs.conversation_id or request_id)[:250],
                "stream": "true",
            })
        except Exception as e:
            logger.warning(f"AGENT.stream: early tags failed: {e}")

        # ── Greeting short-circuit (with LTM personalization) ──
        client_scope = custom_inputs.client_scope
        is_greeting = _is_greeting(user_query)

        if is_greeting:
            logger.info(f"AGENT.stream: GREETING detected | request_id={request_id}")

            greeting_parts = [f"Hi {custom_inputs.user_name}! \U0001f44b"]

            # Personalize from LTM profile if client_scope is present
            if client_scope:
                try:
                    ltm_profile = await self.ltm.get_client_profile(client_scope)
                    greeting_ctx = format_greeting_context(ltm_profile)

                    if greeting_ctx["top_channels"]:
                        channels_str = " & ".join(greeting_ctx["top_channels"])
                        greeting_parts.append(
                            f"Pick up where you left off? Your recent focus: {channels_str}."
                        )
                    if greeting_ctx["recent_query"]:
                        greeting_parts.append(
                            f'Last time: "{greeting_ctx["recent_query"][:80]}..."'
                        )
                except Exception:
                    pass  # Fall through to generic greeting

            if len(greeting_parts) == 1:
                # No LTM data — use generic greeting
                greeting_parts.append(
                    "What would you like to explore today? You can ask me about "
                    "campaign performance, channel comparisons, revenue trends, "
                    "engagement metrics, or conversion analysis across any time period."
                )

            greeting_text = "\n\n".join(greeting_parts)

            item_id = f"msg_{uuid.uuid4().hex[:12]}"
            greeting_items = [{"type": "text", "id": str(uuid.uuid4()), "value": greeting_text}]
            greeting_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item_id=item_id,
                output_index=0,
                item={
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps({"items": [greeting_items], "custom_outputs": greeting_co})}],
                },
                custom_outputs=greeting_co,
            )
            try:
                mlflow.update_current_trace(tags={"intent": "greeting", "status": "success"})
            except Exception:
                pass
            return

        # ── Yield acknowledgment FIRST (before graph, zero latency to user) ──
        if sp_token:
            try:
                ack_llm = _build_llm(sp_token)
                ack_result = create_acknowledgment(user_query, custom_inputs, ack_llm)
                if ack_result:
                    ack_id = f"ack_{uuid.uuid4().hex[:12]}"
                    ack_co = build_custom_outputs(custom_inputs, "VEF9O1SFFR", "RATIONALE")
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item_id=ack_id,
                        output_index=0,
                        item={
                            "type": "message",
                            "id": ack_id,
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": json.dumps({"items": [ack_result["items"]], "custom_outputs": ack_co})}],
                        },
                        custom_outputs=ack_co,
                    )
                    logger.info(f"AGENT.stream: ack yielded | request_id={request_id}")
            except Exception as e:
                logger.warning(f"AGENT.stream: ack failed: {e}")

        # ═══ LTM READ (runs after ack is visible to user) ═══
        ltm_profile = {}
        ltm_episodes = []

        if client_scope and not is_greeting:
            try:
                with mlflow.start_span(name="ltm_gated_read") as ltm_span:
                    # 1. Always load profile (fast, ~30-50ms)
                    ltm_profile = await self.ltm.get_client_profile(client_scope)

                    # 2. Search episodes ONLY for optimization/complex queries
                    query_lower = user_query.lower()
                    needs_episodes = any(kw in query_lower for kw in [
                        "improve", "optimize", "reduce", "increase", "boost",
                        "compare", "trend", "breakdown", "comprehensive",
                        "analyze", "analysis", "recommend", "suggestion",
                    ])
                    if needs_episodes:
                        ltm_episodes = await self.ltm.search_past_episodes(client_scope, user_query)

                    # 3. Format into prompt context string
                    ltm_text = format_ltm_context(ltm_profile, ltm_episodes)
                    if ltm_text:
                        custom_inputs.ltm_context = ltm_text

                    ltm_span.set_attributes({
                        "ltm.client_scope": client_scope,
                        "ltm.profile_loaded": bool(ltm_profile.get("total_queries")),
                        "ltm.episodes_loaded": len(ltm_episodes),
                        "ltm.context_length": len(ltm_text) if ltm_text else 0,
                    })
                    logger.info(
                        f"LTM READ: scope={client_scope} "
                        f"profile={'yes' if ltm_profile.get('total_queries') else 'new'} "
                        f"episodes={len(ltm_episodes)}"
                    )
            except Exception as e:
                logger.error(f"LTM READ ERROR (non-fatal): {e}")

        # ── Run graph with progressive streaming ──
        trace_id = trace_logger.start_trace(request_id)
        record = TraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            client_id=client_info["client_id"],
            sp_id=client_info["sp_id"],
            user_query=user_query[:500],
            conversation_id=custom_inputs.conversation_id or request_id,
            environment=settings.ENV,
        )
        accumulated = {}  # collect node outputs for MLflow tagging

        try:
            with mlflow.start_span(name="graph_stream") as span:
                span.set_attributes({
                    "request_id": request_id,
                    "client_id": client_info["client_id"],
                    "user_query": user_query[:200],
                    "has_sp_token": bool(sp_token),
                    "user_name": custom_inputs.user_name,
                })

                graph = get_compiled_graph(checkpointer=await _get_stm_checkpointer())

                thread_id = custom_inputs.thread_id or custom_inputs.conversation_id or request_id
                invoke_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "sp_token": sp_token,
                        "sp_identity": client_info.get("sp_id", "default"),
                    },
                }

                # STM: load checkpoint + trim old messages
                existing_msgs = []
                trim_ops = []
                with mlflow.start_span(name="stm_checkpoint_load") as stm_span:
                    stm_span.set_attributes({"stm.thread_id": thread_id})
                    try:
                        state_snapshot = await graph.aget_state(invoke_config)
                        existing_msgs = state_snapshot.values.get("messages", []) if state_snapshot and state_snapshot.values else []
                        trim_ops = get_trim_removals(existing_msgs)
                        stm_span.set_attributes({
                            "stm.status": "loaded",
                            "stm.history_messages": len(existing_msgs),
                            "stm.trimmed_count": len(trim_ops),
                        })
                    except Exception as e:
                        stm_span.set_attributes({"stm.status": "error", "stm.error": str(e)[:200]})
                        logger.warning(f"STM: checkpoint load failed (non-fatal): {e}")

                initial_state = {
                    "messages": trim_ops + [HumanMessage(content=user_query)],
                    "client_id": client_info["client_id"],
                    "client_name": client_info["display_name"],
                    "sp_id": client_info["sp_id"],
                    "request_id": request_id,
                    "conversation_id": custom_inputs.conversation_id or request_id,
                    "original_question": user_query,
                    "rewritten_question": "",
                    "intent": "",
                    "response_text": "",
                    "response_items": [],
                    "supervisor_json": None,
                    "genie_summary": "",
                    "genie_sql": "",
                    "genie_table": "",
                    "genie_tables": [],
                    "genie_columns": [],
                    "genie_data_array": [],
                    "genie_text_content": "",
                    "genie_insights": "",
                    "genie_query_description": "",
                    "follow_up_suggestions": [],
                    "plan": [],
                    "plan_count": 0,
                    "worker_results": [],
                    "llm_call_count": 0,
                    "genie_trace_id": "",
                    "genie_retry_count": 0,
                    "error": None,
                }

                logger.info(f"AGENT.stream: graph.astream() starting | request_id={request_id} | thread_id={thread_id} | history_msgs={len(existing_msgs)}")

                table_already_sent = False  # Track if table was sent via StreamWriter
                worker_tables_sent = set()  # Track which worker tables were sent via StreamWriter

                async for chunk in graph.astream(initial_state, config=invoke_config,
                                                 stream_mode=["updates", "custom"]):
                    mode, data = chunk  # tuple when multiple stream modes

                    # ── Mid-node custom events (StreamWriter) ──
                    if mode == "custom":
                        event_type = data.get("event_type", "") if isinstance(data, dict) else ""

                        # Simple path: table from genie_data StreamWriter
                        if event_type == "table_ready":
                            table_data = data.get("table")
                            if isinstance(table_data, dict) and table_data.get("data"):
                                table_items = [{"type": "table", "id": str(uuid.uuid4()), "value": table_data}]
                                t_id = f"table_{uuid.uuid4().hex[:12]}"
                                table_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=t_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": t_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [table_items], "custom_outputs": table_co})}],
                                    },
                                    custom_outputs=table_co,
                                )
                                table_already_sent = True
                                logger.info(f"AGENT.stream: TABLE emitted (mid-node via StreamWriter) | request_id={request_id}")

                        # Complex path: plan from planner StreamWriter
                        elif event_type == "plan_ready":
                            # New schema (AdaptivePlanner): data["steps"] = [{step_id, query, dim}, ...], data["budget"]
                            # Legacy schema (supervisor planner): data["plan"] = [{sub_question}, ...]
                            plan_steps = data.get("steps") or data.get("plan") or []
                            plan_count = data.get("plan_count", len(plan_steps))
                            if plan_steps:
                                def _step_label(i, p):
                                    if isinstance(p, dict):
                                        return p.get("query") or p.get("sub_question") or ""
                                    return str(p)
                                plan_text = f"I'll analyze this from {plan_count} angles:\n" + "\n".join(
                                    f"  {i+1}. {_step_label(i, p)}" for i, p in enumerate(plan_steps)
                                )
                                budget = data.get("budget")
                                if budget:
                                    plan_text += f"\n(budget: {budget})"
                                p_id = f"plan_{uuid.uuid4().hex[:12]}"
                                p_items = [{"type": "text", "id": str(uuid.uuid4()), "value": plan_text}]
                                plan_co = build_custom_outputs(custom_inputs, "VEF9O1SFFR", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=p_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": p_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [p_items], "custom_outputs": plan_co})}],
                                    },
                                    custom_outputs=plan_co,
                                )
                                logger.info(f"AGENT.stream: PLAN emitted | count={plan_count} | request_id={request_id}")

                        # Complex path: worker table from genie_worker StreamWriter
                        elif event_type == "worker_table_ready":
                            w_idx = data.get("worker_index", -1)
                            table_data = data.get("table")
                            if isinstance(table_data, dict) and table_data.get("data"):
                                wt_items = [{"type": "table", "id": str(uuid.uuid4()), "value": table_data}]
                                wt_id = f"wtable_{w_idx}_{uuid.uuid4().hex[:8]}"
                                wt_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=wt_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": wt_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [wt_items], "custom_outputs": wt_co})}],
                                    },
                                    custom_outputs=wt_co,
                                )
                                worker_tables_sent.add(w_idx)
                                logger.info(f"AGENT.stream: WORKER TABLE emitted | worker={w_idx} | request_id={request_id}")

                        # ── Transient status events (genie poll + node start) ──
                        elif event_type == "genie_status":
                            phase = data.get("phase")
                            # New boundary-phase flow (tool_handler): phase in {"start","done"}
                            if phase in ("start", "done"):
                                if phase == "start":
                                    label = data.get("message") or "Fetching data..."
                                else:
                                    rows = data.get("rows")
                                    status = data.get("status")
                                    if isinstance(rows, int):
                                        label = f"Received {rows:,} rows"
                                    elif status:
                                        label = f"Fetch {status}"
                                    else:
                                        label = "Fetch complete"
                                yield _build_status_event(label, custom_inputs, agent_id="ZECDLGGP3J", request_id=request_id)
                                logger.debug(f"AGENT.stream: STATUS emitted | genie_phase={phase} | request_id={request_id}")
                            else:
                                # Legacy numeric-status flow
                                label = GENIE_STATUS_LABELS.get(data.get("status"), None)
                                if label:
                                    w_idx = data.get("worker_index")
                                    if w_idx is not None:
                                        label = f"Worker {w_idx + 1}: {label}"
                                    yield _build_status_event(label, custom_inputs, agent_id="ZECDLGGP3J", request_id=request_id)
                                    logger.debug(f"AGENT.stream: STATUS emitted | genie={data.get('status')} | request_id={request_id}")

                        elif event_type == "phase_progress":
                            phase = data.get("phase", "")
                            if phase == "interpret":
                                n = data.get("insights")
                                label = f"Interpreting results ({n} insights)..." if isinstance(n, int) else "Interpreting results..."
                            elif phase == "recommend":
                                n = data.get("recs")
                                label = f"Drafting recommendations ({n})..." if isinstance(n, int) else "Drafting recommendations..."
                            elif phase == "reflect":
                                rewritten = data.get("rewritten")
                                label = "Reflecting (rewrite applied)..." if rewritten else "Reflecting..."
                            else:
                                label = f"{phase}..." if phase else "Processing..."
                            yield _build_status_event(label, custom_inputs, agent_id="ZECDLGGP3J", request_id=request_id)
                            logger.debug(f"AGENT.stream: STATUS emitted | phase_progress={phase} | request_id={request_id}")

                        elif event_type == "chart_ready":
                            if not data.get("skipped"):
                                ct = data.get("chart_type") or "chart"
                                yield _build_status_event(f"Chart ready ({ct})", custom_inputs, agent_id="ZECDLGGP3J", request_id=request_id)
                                logger.debug(f"AGENT.stream: STATUS emitted | chart_type={ct} | request_id={request_id}")
                            # When skipped=True, emit nothing (keeps the stream quiet for non-chartable queries)

                        elif event_type == "node_started":
                            message = data.get("message", "Processing...")
                            node = data.get("node", "")
                            # Genie/insight work → ZECDLGGP3J; supervisor/coordinator work → VEF9O1SFFR
                            aid = "ZECDLGGP3J" if node in ("genie_analysis", "synthesizer", "campaign_insight_agent") else "VEF9O1SFFR"
                            yield _build_status_event(message, custom_inputs, agent_id=aid, request_id=request_id)
                            logger.debug(f"AGENT.stream: STATUS emitted | node={node} | agent_id={aid} | request_id={request_id}")

                        continue  # Don't process custom events as node completions

                    # ── Node completion events ──
                    if mode != "updates":
                        continue

                    for node_name, node_data in data.items():
                        accumulated.update(node_data)
                        logger.info(f"AGENT.stream: node={node_name} done | request_id={request_id}")

                        # ── supervisor / supervisor_classify: routing only, no event to user ──
                        if node_name in ("supervisor", "supervisor_classify"):
                            continue

                        # ── campaign_insight_agent: subagent output is consumed by supervisor_synthesize ──
                        if node_name == "campaign_insight_agent":
                            continue

                        # ── supervisor_synthesize: yield final response_items (text/table/chart) ──
                        if node_name == "supervisor_synthesize":
                            synth_items = node_data.get("response_items") or []
                            if synth_items:
                                ss_id = f"synth_{uuid.uuid4().hex[:12]}"
                                ss_co = build_custom_outputs(custom_inputs, "VEF9O1SFFR", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=ss_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": ss_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [synth_items], "custom_outputs": ss_co})}],
                                    },
                                    custom_outputs=ss_co,
                                )
                                logger.info(f"AGENT.stream: SYNTHESIZE yielded | items={len(synth_items)} | request_id={request_id}")
                            continue

                        # ── out_of_scope: yield canned response_items ──
                        if node_name == "out_of_scope":
                            oos_items = node_data.get("response_items") or []
                            if oos_items:
                                oo_id = f"oos_{uuid.uuid4().hex[:12]}"
                                oo_co = build_custom_outputs(custom_inputs, "VEF9O1SFFR", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=oo_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": oo_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [oos_items], "custom_outputs": oo_co})}],
                                    },
                                    custom_outputs=oo_co,
                                )
                                logger.info(f"AGENT.stream: OUT_OF_SCOPE yielded | request_id={request_id}")
                            continue

                        # ── greeting: stream greeting text (2D array + custom_outputs) ──
                        if node_name == "greeting":
                            g_id = f"msg_{uuid.uuid4().hex[:12]}"
                            g_text = node_data.get("response_text", "")
                            g_items = [{"type": "text", "id": str(uuid.uuid4()), "value": g_text}]
                            g_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item_id=g_id,
                                output_index=0,
                                item={
                                    "type": "message",
                                    "id": g_id,
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": json.dumps({"items": [g_items], "custom_outputs": g_co})}],
                                },
                                custom_outputs=g_co,
                            )

                        # ── clarification: stream text answer (no table, no chart) ──
                        elif node_name == "clarification":
                            c_id = f"msg_{uuid.uuid4().hex[:12]}"
                            c_text = node_data.get("response_text", "")
                            c_items = [{"type": "text", "id": str(uuid.uuid4()), "value": c_text}]
                            c_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item_id=c_id,
                                output_index=0,
                                item={
                                    "type": "message",
                                    "id": c_id,
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": json.dumps({"items": [c_items], "custom_outputs": c_co})}],
                                },
                                custom_outputs=c_co,
                            )

                        # ── genie_data: stream table (table may already be sent via StreamWriter) ──
                        elif node_name == "genie_data":
                            data_items = []

                            # Fallback: send table here ONLY if StreamWriter didn't fire
                            if not table_already_sent:
                                genie_tables = node_data.get("genie_tables") or []
                                for table_2d in genie_tables:
                                    if isinstance(table_2d, dict) and table_2d.get("data"):
                                        data_items.append({
                                            "type": "table",
                                            "id": str(uuid.uuid4()),
                                            "value": table_2d,
                                        })

                            if data_items:
                                d_id = f"data_{uuid.uuid4().hex[:12]}"
                                data_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=d_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": d_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [data_items], "custom_outputs": data_co})}],
                                    },
                                    custom_outputs=data_co,
                                )
                                logger.info(
                                    f"AGENT.stream: TABLE yielded (fallback) | request_id={request_id}"
                                )

                        # ── genie_analysis: stream LLM analysis summary ──
                        elif node_name == "genie_analysis":
                            analysis_text = node_data.get("genie_summary", "")
                            if analysis_text:
                                summary_items = [{"type": "text", "id": str(uuid.uuid4()), "value": analysis_text}]
                                s_id = f"summary_{uuid.uuid4().hex[:12]}"
                                summary_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=s_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": s_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [summary_items], "custom_outputs": summary_co})}],
                                    },
                                    custom_outputs=summary_co,
                                )
                                logger.info(
                                    f"AGENT.stream: ANALYSIS yielded | summary={len(analysis_text)}ch | request_id={request_id}"
                                )

                        # ── planner: plan already streamed via StreamWriter, no event needed ──
                        elif node_name == "planner":
                            continue

                        # ── genie_worker: fallback table emit if StreamWriter missed ──
                        elif node_name == "genie_worker":
                            w_results = node_data.get("worker_results") or []
                            for wr in w_results:
                                w_idx = wr.get("worker_index", -1)
                                if w_idx in worker_tables_sent:
                                    continue  # Already sent via StreamWriter
                                if wr.get("error"):
                                    continue  # Skip errored workers
                                for table_2d in wr.get("genie_tables", []):
                                    if isinstance(table_2d, dict) and table_2d.get("data"):
                                        wf_items = [{"type": "table", "id": str(uuid.uuid4()), "value": table_2d}]
                                        wf_id = f"wfallback_{w_idx}_{uuid.uuid4().hex[:8]}"
                                        wf_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                        yield ResponsesAgentStreamEvent(
                                            type="response.output_item.done",
                                            item_id=wf_id,
                                            output_index=0,
                                            item={
                                                "type": "message",
                                                "id": wf_id,
                                                "role": "assistant",
                                                "content": [{"type": "output_text", "text": json.dumps({"items": [wf_items], "custom_outputs": wf_co})}],
                                            },
                                            custom_outputs=wf_co,
                                        )
                                        logger.info(f"AGENT.stream: WORKER TABLE fallback | worker={w_idx} | request_id={request_id}")

                        # ── synthesizer: stream synthesis summary text ──
                        elif node_name == "synthesizer":
                            synth_text = node_data.get("genie_summary", "")
                            if synth_text:
                                synth_items = [{"type": "text", "id": str(uuid.uuid4()), "value": synth_text}]
                                sy_id = f"synthesis_{uuid.uuid4().hex[:12]}"
                                synth_co = build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=sy_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": sy_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [synth_items], "custom_outputs": synth_co})}],
                                    },
                                    custom_outputs=synth_co,
                                )
                                logger.info(f"AGENT.stream: SYNTHESIS yielded | len={len(synth_text)}ch | request_id={request_id}")

                        # ── format_supervisor: stream recommendations + charts ──
                        elif node_name == "format_supervisor":
                            sup_json = node_data.get("supervisor_json", {})
                            all_items = sup_json.get("items", [])
                            # Filter: only text (recommendations) + chart — tables already sent earlier
                            analysis_items = [
                                it for it in all_items
                                if it.get("type") in ("text", "chart", "collapsedText")
                            ]

                            if analysis_items:
                                a_id = f"analysis_{uuid.uuid4().hex[:12]}"
                                analysis_co = build_custom_outputs(custom_inputs, "VEF9O1SFFR", "observation")
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item_id=a_id,
                                    output_index=0,
                                    item={
                                        "type": "message",
                                        "id": a_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": json.dumps({"items": [analysis_items], "custom_outputs": analysis_co})}],
                                    },
                                    custom_outputs=analysis_co,
                                )
                                logger.info(f"AGENT.stream: RECOMMENDATIONS yielded | items={len(analysis_items)} | request_id={request_id}")

                logger.info(f"AGENT.stream: graph.astream() complete | request_id={request_id}")

            record.intent = accumulated.get("intent", "data_query")
            intent_val = accumulated.get("intent", "data_query")
            route_map = {
                "data_query": "campaign_insight_agent",
                "performance_lookup": "campaign_insight_agent",
                "complex_query": "campaign_insight_agent",
                "clarification": "clarification",
                "greeting": "greeting",
                "out_of_scope": "out_of_scope",
            }
            record.agent_route = route_map.get(intent_val, "campaign_insight_agent")
            record.response_text = (accumulated.get("response_text") or accumulated.get("genie_summary", ""))[:2000]
            record.llm_call_count = accumulated.get("llm_call_count", 0)
            record.status = "error" if accumulated.get("error") else "success"
            if accumulated.get("error"):
                record.error_message = str(accumulated["error"])[:500]

        except Exception as e:
            logger.error(f"AGENT.stream: graph error | request_id={request_id} | error={e}", exc_info=True)
            record.status = "error"
            record.error_message = str(e)[:500]
            error_id = f"err_{uuid.uuid4().hex[:12]}"
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item_id=error_id,
                output_index=0,
                item={
                    "type": "message",
                    "id": error_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I encountered an error processing your request. Please try again."}],
                },
                custom_outputs=build_custom_outputs(custom_inputs, "VEF9O1SFFR", "observation"),
            )

        trace_logger.end_trace(record)

        # ── Result MLflow tags ──
        try:
            result_tags = {
                "intent": record.intent or "unknown",
                "agent_route": record.agent_route or "unknown",
                "status": record.status or "unknown",
                "llm_call_count": str(record.llm_call_count or 0),
                "latency_ms": str(record.latency_ms or 0),
            }
            if accumulated.get("intent") == "data_query":
                if accumulated.get("genie_trace_id"):
                    result_tags["genie_trace_id"] = str(accumulated["genie_trace_id"])
                rewritten_q = accumulated.get("rewritten_question", "")
                if rewritten_q:
                    result_tags["genie_rewritten_question"] = str(rewritten_q)[:250]
                if accumulated.get("genie_summary"):
                    result_tags["genie_summary"] = str(accumulated["genie_summary"])[:250]
                if accumulated.get("genie_sql"):
                    result_tags["genie_sql"] = str(accumulated["genie_sql"])[:250]
                genie_query_desc = accumulated.get("genie_query_description", "")
                if genie_query_desc:
                    result_tags["genie_query_description"] = str(genie_query_desc)[:250]
                follow_ups = accumulated.get("follow_up_suggestions", [])
                if follow_ups:
                    result_tags["genie_follow_ups"] = json.dumps(follow_ups)[:250]
                if accumulated.get("genie_table"):
                    result_tags["genie_has_table"] = "true"
                if accumulated.get("genie_insights"):
                    result_tags["genie_has_insights"] = "true"
                if accumulated.get("error"):
                    result_tags["genie_error"] = str(accumulated["error"])[:250]
            mlflow.update_current_trace(tags=result_tags)
            logger.info(f"AGENT.stream: result tags OK | request_id={request_id}")
        except Exception as e:
            logger.warning(f"AGENT.stream: result tags failed: {e}")

        # ═══ POST-YIELD: LTM Write Pipeline (zero user-facing latency) ═══
        if client_scope and not is_greeting and user_query:
            try:
                with mlflow.start_span(name="ltm_post_yield_write") as span:
                    # 1. Extract entities from user query
                    entities = extract_entities_from_query(user_query)

                    # 2. Update client profile (channel freq, metric freq, etc.)
                    profile_updated = await self.ltm.update_client_profile(
                        client_scope=client_scope,
                        entities=entities,
                        user_query=user_query,
                    )

                    # 3. Save episode if we extracted findings from the response
                    episodes_saved = 0
                    last_valid_content = ""
                    sup_json = accumulated.get("supervisor_json")
                    if sup_json:
                        try:
                            last_valid_content = json.dumps(sup_json)
                        except (TypeError, ValueError):
                            pass

                    if last_valid_content:
                        insights = extract_insights_from_response(last_valid_content)
                        for insight in insights[:2]:  # max 2 episodes per request
                            saved = await self.ltm.save_episode(
                                client_scope=client_scope,
                                query=user_query,
                                agent_route=accumulated.get("intent", "insight_agent"),
                                finding=insight.get("finding", ""),
                                channels_involved=entities.get("channels", []),
                                metrics_involved=entities.get("metrics", []),
                                success=True,
                            )
                            if saved:
                                episodes_saved += 1

                    span.set_attributes({
                        "ltm.client_scope": client_scope,
                        "ltm.profile_updated": profile_updated,
                        "ltm.episodes_saved": episodes_saved,
                        "ltm.channels_extracted": len(entities.get("channels", [])),
                        "ltm.metrics_extracted": len(entities.get("metrics", [])),
                    })
                    logger.info(
                        f"LTM WRITE: scope={client_scope} "
                        f"profile={'ok' if profile_updated else 'fail'} "
                        f"episodes={episodes_saved}"
                    )
            except Exception as e:
                logger.error(f"LTM WRITE ERROR (non-fatal): {e}")

        # ── Yield final TRACE_DONE event so UI can display feedback buttons ──
        # UI captures mlflow_trace_id from this event and attaches it to POST /ui/feedback
        if mlflow_trace_id:
            _tid = f"trace_meta_{uuid.uuid4().hex[:8]}"
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item_id=_tid,
                output_index=99,
                item={
                    "type": "message",
                    "id": _tid,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                },
                custom_outputs={"type": "TRACE_DONE", "mlflow_trace_id": mlflow_trace_id},
            )
            logger.info(f"AGENT.stream: TRACE_DONE yielded | trace_id={mlflow_trace_id} | request_id={request_id}")


from mlflow.genai.agent_server import invoke, stream

agent = CoMarketerAgent()
set_model(agent)


@invoke()
async def handle_invoke(request):
    """Registered with AgentServer for non-streaming /invocations."""
    return await agent.predict(request)


@stream()
async def handle_stream(request):
    """Async generator — consumes agent.predict_stream directly (no executor hop).

    predict_stream is now an async generator; both STM (AsyncCheckpointSaver) and
    LTM (AsyncDatabricksStore) run natively in this loop. Heartbeat runs as a
    concurrent task to keep the reverse proxy (~60s idle timeout) warm between
    slow chunks.
    """
    q: asyncio.Queue = asyncio.Queue()
    DONE = object()
    HEARTBEAT = object()

    async def run_stream():
        """Consume the agent async generator and funnel chunks through the queue."""
        try:
            async for chunk in agent.predict_stream(request):
                q.put_nowait(chunk)
        except Exception as e:
            q.put_nowait(e)
        finally:
            q.put_nowait(DONE)

    stream_task = asyncio.create_task(run_stream())

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(15)
                q.put_nowait(HEARTBEAT)
        except asyncio.CancelledError:
            pass

    hb_task = asyncio.create_task(heartbeat())

    # Yield chunks as they arrive — await q.get() releases event loop to flush SSE
    try:
        while True:
            chunk = await q.get()
            if chunk is DONE:
                break
            if chunk is HEARTBEAT:
                yield _build_heartbeat_event()
                continue
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
    finally:
        hb_task.cancel()
        if not stream_task.done():
            stream_task.cancel()
