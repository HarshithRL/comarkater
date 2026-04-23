"""Cached StateGraph — new supervisor + campaign insight subagent topology.

Flow:
  START → supervisor_classify ─Command→ greeting ──────────────────────→ END
                              ─Command→ clarification ──────────────────→ END
                              ─Command→ out_of_scope ───────────────────→ END
                              ─Command→ campaign_insight_agent
                                          → supervisor_synthesize → END

The new supervisor returns ``Command(goto=...)`` to dispatch one of four
targets. Campaign-insight analysis flows through the new ``CampaignInsightAgent``
(5-phase orchestrator) and then through ``SupervisorSynthesizer`` which
assembles the final ``response_items`` for the UI.

Legacy nodes (genie_data, genie_analysis, planner, insight_agent, synthesizer,
genie_worker) remain importable in the codebase but are NOT wired into the
compiled graph.

The compiled graph is cached process-wide; SP token is injected per request
via ``config["configurable"]["sp_token"]``.
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from core.config import settings
from core.state import AgentState
from supervisor.supervisor_node import supervisor_node
from supervisor.synthesizer import SupervisorSynthesizer
from supervisor.domain_context import SupervisorDomainContext
from agents.campaign_insight.agent import CampaignInsightAgent
from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge
from agents.greeting import greeting_node
from agents.clarification import clarification_node

logger = logging.getLogger(__name__)

# ── Module-level cache ──
_compiled_graph: Optional[CompiledStateGraph] = None
_compiled_checkpointer: object = None
_graph_lock = threading.Lock()

_campaign_agent: Optional[CampaignInsightAgent] = None
_domain_knowledge: Optional[InsightAgentDomainKnowledge] = None
_domain_context: Optional[SupervisorDomainContext] = None

_OOS_FALLBACK_TEXT = (
    "I can't answer that because it's outside the campaign analytics scope. "
    "I can help with campaign performance, trends, comparisons, and diagnostics "
    "across Email, SMS, WhatsApp, APN, and BPN channels."
)


def _get_domain_knowledge() -> InsightAgentDomainKnowledge:
    global _domain_knowledge
    if _domain_knowledge is None:
        path = Path(__file__).resolve().parent.parent / "agents" / "campaign_insight" / "domain_knowledge"
        _domain_knowledge = InsightAgentDomainKnowledge(path)
    return _domain_knowledge


def _get_domain_context() -> SupervisorDomainContext:
    global _domain_context
    if _domain_context is None:
        _domain_context = SupervisorDomainContext(_get_domain_knowledge())
    return _domain_context


def _get_campaign_agent() -> CampaignInsightAgent:
    global _campaign_agent
    if _campaign_agent is None:
        default_kb = Path(__file__).resolve().parent.parent / "agents" / "campaign_insight" / "domain_knowledge"
        _campaign_agent = CampaignInsightAgent({
            "genie_space_id": settings.GENIE_SPACE_ID,
            "domain_knowledge_path": default_kb,
            "model_name": settings.LLM_ENDPOINT_NAME,
            "feature_flags": {
                "ENABLE_AUDIENCE_ANALYSIS": True,
                "ENABLE_CONTENT_ANALYSIS": True,
            },
        })
    return _campaign_agent


# ── Campaign insight agent node (async — wraps CampaignInsightAgent.run) ──

async def campaign_insight_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Run the 5-phase Campaign Insight Agent and return a state update."""
    request_id = state.get("request_id", "unknown")
    try:
        writer = get_stream_writer()
        writer({
            "event_type": "node_started",
            "node": "campaign_insight_agent",
            "message": "Analyzing your campaign data...",
        })
    except Exception:
        pass

    agent = _get_campaign_agent()
    try:
        return await agent.run(state, config)
    except Exception as exc:
        logger.exception(
            "campaign_insight_agent_node failed | request_id=%s | error=%s",
            request_id, exc,
        )
        return {"subagent_output": None, "error": f"campaign_insight_agent failed: {exc}"}


# ── Out-of-scope node (no LLM, canned response) ──

def out_of_scope_node(state: AgentState) -> dict:
    """Return a canned out-of-scope response with an alternative."""
    request_id = state.get("request_id", "unknown")
    text = state.get("response_text") or _OOS_FALLBACK_TEXT

    logger.info("OUT_OF_SCOPE: request_id=%s", request_id)

    items = [{
        "type": "text",
        "id": str(uuid.uuid4()),
        "value": text,
        "hidden": False,
        "name": "out_of_scope",
    }]
    try:
        writer = get_stream_writer()
        writer({"event_type": "node_complete", "node": "out_of_scope"})
    except Exception:
        pass

    return {
        "messages": [AIMessage(content=text)],
        "response_text": text,
        "response_items": items,
        "supervisor_json": {"items": items},
    }


# ── Synthesizer node (composes final response_items from subagent_output) ──

def supervisor_synthesize_node(state: AgentState, config: RunnableConfig) -> dict:
    """Assemble the final response using :class:`SupervisorSynthesizer`."""
    request_id = state.get("request_id", "unknown")
    subagent_output = state.get("subagent_output")
    prior_llm_calls = state.get("llm_call_count", 0)

    try:
        writer = get_stream_writer()
        writer({
            "event_type": "node_started",
            "node": "supervisor_synthesize",
            "message": "Composing final response...",
        })
    except Exception:
        writer = None

    if not subagent_output:
        fallback_text = "The analysis could not be completed. Please rephrase and try again."
        items = [{
            "type": "text",
            "id": str(uuid.uuid4()),
            "value": fallback_text,
            "hidden": False,
            "name": "analysis",
        }]
        return {
            "messages": [AIMessage(content=fallback_text)],
            "response_text": fallback_text,
            "response_items": items,
            "supervisor_json": {"items": items},
            "llm_call_count": prior_llm_calls,
        }

    sp_token = config["configurable"]["sp_token"]
    llm = ChatOpenAI(
        model=settings.LLM_ENDPOINT_NAME,
        api_key=sp_token,
        base_url=settings.AI_GATEWAY_URL,
        temperature=0.0,
    )
    synthesizer = SupervisorSynthesizer(llm, _get_domain_context())

    try:
        items = synthesizer.synthesize(subagent_output)
    except Exception as exc:
        logger.exception("supervisor_synthesize failed: %s", exc)
        items = [{
            "type": "text",
            "id": str(uuid.uuid4()),
            "value": "Synthesis failed. Please retry.",
            "hidden": False,
            "name": "analysis",
        }]

    # Short summary for STM history
    summary_for_history = ""
    for item in items:
        if item.get("type") == "text" and item.get("value"):
            summary_for_history = str(item["value"])[:300]
            break
    if not summary_for_history:
        summary_for_history = "Analysis complete."

    logger.info(
        "SUPERVISOR_SYNTHESIZE: request_id=%s items=%d",
        request_id, len(items),
    )
    if writer is not None:
        try:
            writer({"event_type": "node_complete", "node": "supervisor_synthesize", "item_count": len(items)})
        except Exception:
            pass

    return {
        "messages": [AIMessage(content=summary_for_history)],
        "response_items": items,
        "supervisor_json": {"items": items},
        "llm_call_count": prior_llm_calls + 1,
        # Clear large transient fields to keep checkpoints small
        "genie_data_array": [],
        "genie_columns": [],
        "genie_table": "",
        "genie_text_content": "",
    }


# ── Compiled graph builder ──

def get_compiled_graph(checkpointer=None) -> CompiledStateGraph:
    """Return a compiled graph, building it only on first call.

    Args:
        checkpointer: Optional CheckpointSaver for STM.

    Returns:
        Compiled StateGraph, reused across all requests.
    """
    global _compiled_graph, _compiled_checkpointer

    if _compiled_graph is not None:
        if checkpointer is not None and checkpointer is not _compiled_checkpointer:
            raise RuntimeError(
                "GRAPH: get_compiled_graph called with a checkpointer that differs from "
                "the one baked into the cached graph. Compile-time coupling must not be "
                "broken — recompilation with a different saver is not supported."
            )
        return _compiled_graph

    with _graph_lock:
        if _compiled_graph is not None:
            if checkpointer is not None and checkpointer is not _compiled_checkpointer:
                raise RuntimeError(
                    "GRAPH: get_compiled_graph called with a checkpointer that differs from "
                    "the one baked into the cached graph."
                )
            return _compiled_graph

        graph = StateGraph(AgentState)

        graph.add_node("supervisor_classify", supervisor_node)
        graph.add_node("greeting", greeting_node)
        graph.add_node("clarification", clarification_node)
        graph.add_node("out_of_scope", out_of_scope_node)
        graph.add_node("campaign_insight_agent", campaign_insight_agent_node)
        graph.add_node("supervisor_synthesize", supervisor_synthesize_node)

        graph.add_edge(START, "supervisor_classify")
        # supervisor_classify → greeting | clarification | out_of_scope | campaign_insight_agent
        # via Command(goto=...)
        graph.add_edge("greeting", END)
        graph.add_edge("clarification", END)
        graph.add_edge("out_of_scope", END)
        graph.add_edge("campaign_insight_agent", "supervisor_synthesize")
        graph.add_edge("supervisor_synthesize", END)

        compile_kwargs = {}
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer

        _compiled_graph = graph.compile(**compile_kwargs)
        _compiled_checkpointer = checkpointer
        logger.info(
            "GRAPH: Compiled ONCE | START → supervisor_classify → "
            "[greeting | clarification | out_of_scope | "
            "campaign_insight_agent → supervisor_synthesize] → END"
        )

        return _compiled_graph
