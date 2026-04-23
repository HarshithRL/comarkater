"""New supervisor LangGraph node — does NOT replace agents/supervisor.py.

This is a parallel implementation wired to the new supervisor package. The
existing ``agents/supervisor.py`` remains the live node; integration is a
separate task.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import mlflow
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge
from core.config import settings
from core.state import AgentState
from .domain_context import SupervisorDomainContext
from .intent_classifier import IntentClassifier
from .planner import SupervisorPlanner
from .router import SupervisorRouter

logger = logging.getLogger(__name__)

_domain_knowledge: InsightAgentDomainKnowledge | None = None
_domain_context: SupervisorDomainContext | None = None


def _get_domain_knowledge() -> InsightAgentDomainKnowledge:
    global _domain_knowledge
    if _domain_knowledge is None:
        path = Path(__file__).parent.parent / "agents" / "campaign_insight" / "domain_knowledge"
        _domain_knowledge = InsightAgentDomainKnowledge(path)
    return _domain_knowledge


def _get_domain_context() -> SupervisorDomainContext:
    global _domain_context
    if _domain_context is None:
        _domain_context = SupervisorDomainContext(_get_domain_knowledge())
    return _domain_context


def supervisor_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["greeting", "clarification", "out_of_scope", "campaign_insight_agent"]]:
    """Classify, plan (if complex), and route via a LangGraph ``Command``.

    Args:
        state: The current :class:`AgentState`.
        config: LangGraph runnable config — must carry ``configurable.sp_token``.

    Returns:
        A ``Command`` with a state update and a ``goto`` target.
    """
    request_id = state.get("request_id", "unknown")
    query = state.get("original_question", "") or ""
    client_id = state.get("client_id", "")

    try:
        messages = state.get("messages", []) or []
        history: list[dict] = []
        for msg in messages[-6:]:
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            content = getattr(msg, "content", None) or str(msg)
            history.append({"role": role, "content": content})

        sp_token = config["configurable"]["sp_token"]
        llm = ChatOpenAI(
            model=settings.LLM_ENDPOINT_NAME,
            api_key=sp_token,
            base_url=settings.AI_GATEWAY_URL,
            temperature=0.0,
        )

        domain_context = _get_domain_context()

        with mlflow.start_span(name="supervisor_intent_classify") as _s:
            _s.set_inputs({"query": query[:500], "history_len": len(history)})
            classifier = IntentClassifier(llm, domain_context)
            intent = classifier.classify(query, history)
            _s.set_outputs({
                "intent_type": intent.get("intent_type", ""),
                "complexity": intent.get("complexity", ""),
                "channels_mentioned": intent.get("channels_mentioned", []),
                "requires_audience": intent.get("requires_audience", False),
                "requires_content": intent.get("requires_content", False),
            })

        with mlflow.start_span(name="supervisor_plan") as _s:
            _s.set_inputs({"query": query[:500], "intent_type": intent.get("intent_type", "")})
            planner = SupervisorPlanner(llm, domain_context)
            plan = planner.plan(query, intent)
            _s.set_outputs({"plan_steps": len(plan or []), "plan": list(plan or [])[:10]})

        with mlflow.start_span(name="supervisor_route") as _s:
            _s.set_inputs({
                "query": query[:500],
                "intent_type": intent.get("intent_type", ""),
                "plan_steps": len(plan or []),
            })
            router = SupervisorRouter()
            context = {
                "request_id": request_id,
                "client_id": client_id,
                "conversation_history": state.get("conversation_history", ""),
                "memory": state.get("ltm_context", ""),
            }
            target, update = router.route(query, intent, plan, context)
            _s.set_outputs({
                "target": target,
                "intent": update.get("intent", ""),
                "rewritten_question": update.get("rewritten_question", "")[:500],
            })

        logger.info(
            "SUPERVISOR(new): request_id=%s intent=%s complexity=%s target=%s",
            request_id,
            intent.get("intent_type"),
            intent.get("complexity"),
            target,
        )
        return Command(update=update, goto=target)
    except Exception as exc:
        logger.exception("SUPERVISOR(new) failed: %s", exc)
        return Command(
            update={
                "intent": "data_query",
                "rewritten_question": query,
                "error": f"new supervisor failed: {exc}",
            },
            goto="campaign_insight_agent",
        )
