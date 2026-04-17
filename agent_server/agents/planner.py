"""Planner node — decomposes complex queries into capability-aware steps.

Called when supervisor classifies intent as COMPLEX_QUERY.
Emits plan via StreamWriter so the user sees it before execution starts.

Flow:
  1. LLM decomposes question into 2-6 steps with capability assignments
  2. StreamWriter emits plan_ready event (user sees plan immediately)
  3. Returns plan + sets current_task/current_capability for first step
  4. Graph builder loops insight_agent through steps sequentially
"""
from __future__ import annotations

import json
import logging

import mlflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from capabilities.registry import list_capabilities
from core.config import settings
from core.state import AgentState
from prompts.planner_prompt import PLANNER_PROMPT

logger = logging.getLogger(__name__)

MAX_STEPS = 6  # Match MAX_WORKERS — Genie rate-limits at ~5-6 concurrent


def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    """Decompose complex query into capability-aware steps.

    Reads: rewritten_question, original_question, messages (for context)
    Writes: plan, plan_count, current_task, current_capability, current_step_index
    Emits: StreamWriter plan_ready event
    MLflow: planner_decompose span
    """
    request_id = state.get("request_id", "unknown")
    question = state.get("rewritten_question") or state.get("original_question", "")
    prior_llm_calls = state.get("llm_call_count", 0)
    writer = get_stream_writer()

    logger.info("PLANNER: START | request_id=%s | question=%s", request_id, question[:120])

    # Build conversation context (same pattern as supervisor)
    messages = state.get("messages", [])
    conversation_context = ""
    if len(messages) > 1:
        recent = messages[-7:-1] if len(messages) > 7 else messages[:-1]
        lines = []
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:200] if hasattr(msg, "content") else str(msg)[:200]
            lines.append(f"{role}: {content}")
        if lines:
            conversation_context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    capabilities = list_capabilities()

    sp_token = config["configurable"]["sp_token"]
    llm = ChatOpenAI(
        model=settings.LLM_ENDPOINT_NAME,
        api_key=sp_token,
        base_url=settings.AI_GATEWAY_URL,
        temperature=0.0,
    )

    try:
        with mlflow.start_span(name="planner_decompose") as span:
            span.set_attributes({
                "request_id": request_id,
                "original_question": question[:200],
                "available_capabilities": capabilities,
            })

            prompt = PLANNER_PROMPT.format(
                conversation_context=conversation_context,
                question=question,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            span.set_attributes({"raw_response": raw[:500]})

            # Parse JSON array of steps
            plan = _parse_plan(raw, question)
            plan = plan[:MAX_STEPS]

            # Ensure each step has a capability field (default: insight_agent)
            for step in plan:
                if "capability" not in step:
                    step["capability"] = "insight_agent"
                if step["capability"] not in capabilities:
                    logger.warning(
                        "PLANNER: unknown capability '%s', defaulting to insight_agent",
                        step["capability"],
                    )
                    step["capability"] = "insight_agent"

            span.set_attributes({
                "plan_count": len(plan),
                "sub_questions": json.dumps([p["sub_question"][:100] for p in plan])[:500],
            })

        logger.info(
            "PLANNER: decomposed into %d steps | request_id=%s",
            len(plan), request_id,
        )

    except Exception as e:
        logger.error("PLANNER: LLM failed, falling back to single step | request_id=%s | error=%s", request_id, e)
        plan = [{
            "sub_question": question,
            "purpose": "direct query (planner fallback)",
            "capability": "insight_agent",
        }]

    # Emit plan via StreamWriter — user sees it before execution starts
    try:
        writer({
            "event_type": "plan_ready",
            "plan": plan,
            "plan_count": len(plan),
        })
    except Exception as e:
        logger.warning("PLANNER: StreamWriter failed (non-fatal) | error=%s", e)

    # Set up first step for immediate execution
    first_step = plan[0] if plan else {"sub_question": question, "capability": "insight_agent"}

    return {
        "plan": plan,
        "plan_count": len(plan),
        "current_step_index": 0,
        "current_task": first_step.get("sub_question", question),
        "current_capability": first_step.get("capability", "insight_agent"),
        "llm_call_count": prior_llm_calls + 1,
    }


def _parse_plan(raw: str, fallback_question: str) -> list[dict]:
    """Parse LLM output into list of step dicts.

    Handles JSON wrapped in markdown code blocks. Falls back to
    single step if parsing fails.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list) or len(parsed) < 1:
            raise ValueError("Expected non-empty JSON array")

        plan = []
        for item in parsed:
            if isinstance(item, dict) and "sub_question" in item:
                plan.append({
                    "sub_question": str(item["sub_question"]).strip(),
                    "purpose": str(item.get("purpose", "")).strip(),
                    "capability": str(item["capability"]).strip() if isinstance(item.get("capability"), str) else "insight_agent",
                })
        if not plan:
            raise ValueError("No valid sub_question items found")
        return plan

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("PLANNER._parse_plan: parse failed (%s), using fallback", e)
        return [{
            "sub_question": fallback_question,
            "purpose": "direct query (parse fallback)",
            "capability": "insight_agent",
        }]
