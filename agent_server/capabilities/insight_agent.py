"""Insight agent — ReAct subgraph for campaign analytics.

Uses langchain.agents.create_agent (the current location for what was
previously langgraph.prebuilt.create_react_agent in LangChain/LangGraph
1.x). The subgraph runs a ReAct loop: LLM reasons → calls tools → analyzes
results → decides if more tools needed → repeats until done or the hard
recursion_limit is hit.

Called as a regular node from the parent graph via insight_agent_node.
The parent passes current_task via AgentState; SP token comes from
config["configurable"]["sp_token"].
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from capabilities.registry import get_capability_config
from core.config import settings
from core.state import AgentState
from tools.registry import get_tools, set_current_sp_token

logger = logging.getLogger(__name__)


def _recursion_limit_for(max_iterations: int) -> int:
    """Convert max ReAct iterations (tool calls) into a LangGraph recursion_limit.

    Each ReAct iteration visits 2 nodes (agent → tools). After the final tool
    call, the agent node runs once more to produce the final answer. So the
    total node visits are exactly 2*N + 1 — no slack. With max_iterations=3,
    the agent gets at most 3 tool calls and 1 final answer turn, then stops.
    """
    return 2 * max_iterations + 1


def insight_agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Wrapper node that invokes a ReAct subgraph for one plan step.

    - Reads SP token from config["configurable"]["sp_token"]
    - Injects the token into the shared tools module (sequential-safe)
    - Builds a ChatOpenAI LLM authenticated with the SP token
    - Calls create_react_agent with a hard recursion_limit derived from
      the capability's configured max_iterations
    - Returns one step_result dict (accumulated via operator.add reducer)
    """
    request_id = state.get("request_id", "unknown")
    current_task = state.get("current_task", "")
    question = state.get("rewritten_question") or state.get("original_question", "")
    sp_token = config["configurable"]["sp_token"]
    prior_llm_calls = state.get("llm_call_count", 0)
    step_index = state.get("current_step_index", 0)

    task = current_task if current_task else question

    logger.info(
        "INSIGHT_AGENT.node: START | step=%d | task='%s' | request_id=%s",
        step_index, task[:120], request_id,
    )

    # Tools pull the token from a module-level holder — this is safe because
    # LangGraph nodes run sequentially within a single graph invocation.
    set_current_sp_token(sp_token)

    cap_config = get_capability_config("insight_agent")
    max_iterations = cap_config["max_iterations"]
    recursion_limit = _recursion_limit_for(max_iterations)

    # Build system prompt. Use .replace() (not .format()) so a task text
    # containing curly braces does not crash the prompt render. Task itself
    # is NOT injected into the system prompt — it is passed as a HumanMessage
    # so it lives in state.messages where the ReAct loop expects it.
    system_prompt_text = cap_config["system_prompt"].replace(
        "{max_iterations}", str(max_iterations)
    )

    # Per-request LLM — api_key must be the real SP token, so caching is
    # not possible without a deeper refactor to config-based auth injection.
    llm = ChatOpenAI(
        model=settings.LLM_ENDPOINT_NAME,
        base_url=settings.AI_GATEWAY_URL,
        temperature=0.1,
        api_key=sp_token,
    )

    tools = get_tools(cap_config["tools"])

    final_text = ""
    tool_calls_made: list = []
    hit_recursion_limit = False

    try:
        with mlflow.start_span(name=f"insight_agent.step_{step_index}") as span:
            span.set_attributes({
                "request_id": request_id,
                "step_index": step_index,
                "task": task[:200],
                "max_iterations": max_iterations,
                "recursion_limit": recursion_limit,
            })

            subgraph = create_agent(
                model=llm,
                tools=tools,
                system_prompt=SystemMessage(content=system_prompt_text),
                name="insight_agent",
            )

            # Hard-cap iterations at the invocation level. LangGraph's default
            # recursion_limit is 25, which lets a stuck ReAct loop burn ~12
            # tool calls before raising — we want a tighter leash.
            subgraph_config: RunnableConfig = {
                **config,
                "recursion_limit": recursion_limit,
            }

            try:
                result = subgraph.invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config=subgraph_config,
                )
            except GraphRecursionError:
                hit_recursion_limit = True
                logger.warning(
                    "INSIGHT_AGENT.node: recursion_limit=%d hit | step=%d | request_id=%s",
                    recursion_limit, step_index, request_id,
                )
                result = {"messages": []}

            messages = result.get("messages", [])

            # Final answer = last non-empty AI message content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            # Count tool calls for tracing
            for msg in messages:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    tool_calls_made.extend(msg.tool_calls)

            iterations = len(tool_calls_made)

            if hit_recursion_limit and not final_text:
                final_text = (
                    f"Analysis for this step exceeded the tool-call budget "
                    f"({max_iterations} calls) without reaching a final answer. "
                    f"Partial data may be available in earlier steps."
                )

            span.set_attributes({
                "iterations": iterations,
                "result_length": len(final_text),
                "tool_calls": iterations,
                "recursion_limit_hit": hit_recursion_limit,
            })
            span.set_outputs({"result": final_text[:2000]})

            logger.info(
                "INSIGHT_AGENT.node: DONE | step=%d | iterations=%d | "
                "result=%dch | recursion_hit=%s | request_id=%s",
                step_index, iterations, len(final_text), hit_recursion_limit, request_id,
            )

    except Exception as e:
        logger.error(
            "INSIGHT_AGENT.node: failed | step=%d | request_id=%s | error=%s",
            step_index, request_id, e, exc_info=True,
        )
        final_text = f"Analysis could not be completed for this step: {str(e)[:200]}"

    step_result = {
        "step_index": step_index,
        "task": task,
        "result": final_text,
        "tool_calls_count": len(tool_calls_made),
        "recursion_limit_hit": hit_recursion_limit,
        "error": None if final_text else "No result produced",
    }

    return {
        "step_results": [step_result],
        "llm_call_count": prior_llm_calls + 1 + len(tool_calls_made),
    }
