"""AgentState + CapabilityState — data schemas for the LangGraph graph.

AgentState: main graph state (supervisor → planner → insight_agent → synthesizer).
CapabilityState: subgraph state for capability agents (e.g. insight_agent ReAct loop).

Every node reads from and writes to these states. Keep them flat and explicit.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the CoMarketer LangGraph graph."""

    # ── Input ──
    messages: Annotated[list, add_messages]  # Conversation history
    client_id: str                            # Resolved from auth
    client_name: str                          # Human-readable client name
    sp_id: str                                # Service Principal ID
    request_id: str                           # Unique per request (for tracing)

    # ── User context (from custom_inputs) ──
    user_name: str                            # Display name for personalization
    user_id: str                              # Unique user ID (LTM key)
    thread_id: str                            # Thread ID (STM checkpoint key)
    conversation_id: str                      # Session/conversation identifier
    task_type: str                            # Task type (general, analytics, etc.)

    # ── Routing (set by supervisor via Command) ──
    intent: str                               # "greeting" | "data_query" | "complex_query" | "clarification"
    original_question: str                    # Raw user question
    rewritten_question: str                   # LLM-rewritten question for Genie

    # ── Output ──
    response_text: str                        # Final response to stream back
    response_items: list                      # Structured items [{type, id, value}]
    acknowledgment_text: str                  # Acknowledgment message (pre-graph)

    # ── Memory context (injected before graph) ──
    ltm_context: str                          # Formatted LTM markdown for prompt
    conversation_history: str                 # Serialized STM history

    # ── Genie data node output (Phase 1: API + table) ──
    genie_summary: str                        # Executive summary / full analysis text
    genie_sql: str                            # SQL query used by Genie
    genie_table: str                          # Readable text for LLM prompt context
    genie_tables: list                        # 2D array dicts for UI rendering
    genie_insights: str                       # Campaign insights/analysis text
    genie_query_description: str              # Natural language description of the query
    genie_columns: list                       # Raw column metadata from Genie API
    genie_data_array: list                    # Raw data_array from Genie API
    genie_text_content: str                   # Genie's own text attachment summary
    follow_up_suggestions: list[str]          # Suggested follow-up questions

    # ── Genie analysis node output (Phase 2: LLM analysis) ──
    # genie_summary is written by genie_analysis_node (above)

    # ── Supervisor formatted output ──
    supervisor_json: Optional[dict]           # Structured {items: [...]} from format_supervisor node

    # ── Complex query planning (set by planner, consumed by synthesizer) ──
    plan: list                                # [{sub_question, purpose, capability}, ...] from planner
    plan_count: int                           # Number of sub-questions (for tracing)
    worker_results: Annotated[list, operator.add]  # Accumulated results from parallel genie_workers

    # ── Insight agent subgraph (capability-based execution) ──
    current_step_index: int                   # Index into plan["steps"] for sequential execution
    current_task: str                         # Current step's task description
    current_capability: str                   # Capability name for current step (e.g. "insight_agent")
    step_results: Annotated[list, operator.add]  # Accumulated results from sequential capability steps

    # ── Subagent envelopes (new architecture) ──
    subagent_input: Optional[dict]            # Envelope handed to CampaignInsightAgent
    subagent_output: Optional[dict]           # SubagentOutput returned to synthesizer

    # ── Metadata ──
    llm_call_count: int                       # Total LLM calls in this request
    genie_trace_id: str                       # Genie conversation/message ID (for cross-referencing)
    genie_retry_count: int                    # Retry counter for error recovery loop
    error: Optional[str]                      # Error message if something failed


class CapabilityState(TypedDict):
    """State schema for capability subgraphs (e.g. insight_agent ReAct loop).

    Overlapping keys (messages, user_query, cid, current_task) allow
    LangGraph to pass data between parent and subgraph automatically.
    """

    # ── Inputs (from parent graph) ──
    user_query: str                           # The question/task for this capability
    cid: str                                  # Client ID for Genie RLS
    current_task: str                         # Task description from planner step

    # ── ReAct loop state ──
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Agent conversation
    iterations_used: int                      # Track ReAct loop iterations

    # ── Outputs (back to parent graph) ──
    result: str                               # Final text result from capability
    tool_calls: List[Dict[str, Any]]          # Tool calls made during execution
    confidence: float                         # Confidence score (0.0–1.0)
    error: Optional[str]                      # Error message if something failed


def init_agent_state(
    user_query: str,
    cid: str,
    user_id: str,
    session_id: str,
) -> Dict[str, Any]:
    """Create initial AgentState dict for testing or programmatic invocation."""
    return {
        "messages": [],
        "client_id": cid,
        "client_name": "",
        "sp_id": "",
        "request_id": "",
        "user_name": "User",
        "user_id": user_id,
        "thread_id": session_id,
        "conversation_id": session_id,
        "task_type": "general",
        "intent": "",
        "original_question": user_query,
        "rewritten_question": "",
        "response_text": "",
        "response_items": [],
        "acknowledgment_text": "",
        "ltm_context": "",
        "conversation_history": "",
        "genie_summary": "",
        "genie_sql": "",
        "genie_table": "",
        "genie_tables": [],
        "genie_insights": "",
        "genie_query_description": "",
        "genie_columns": [],
        "genie_data_array": [],
        "genie_text_content": "",
        "follow_up_suggestions": [],
        "supervisor_json": None,
        "plan": [],
        "plan_count": 0,
        "worker_results": [],
        "current_step_index": 0,
        "current_task": "",
        "current_capability": "insight_agent",
        "step_results": [],
        "subagent_input": None,
        "subagent_output": None,
        "llm_call_count": 0,
        "genie_trace_id": "",
        "genie_retry_count": 0,
        "error": None,
    }


def init_capability_state(
    user_query: str,
    cid: str,
    current_task: str,
) -> Dict[str, Any]:
    """Create initial CapabilityState dict for testing or programmatic invocation."""
    return {
        "user_query": user_query,
        "cid": cid,
        "current_task": current_task,
        "messages": [],
        "iterations_used": 0,
        "result": "",
        "tool_calls": [],
        "confidence": 0.0,
        "error": None,
    }
