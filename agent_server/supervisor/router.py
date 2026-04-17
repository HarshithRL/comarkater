"""Deterministic supervisor router — no LLM calls.

Turns (query, intent, plan, context) into a (target_node, state_update) pair
that the supervisor node returns via a LangGraph ``Command``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from agents.campaign_insight.contracts import SubagentInput

logger = logging.getLogger(__name__)

_GREETING_TEXT = (
    "Hi! I can help you analyze campaign performance across channels. "
    "What would you like to explore?"
)
_CLARIFICATION_TEXT = (
    "Could you clarify what you'd like to analyze? For example, mention a "
    "channel (email, SMS, WhatsApp, APN, BPN), a metric, or a time range."
)
_OOS_TEXT = (
    "I can't answer that because it's outside the campaign analytics scope. "
    "I can help with performance, trends, comparisons, and diagnostics across "
    "channels and campaigns."
)


class SupervisorRouter:
    """Routes classified queries to the appropriate downstream node."""

    def route(
        self,
        query: str,
        intent: dict,
        plan: list[str],
        context: dict,
    ) -> tuple[str, dict[str, Any]]:
        """Decide the target node and the state update to apply.

        Args:
            query: Original or rewritten user query.
            intent: Output of :class:`IntentClassifier`.
            plan: Output of :class:`SupervisorPlanner` (may be empty).
            context: Carries ``request_id``, ``client_id``,
                ``conversation_history``, and ``memory``.

        Returns:
            A tuple ``(target_node_name, state_update_dict)``.
        """
        intent_type = (intent or {}).get("intent_type", "")
        request_id = (context or {}).get("request_id", "")
        logger.info(
            f"[ROUTER] route() start | request_id={request_id} intent_type={intent_type} "
            f"plan_steps={len(plan or [])} query={str(query or '')[:80]!r}"
        )

        if intent_type == "greeting":
            logger.info(f"[ROUTER] → greeting | request_id={request_id}")
            return (
                "greeting",
                {"response_text": _GREETING_TEXT, "intent": "greeting"},
            )
        if intent_type == "clarification":
            logger.info(f"[ROUTER] → clarification | request_id={request_id}")
            return (
                "clarification",
                {"response_text": _CLARIFICATION_TEXT, "intent": "clarification"},
            )
        if intent_type == "out_of_scope":
            requires_audience = bool((intent or {}).get("requires_audience", False))
            requires_content = bool((intent or {}).get("requires_content", False))
            override_out_of_scope = requires_audience or requires_content

            if not override_out_of_scope:
                query = (query or "").lower()
                audience_keywords = ["segment", "audience", "lifecycle", "user", "intent"]
                content_keywords = ["content", "template", "cta", "keyword", "subject", "message"]
                override_out_of_scope = any(
                    keyword in query for keyword in audience_keywords + content_keywords
                )

            if not override_out_of_scope:
                return (
                    "out_of_scope",
                    {"response_text": _OOS_TEXT, "intent": "out_of_scope"},
                )

            intent = dict(intent or {})
            intent["intent_override"] = True

        sub_input = SubagentInput(
            request_id=(context or {}).get("request_id", ""),
            query=query or "",
            intent=dict(intent or {}),
            plan=list(plan or []),
            context={
                "client_id": (context or {}).get("client_id", ""),
                "conversation_history": (context or {}).get("conversation_history", ""),
                "memory": (context or {}).get("memory", ""),
            },
            config={"feature_flags": (context or {}).get("feature_flags", {})},
        )

        update: dict[str, Any] = {
            "subagent_input": asdict(sub_input),
            "rewritten_question": query or "",
            "intent": intent_type or "data_query",
        }
        return ("campaign_insight_agent", update)
