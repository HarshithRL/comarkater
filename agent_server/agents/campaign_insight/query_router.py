"""Rules-first query router for GENIE_DIRECT / HYBRID / AGENT_DECOMPOSE."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import DimensionClassification
from agents.campaign_insight.prompts.routing_prompt import (
    DECOMPOSE_KEYWORDS,
    GENIE_DIRECT_KEYWORDS,
    HYBRID_KEYWORDS,
    ROUTING_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Execution strategy picked by the router."""

    GENIE_DIRECT = "genie_direct"
    HYBRID = "hybrid"
    AGENT_DECOMPOSE = "agent_decompose"


@dataclass
class RoutingDecision:
    """Outcome of the router classification."""

    strategy: RoutingStrategy = RoutingStrategy.AGENT_DECOMPOSE
    reason: str = ""
    confidence: float = 0.0
    source: str = "rules"


_STRATEGY_LITERAL = Literal["genie_direct", "hybrid", "agent_decompose"]


class _RoutingSchema(BaseModel):
    """Pydantic schema for the LLM tiebreaker."""

    strategy: _STRATEGY_LITERAL = Field(
        default="agent_decompose",
        description="Execution strategy to use.",
    )
    reason: str = Field(default="", description="<=20 word reason.")


class QueryRouter:
    """Classify the user query by keyword rules, with optional LLM tiebreaker."""

    def __init__(
        self,
        llm: Any,
        domain_knowledge: Any,
        llm_tiebreaker_enabled: bool = False,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Initialize the router.

        Args:
            llm: Pre-built LLM supporting ``with_structured_output``. Used only
                when rules-based classification is below ``confidence_threshold``
                and ``llm_tiebreaker_enabled`` is True.
            domain_knowledge: Shared domain-knowledge helper (reserved for
                future domain-specific rules).
            llm_tiebreaker_enabled: If False, the router returns the best rules
                guess even when confidence is low.
            confidence_threshold: Below this, trigger the LLM tiebreaker (if
                enabled).
        """
        self._llm = llm
        self._domain = domain_knowledge
        self._llm_tiebreaker_enabled = llm_tiebreaker_enabled
        self._confidence_threshold = confidence_threshold

    def classify(
        self,
        query: str,
        intent: dict,
        dim_config: DimensionClassification,
    ) -> RoutingDecision:
        """Return a :class:`RoutingDecision` for the user query."""
        rules_decision = self._rules_classify(query, intent, dim_config)

        if (
            self._llm_tiebreaker_enabled
            and rules_decision.confidence < self._confidence_threshold
        ):
            llm_decision = self._llm_classify(query, intent, dim_config)
            if llm_decision is not None:
                return llm_decision

        self._emit(rules_decision)
        return rules_decision

    # ---- rules --------------------------------------------------------

    def _rules_classify(
        self,
        query: str,
        intent: dict,
        dim_config: DimensionClassification,
    ) -> RoutingDecision:
        q = (query or "").lower()
        decompose_hits = sum(1 for kw in DECOMPOSE_KEYWORDS if kw in q)
        direct_hits = sum(1 for kw in GENIE_DIRECT_KEYWORDS if kw in q)
        hybrid_hits = sum(1 for kw in HYBRID_KEYWORDS if kw in q)

        active_dims = len(dim_config.active_dimensions)
        intent_type = str((intent or {}).get("intent_type", "")).lower()

        if decompose_hits > 0:
            confidence = min(1.0, 0.6 + 0.15 * decompose_hits)
            return RoutingDecision(
                strategy=RoutingStrategy.AGENT_DECOMPOSE,
                reason=f"decompose keyword hit x{decompose_hits}",
                confidence=confidence,
                source="rules",
            )

        if active_dims >= 2:
            return RoutingDecision(
                strategy=RoutingStrategy.AGENT_DECOMPOSE,
                reason=f"{active_dims} active dimensions",
                confidence=0.75,
                source="rules",
            )

        if direct_hits > 0 and hybrid_hits == 0:
            confidence = min(1.0, 0.6 + 0.1 * direct_hits)
            return RoutingDecision(
                strategy=RoutingStrategy.GENIE_DIRECT,
                reason=f"direct keyword hit x{direct_hits}",
                confidence=confidence,
                source="rules",
            )

        if hybrid_hits > 0:
            confidence = min(1.0, 0.6 + 0.1 * hybrid_hits)
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID,
                reason=f"hybrid keyword hit x{hybrid_hits}",
                confidence=confidence,
                source="rules",
            )

        if intent_type in {"performance_lookup", "single_metric", "lookup"}:
            return RoutingDecision(
                strategy=RoutingStrategy.GENIE_DIRECT,
                reason=f"intent={intent_type}",
                confidence=0.55,
                source="rules",
            )

        return RoutingDecision(
            strategy=RoutingStrategy.HYBRID,
            reason="no strong signal; defaulting to hybrid",
            confidence=0.4,
            source="rules",
        )

    # ---- llm tiebreaker ----------------------------------------------

    def _llm_classify(
        self,
        query: str,
        intent: dict,
        dim_config: DimensionClassification,
    ) -> Optional[RoutingDecision]:
        try:
            active_lines = [
                f"- {name}: role={getattr(dim_config, name).role.value} "
                f"budget={getattr(dim_config, name).budget}"
                for name in ("campaign", "audience", "content")
            ]
            user_parts = [
                f"User query: {query.strip()}",
                f"Intent: {(intent or {}).get('intent_type', '')}",
                "Active dimensions:",
                *active_lines,
            ]
            structured = self._llm.with_structured_output(_RoutingSchema)
            raw: _RoutingSchema = structured.invoke(
                [
                    {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(user_parts)},
                ]
            )
            strategy = RoutingStrategy(raw.strategy)
            decision = RoutingDecision(
                strategy=strategy,
                reason=(raw.reason or "llm tiebreaker")[:120],
                confidence=0.8,
                source="llm",
            )
            self._emit(decision)
            return decision
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryRouter LLM tiebreaker failed: %s", exc)
            return None

    # ---- emit ---------------------------------------------------------

    def _emit(self, decision: RoutingDecision) -> None:
        logger.info(
            "[DEBUG][ROUTING] strategy=%s source=%s confidence=%.2f reason=%r",
            decision.strategy.value,
            decision.source,
            decision.confidence,
            decision.reason,
        )
        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({
                "event_type": "routing_ready",
                "strategy": decision.strategy.value,
                "source": decision.source,
                "confidence": decision.confidence,
            })
        except Exception:  # noqa: BLE001
            pass
