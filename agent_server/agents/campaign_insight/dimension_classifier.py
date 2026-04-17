"""Dimension classifier for the Campaign Insight Agent.

One LLM call that decides which of the three analytical dimensions
(``campaign``, ``audience``, ``content``) to activate for a given user
query and assigns each a role + query budget.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    DimensionClassification,
    DimensionConfig,
    DimensionRole,
)
from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge
from memory.constants import CHANNEL_NAMES

logger = logging.getLogger(__name__)


_ROLE_LITERAL = Literal["primary", "supporting", "scope_only", "none"]


class _DimensionSlot(BaseModel):
    """Pydantic schema for one dimension slot in the LLM output."""

    role: _ROLE_LITERAL = Field(
        default="none",
        description="Role this dimension plays in the analysis.",
    )
    budget: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of queries (0-5) allocated to this dimension.",
    )


class _ClassificationSchema(BaseModel):
    """Pydantic schema mirroring DimensionClassification for structured output."""

    primary_analysis: str = Field(
        default="campaign",
        description='Which dimension is the focus: "campaign", "audience", or "content".',
    )
    campaign: _DimensionSlot = Field(default_factory=_DimensionSlot)
    audience: _DimensionSlot = Field(default_factory=_DimensionSlot)
    content: _DimensionSlot = Field(default_factory=_DimensionSlot)


_SYSTEM_PROMPT = """You are the Dimension Classifier for a marketing analytics agent.

Given a user question about email/SMS/push/WhatsApp campaign performance, decide
which of three dimensions must be analyzed, and how much query budget each gets.

Dimensions
----------
- campaign: campaign-level performance metrics (delivery, opens, clicks,
  conversions, revenue, bounces, unsubscribes). ALWAYS at least SCOPE_ONLY;
  PRIMARY for performance/metrics questions.
- audience: segment-level behavior - who clicked, converted, disengaged,
  lifecycle stage, intent, engagement health, app install status, retarget
  frequency. Activate when the query references segments / audience behavior.
- content: creative quality - emotion, CTA, subject line, template, content
  scores, readability, personalisation, emoji usage, message quality. Activate
  only when the query references creative/content attributes.

Roles
-----
- PRIMARY: the main focus of the answer.
- SUPPORTING: needed to explain/enrich the primary.
- SCOPE_ONLY: provides filtering/context but no standalone analysis.
- NONE: do not touch this dimension.

Budget rules
------------
- Each dimension: 0-5 queries.
- PRIMARY: 2-5. SUPPORTING: 1-3. SCOPE_ONLY: 1. NONE: 0.
- TOTAL budget across all dimensions MUST NOT EXCEED 8.
- Be conservative - only spend budget you will actually use.

Output exactly the required JSON schema.
"""


class DimensionClassifier:
    """Decide which dimensions to activate via a single structured LLM call."""

    def __init__(self, llm, domain_knowledge: InsightAgentDomainKnowledge) -> None:
        """Initialize the classifier.

        Args:
            llm: Pre-configured LangChain chat model (supports
                ``with_structured_output``). Temperature is the caller's choice.
            domain_knowledge: Shared domain-knowledge helper.
        """
        self._llm = llm
        self._domain = domain_knowledge

    def classify(self, query: str, intent: dict) -> DimensionClassification:
        """Classify the dimensions required to answer ``query``.

        Args:
            query: Raw user question.
            intent: Dict with at least an ``intent_type`` key from the
                supervisor.

        Returns:
            A populated :class:`DimensionClassification`. On failure, returns
            a safe default (campaign PRIMARY budget=3, others NONE).
        """
        intent_type = (intent or {}).get("intent_type", "")
        intent_complexity = (intent or {}).get("complexity", "")

        user_prompt = (
            f"User query:\n{query.strip()}\n\n"
            f"Intent: {intent_type} (complexity={intent_complexity})\n\n"
            "Decide per-dimension role + budget. Remember: total budget <= 8, "
            "campaign >= SCOPE_ONLY."
        )

        try:
            structured = self._llm.with_structured_output(_ClassificationSchema)
            result: _ClassificationSchema = structured.invoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            classification = self._to_dataclass(result)
            classification.channel = self._extract_channel(query)
            logger.info(
                "DimensionClassifier: primary=%s campaign=%s/%d audience=%s/%d content=%s/%d total=%d",
                classification.primary_analysis,
                classification.campaign.role.value,
                classification.campaign.budget,
                classification.audience.role.value,
                classification.audience.budget,
                classification.content.role.value,
                classification.content.budget,
                classification.total_budget,
            )
            return classification
        except Exception as exc:  # noqa: BLE001 - defensive fallback
            logger.error(
                "DimensionClassifier failed (%s) - returning safe default.", exc,
                exc_info=True,
            )
            return self._default_classification()

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_dataclass(schema: _ClassificationSchema) -> DimensionClassification:
        """Convert Pydantic schema output to the dataclass contract."""
        def _slot(s: _DimensionSlot) -> DimensionConfig:
            try:
                role = DimensionRole(s.role)
            except ValueError:
                role = DimensionRole.NONE
            budget = max(0, min(5, int(s.budget)))
            if role is DimensionRole.NONE:
                budget = 0
            return DimensionConfig(role=role, budget=budget)

        return DimensionClassification(
            primary_analysis=schema.primary_analysis or "campaign",
            campaign=_slot(schema.campaign),
            audience=_slot(schema.audience),
            content=_slot(schema.content),
        )

    @staticmethod
    def _default_classification() -> DimensionClassification:
        """Return the conservative default used on failures."""
        return DimensionClassification(
            primary_analysis="campaign",
            campaign=DimensionConfig(role=DimensionRole.PRIMARY, budget=3),
            audience=DimensionConfig(role=DimensionRole.NONE, budget=0),
            content=DimensionConfig(role=DimensionRole.NONE, budget=0),
        )

    @staticmethod
    def _extract_channel(query: str) -> str:
        """Return the first channel name found in the query, or ''."""
        q = (query or "").lower()
        for ch in CHANNEL_NAMES:
            if ch in q:
                return ch
        return ""
