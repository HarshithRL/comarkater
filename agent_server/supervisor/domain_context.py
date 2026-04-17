"""Condensed domain context for the fast supervisor model.

Thin wrapper around :class:`InsightAgentDomainKnowledge.format_for_supervisor`
so supervisor modules can depend on a stable, narrow surface.
"""
from __future__ import annotations

import logging

from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge

logger = logging.getLogger(__name__)


class SupervisorDomainContext:
    """Provides a condensed domain block for supervisor prompts.

    The underlying domain knowledge object already produces a supervisor-sized
    reference via ``format_for_supervisor`` — this class simply caches it so
    the expensive string assembly runs once per supervisor instance.
    """

    def __init__(self, domain_knowledge: InsightAgentDomainKnowledge) -> None:
        """Store the domain-knowledge source.

        Args:
            domain_knowledge: Loaded :class:`InsightAgentDomainKnowledge`.
        """
        self._domain_knowledge = domain_knowledge
        self._cached: str | None = None

    def format_for_prompt(self) -> str:
        """Return the condensed supervisor-facing domain block.

        Returns:
            A markdown-formatted domain reference suitable for inclusion in
            supervisor-tier prompts (vocabulary, channels, intents, and
            channel constraints — no formulas or thresholds).
        """
        if self._cached is None:
            try:
                self._cached = self._domain_knowledge.format_for_supervisor()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("format_for_supervisor failed: %s", exc)
                self._cached = "# DOMAIN CONTEXT\n(unavailable)"
        return self._cached
