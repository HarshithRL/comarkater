"""Deterministic validator/corrector for :class:`DimensionClassification`.

No LLM is used here. Applies keyword gating, feature-flag gating, content/
campaign overlap guards, and a budget clamp so that downstream planning
sees a sane configuration.
"""
from __future__ import annotations

import logging

from agents.campaign_insight.contracts import (
    DimensionClassification,
    DimensionConfig,
    DimensionRole,
)

logger = logging.getLogger(__name__)


_AUDIENCE_KEYWORDS: tuple[str, ...] = (
    "segment",
    "who clicked",
    "who converted",
    "intent",
    "lifecycle",
    "engaged",
    "disengaged",
    "audience",
    "app status",
    "retarget",
    "communication health",
    "install",
    "active user",
)

_CONTENT_KEYWORDS: tuple[str, ...] = (
    "emotion",
    "cta",
    "content score",
    "subject line",
    "template",
    "readability",
    "personalisation",
    "personalization",
    "emoji",
    "message quality",
)

_BUDGET_CAP = 8


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    """Case-insensitive substring match for any needle."""
    return any(n in haystack for n in needles)


class DimensionValidator:
    """Apply deterministic correction rules to a classification."""

    def validate(
        self,
        classification: DimensionClassification,
        query: str,
        feature_flags: dict,
    ) -> DimensionClassification:
        """Mutate ``classification`` in place and return it.

        Args:
            classification: Output from the :class:`DimensionClassifier`.
            query: Raw user query - inspected for keyword gating.
            feature_flags: Runtime flags; recognized keys are
                ``ENABLE_AUDIENCE_ANALYSIS`` and ``ENABLE_CONTENT_ANALYSIS``.

        Returns:
            The (possibly modified) ``classification``.
        """
        corrections: list[str] = []
        q = (query or "").lower()
        flags = feature_flags or {}

        # 1. Audience keyword gating ------------------------------------------------
        if classification.audience.role is not DimensionRole.NONE and not _contains_any(q, _AUDIENCE_KEYWORDS):
            corrections.append("audience-gated: no audience keywords in query")
            classification.audience = DimensionConfig(role=DimensionRole.NONE, budget=0)

        # 2. Content keyword gating -------------------------------------------------
        if classification.content.role is not DimensionRole.NONE and not _contains_any(q, _CONTENT_KEYWORDS):
            corrections.append("content-gated: no content keywords in query")
            classification.content = DimensionConfig(role=DimensionRole.NONE, budget=0)

        # 3. Feature flag check -----------------------------------------------------
        if flags.get("ENABLE_AUDIENCE_ANALYSIS") is False and classification.audience.role is not DimensionRole.NONE:
            corrections.append("audience disabled by feature flag")
            classification.audience = DimensionConfig(role=DimensionRole.NONE, budget=0)
        if flags.get("ENABLE_CONTENT_ANALYSIS") is False and classification.content.role is not DimensionRole.NONE:
            corrections.append("content disabled by feature flag")
            classification.content = DimensionConfig(role=DimensionRole.NONE, budget=0)

        # 4. Content-primary guard --------------------------------------------------
        if (
            classification.content.role is DimensionRole.PRIMARY
            and classification.campaign.role is DimensionRole.PRIMARY
        ):
            corrections.append("content-primary guard: demoted campaign to SUPPORTING")
            classification.campaign = DimensionConfig(
                role=DimensionRole.SUPPORTING,
                budget=max(1, classification.campaign.budget),
            )

        # 5. Budget clamping --------------------------------------------------------
        self._clamp_budget(classification, corrections)

        # 6. Campaign safety net ----------------------------------------------------
        if classification.campaign.role is DimensionRole.NONE or classification.campaign.budget < 1:
            corrections.append("restored campaign to SCOPE_ONLY budget=1")
            classification.campaign = DimensionConfig(role=DimensionRole.SCOPE_ONLY, budget=1)

        if corrections:
            logger.info("DimensionValidator corrections: %s", "; ".join(corrections))
        else:
            logger.debug("DimensionValidator: no corrections")

        return classification

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _clamp_budget(
        classification: DimensionClassification,
        corrections: list[str],
    ) -> None:
        """Reduce budgets until total <= cap, lowest-role-first.

        Order of decrement: SUPPORTING first, then SCOPE_ONLY, then (only as
        a last resort) PRIMARY - but PRIMARY is never taken below 1.
        """
        dimension_names = ("campaign", "audience", "content")

        def total() -> int:
            return classification.total_budget

        def _decrement_pass(target_roles: tuple[DimensionRole, ...]) -> bool:
            """Remove 1 budget from one dimension matching target_roles."""
            reduced = False
            for name in dimension_names:
                if total() <= _BUDGET_CAP:
                    return reduced
                cfg: DimensionConfig = getattr(classification, name)
                if cfg.role not in target_roles:
                    continue
                min_floor = 1 if cfg.role is DimensionRole.PRIMARY else 0
                if cfg.budget > min_floor:
                    cfg.budget -= 1
                    reduced = True
                    corrections.append(f"clamp: {name} budget -1 (role={cfg.role.value})")
                    if cfg.budget == 0 and cfg.role in (DimensionRole.SUPPORTING, DimensionRole.SCOPE_ONLY):
                        corrections.append(f"clamp: {name} dropped to NONE (budget=0)")
                        cfg.role = DimensionRole.NONE
            return reduced

        # Phase 1: SUPPORTING first.
        while total() > _BUDGET_CAP:
            if not _decrement_pass((DimensionRole.SUPPORTING,)):
                break

        # Phase 2: SCOPE_ONLY next.
        while total() > _BUDGET_CAP:
            if not _decrement_pass((DimensionRole.SCOPE_ONLY,)):
                break

        # Phase 3 (last resort): PRIMARY down to floor of 1.
        while total() > _BUDGET_CAP:
            if not _decrement_pass((DimensionRole.PRIMARY,)):
                break
