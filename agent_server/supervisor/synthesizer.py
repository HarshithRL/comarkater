"""Supervisor synthesizer — composes the final user-facing narrative.

Takes a ``SubagentOutput`` (or its dict form) and emits a list of response
items in the shape the UI expects: an LLM-composed text narrative, followed by
any display tables, and optionally a chart.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from .domain_context import SupervisorDomainContext

logger = logging.getLogger(__name__)


_SYNTH_PROMPT = """You are a marketing analytics expert writing to a marketer.
Compose a clear, decision-oriented narrative from the analysis below.

{domain_context}

Analysis summary: {summary}

Insights (do not invent new ones):
{insights}

Recommendations (do not invent new ones):
{recommendations}

Caveats:
{caveats}

Rules:
- Never claim anything not present in the summary, insights, or recommendations.
- Never re-compute or re-interpret metrics.
- Structure: a short summary paragraph, then bullet insights, then bullet recommendations, then caveats.
- Do NOT mention internal systems (Genie, Databricks, Unity Catalog, SQL).
- Do NOT use currency symbols — use "units".
- Markdown bullets with "-" only.
"""


def _to_dict(obj: Any) -> Any:
    """Return obj as a dict if it's a dataclass, otherwise passthrough."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


class SupervisorSynthesizer:
    """Composes final response items from a subagent output."""

    def __init__(self, llm: Any, domain_context: SupervisorDomainContext) -> None:
        """Store the LLM and domain context.

        Args:
            llm: LangChain chat model (used for narrative generation).
            domain_context: Condensed domain reference supplier.
        """
        self._llm = llm
        self._domain_context = domain_context

    # ------------------------------------------------------------------ public

    def synthesize(self, subagent_output: Any) -> list[dict]:
        """Return a list of response items for the UI.

        Args:
            subagent_output: A :class:`SubagentOutput` dataclass or its dict.

        Returns:
            Ordered list of items: text narrative, tables, optional chart.
        """
        data = _to_dict(subagent_output) or {}
        interpretation = _to_dict(data.get("interpretation")) or {}
        recommendations = [_to_dict(r) or {} for r in (data.get("recommendations") or [])]
        tables = data.get("tables_for_display") or []
        chart_spec = data.get("chart_spec")
        caveats = data.get("caveats") or []
        streamed_types = set((data.get("metadata") or {}).get("streamed_types") or [])

        summary = str(interpretation.get("summary", "") or "")
        insights = [str(i) for i in (interpretation.get("insights") or []) if i]

        narrative = self._compose_narrative(summary, insights, recommendations, caveats)

        items: list[dict] = []
        items.append({
            "type": "text",
            "id": str(uuid.uuid4()),
            "value": narrative,
            "hidden": False,
            "name": "narrative",
        })

        if "table" not in streamed_types:
            for tbl in tables:
                t = _to_dict(tbl) or {}
                columns = list(t.get("columns") or [])
                rows = list(t.get("rows") or [])
                items.append({
                    "type": "table",
                    "id": str(uuid.uuid4()),
                    "value": {
                        "tableHeaders": columns,
                        "data": rows,
                        "alignment": ["left"] * len(columns),
                    },
                    "hidden": False,
                    "name": str(t.get("title", "")) or "table",
                })

        if chart_spec and "chart" not in streamed_types:
            items.append({
                "type": "chart",
                "id": str(uuid.uuid4()),
                "value": chart_spec,
                "hidden": False,
                "name": "chart",
            })

        return items

    # ----------------------------------------------------------------- private

    def _compose_narrative(
        self,
        summary: str,
        insights: list[str],
        recommendations: list[dict],
        caveats: list[str],
    ) -> str:
        """Use the LLM to compose a narrative; fall back to deterministic text."""
        insights_block = "\n".join(f"- {i}" for i in insights) or "- (none)"
        rec_lines: list[str] = []
        for r in recommendations:
            action = str(r.get("action", "")).strip()
            detail = str(r.get("detail", "")).strip()
            if action and detail:
                rec_lines.append(f"- {action}: {detail}")
            elif action:
                rec_lines.append(f"- {action}")
        recs_block = "\n".join(rec_lines) or "- (none)"
        caveats_block = "\n".join(f"- {c}" for c in caveats) or "- (none)"

        prompt = _SYNTH_PROMPT.format(
            domain_context=self._domain_context.format_for_prompt(),
            summary=summary or "(no summary)",
            insights=insights_block,
            recommendations=recs_block,
            caveats=caveats_block,
        )

        try:
            resp = self._llm.invoke(prompt)
            text = getattr(resp, "content", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception as exc:
            logger.warning("Synthesizer LLM failed, using fallback: %s", exc)

        # Deterministic fallback.
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if insights:
            parts.append("\n**Insights**\n" + "\n".join(f"- {i}" for i in insights))
        if rec_lines:
            parts.append("\n**Recommendations**\n" + "\n".join(rec_lines))
        if caveats:
            parts.append("\n**Caveats**\n" + "\n".join(f"- {c}" for c in caveats))
        return "\n".join(parts).strip() or "No analysis available."
