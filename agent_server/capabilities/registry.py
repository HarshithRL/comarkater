"""Capability registry — maps capability names to their configuration.

Each capability defines: tools it uses, system prompt, max ReAct iterations,
and a description. The planner references capability names, the graph builder
resolves them to subgraphs.

Adding a new capability:
  1. Add tools to tools/registry.py
  2. Add prompt to prompts/
  3. Add entry to CAPABILITIES dict below
  4. Create subgraph in capabilities/ (follow insight_agent.py pattern)
"""
from __future__ import annotations

from typing import Dict, List

from prompts.insight_agent_prompt import INSIGHT_AGENT_SYSTEM_PROMPT


CAPABILITIES: Dict[str, Dict] = {
    "insight_agent": {
        "tools": ["genie_search", "genie_query"],
        "system_prompt": INSIGHT_AGENT_SYSTEM_PROMPT,
        "max_iterations": 3,
        "description": "Campaign analytics using natural language and SQL queries via Genie",
    },
    # ── Example: adding a second capability ──
    # "knowledge_agent": {
    #     "tools": ["search_docs", "search_faq"],
    #     "system_prompt": KNOWLEDGE_AGENT_PROMPT,
    #     "max_iterations": 3,
    #     "description": "Domain knowledge Q&A from marketing docs and FAQs",
    # },
}

def get_capability_config(name: str) -> Dict:
    """Get full config dict for a capability.

    Raises:
        ValueError: If capability name is not registered.
    """
    if name not in CAPABILITIES:
        available = ", ".join(CAPABILITIES.keys())
        raise ValueError(
            f"Capability '{name}' not found. Available: {available}"
        )
    return CAPABILITIES[name]


def list_capabilities() -> List[str]:
    """Return all registered capability names."""
    return list(CAPABILITIES.keys())


def get_capability_description(name: str) -> str:
    """Get human-readable description for a capability.

    Returns empty string if capability not found (safe for prompt injection).
    """
    cap = CAPABILITIES.get(name)
    return cap["description"] if cap else ""
