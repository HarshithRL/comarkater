"""Capability registry — maps action verbs to executable tool functions.

All execution logic lives here. The planner references actions (verbs),
the executor dispatches through this registry. Adding a new capability
means adding ONE entry here — no graph or planner changes needed.

Tool contract: each function takes (state: dict, config: RunnableConfig) -> dict
and returns a state-update dict with genie_* fields.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableConfig

from agents.genie_agent import genie_data_node
from agents.genie_analysis import genie_analysis_node


# ── Registry: action verb → tool function ──
# Planner MUST use only keys from this registry.
CAPABILITY_REGISTRY: dict[str, Callable[[dict, RunnableConfig], dict]] = {
    "fetch_data": genie_data_node,
    "analyze_data": genie_analysis_node,
}

# Exported for planner prompt injection
ACTION_VOCABULARY: list[str] = list(CAPABILITY_REGISTRY.keys())

