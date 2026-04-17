"""Long-term memory (LTM) system for CoMarketer.

Per-client behavioral patterns stored in DatabricksStore (Lakebase).

Public API — import from here in agent.py:
    from memory import LTMManager, extract_entities_from_query, format_ltm_context
"""

from memory.extractors import (
    extract_entities_from_query,
    extract_metrics_from_query,
    extract_insights_from_response,
)
from memory.context_formatter import format_ltm_context
from memory.constants import LAKEBASE_INSTANCE_NAME, EMBEDDING_ENDPOINT, EMBEDDING_DIMS


def __getattr__(name):
    """Lazy import for LTMManager — avoids loading databricks_langchain at module import."""
    if name == "LTMManager":
        from memory.ltm_manager import LTMManager
        return LTMManager
    raise AttributeError(f"module 'memory' has no attribute {name!r}")


__all__ = [
    "LTMManager",
    "extract_entities_from_query",
    "extract_metrics_from_query",
    "extract_insights_from_response",
    "format_ltm_context",
    "LAKEBASE_INSTANCE_NAME",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_DIMS",
]
