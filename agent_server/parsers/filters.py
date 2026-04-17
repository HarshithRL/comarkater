"""Message filtering functions for error/thinking/raw sub-agent detection.

LEAF module — no internal imports. Used by FilteredChatDatabricks (Layer 1),
STM cleanup (Layer 2/3), and streaming dedup.

Ported from legacy: lines 1177-1216.
"""

import re


def is_error_message(content: str) -> bool:
    """Detect LangGraph/tool error messages that should be filtered."""
    if not content:
        return False
    error_indicators = [
        "Unable to interpret tool invocation history",
        "Did not find a previous tool call for",
        "Error: Unable to interpret",
    ]
    return any(indicator in content for indicator in error_indicators)


def is_intermediate_thinking(content: str) -> bool:
    """Detect LLM 'thinking aloud' messages (pre-routing chatter).

    Returns False for structured JSON output (items/tableHeaders).
    Only matches short (<500 char) messages with thinking patterns.
    """
    if not content:
        return False
    content_stripped = content.strip()
    if content_stripped.startswith('{') or content_stripped.startswith('[['):
        return False
    if '"items"' in content or '"tableHeaders"' in content:
        return False
    thinking_patterns = [
        "i'll analyze", "i will analyze", "let me analyze",
        "i'll compare", "i will compare", "let me compare",
    ]
    content_lower = content.lower()
    has_thinking = any(p in content_lower for p in thinking_patterns)
    return has_thinking and len(content) < 500


def is_raw_subagent_response(content: str) -> bool:
    """Detect raw sub-agent responses with <name>...</name> markup."""
    if not content:
        return False
    has_name_tags = '<name>' in content and '</name>' in content
    is_json = content.strip().startswith('{')
    return has_name_tags and not is_json


def should_filter_message(content: str) -> bool:
    """Return True if message should be filtered from output/history."""
    return is_error_message(content) or is_intermediate_thinking(content)
