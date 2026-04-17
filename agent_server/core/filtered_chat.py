"""FilteredChatDatabricks — Layer 1 model-level message filter.

Extends ChatDatabricks to:
1. Strip raw sub-agent responses (<name>...</name> markup) from message history
2. Strip tool_calls from AIMessage responses (prevent re-routing loops)

This is defense layer 1 of 3 against dirty messages in conversation history.

Ported from legacy: lines 1938-1955.
"""

import logging

from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage

from parsers.filters import is_raw_subagent_response

logger = logging.getLogger(__name__)


class FilteredChatDatabricks(ChatDatabricks):
    """ChatDatabricks that filters raw sub-agent messages before LLM invoke."""

    def invoke(self, input, config=None, **kwargs):
        """Filter input messages then invoke the underlying model.

        Removes messages with <name>...</name> tags (raw sub-agent responses).
        Strips tool_calls from AIMessage results (prevents re-routing).
        """
        if isinstance(input, list):
            original_count = len(input)
            cleaned = []
            for msg in input:
                content = str(getattr(msg, 'content', ''))
                if is_raw_subagent_response(content):
                    logger.info(f"FILTER-MODEL: Stripping raw sub-agent msg: {content[:100]}")
                    continue
                cleaned.append(msg)
            if len(cleaned) < original_count:
                logger.info(f"FILTER-MODEL: Removed {original_count - len(cleaned)} raw sub-agent msgs")
            input = cleaned

        result = super().invoke(input, config, **kwargs)

        if isinstance(result, AIMessage) and getattr(result, "tool_calls", None):
            return AIMessage(content=result.content or "")

        return result
