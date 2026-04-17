"""STM message trimming — keep conversation history bounded.

Caps messages at MAX_STM_MESSAGES to prevent context window overflow
and unbounded checkpoint growth. Uses RemoveMessage to delete old
messages from the checkpoint via the add_messages reducer.
"""
from langgraph.graph.message import RemoveMessage

MAX_STM_MESSAGES = 20  # 10 exchanges (user + assistant pairs)


def get_trim_removals(messages: list) -> list:
    """Return RemoveMessage objects for messages beyond the window.

    The add_messages reducer processes RemoveMessage by deleting
    messages with matching IDs from the checkpoint.

    Args:
        messages: Current message list from checkpoint state.

    Returns:
        List of RemoveMessage objects for old messages, or empty list.
    """
    if len(messages) <= MAX_STM_MESSAGES:
        return []
    to_remove = messages[:-MAX_STM_MESSAGES]
    return [RemoveMessage(id=m.id) for m in to_remove if hasattr(m, "id") and m.id]
