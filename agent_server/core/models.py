"""Pydantic models, schemas, and constants. LEAF module — no internal imports."""

from typing import Optional

from pydantic import BaseModel


class CustomInputs(BaseModel):
    """Validated custom inputs from the request.

    Fields:
        user_name: Display name for personalization.
        user_id: Unique user identifier (LTM key).
        conversation_id: Session/conversation identifier.
        task_type: Type of task (general, analytics, etc.).
        thread_id: Thread ID for STM checkpoints.
        conversation_history: Serialized STM history (injected).
        ltm_context: Formatted LTM markdown (injected).
    """

    client_scope: str = ""
    client_id: str = ""
    user_name: str = "User"
    user_id: str = ""
    conversation_id: str = ""
    task_type: str = "general"
    thread_id: Optional[str] = None
    conversation_history: Optional[str] = None
    ltm_context: Optional[str] = None


# ── Response format schema (bound to supervisor LLM) ──
RESPONSE_FORMAT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "campaign_analysis_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["text", "table", "chart", "collapsedText"],
                            },
                            "id": {"type": "string"},
                            "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "value": {"type": "string"},
                            "hidden": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                        },
                        "required": ["type", "id", "name", "value", "hidden"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


