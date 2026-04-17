"""Feedback logging — accepts user like/dislike and logs to MLflow traces.

Feedback flow:
  App logs trace → returns mlflow_trace_id to UI via streaming event
  → user clicks 👍/👎 → UI calls POST /ui/feedback with trace_id
  → this module calls mlflow.log_feedback() to attach human assessment
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def log_user_feedback(
    trace_id: str,
    is_helpful: bool,
    comment: Optional[str],
    user_id: str,
) -> bool:
    """Log human feedback to an MLflow trace.

    Args:
        trace_id:   MLflow trace ID returned from mlflow.get_last_active_trace_id()
                    or captured from the active span's request_id during streaming.
        is_helpful: True = thumbs up, False = thumbs down.
        comment:    Optional free-text rationale (shown when thumbs down is clicked).
        user_id:    Identifier for the user submitting feedback (e.g. 'ui-user').

    Returns:
        True on success, False on any error (caller logs HTTP 500).
    """
    try:
        import mlflow
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        source = AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id or "ui-user",
        )
        mlflow.log_feedback(
            trace_id=trace_id,
            name="user_feedback",
            source=source,
            value=is_helpful,
            rationale=comment if comment else None,
        )
        logger.info(
            f"FEEDBACK: logged | trace_id={trace_id} | is_helpful={is_helpful} "
            f"| user_id={user_id} | has_comment={bool(comment)}"
        )
        return True

    except Exception as e:
        logger.error(
            f"FEEDBACK: log failed | trace_id={trace_id} | error={e}",
            exc_info=True,
        )
        return False
