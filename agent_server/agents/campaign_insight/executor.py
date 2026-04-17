"""Execute an ExecutionPlan, respecting step dependencies and parallelism."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.campaign_insight.contracts import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    StepStatus,
)
from agents.campaign_insight.prompts.insight_prompt import (
    INSIGHT_REACT_PROMPT,
    STEP_REASONING_PROMPT,
)

logger = logging.getLogger(__name__)


def _timeout_result(step: PlanStep) -> StepResult:
    return StepResult(
        step_id=step.step_id,
        dimension=step.dimension,
        status=StepStatus.TIMEOUT,
        error_message="Total execution timeout",
    )


class ReActExecutor:
    """Run a plan under a controlled REASON -> ACT -> OBSERVE -> EVALUATE loop."""

    def __init__(
        self,
        llm: Any,
        tool_handler: Any,
        table_analyzer: Any,
        table_builder: Any,
        domain_knowledge: Any,
        max_iterations_per_step: int = 3,
        total_timeout_seconds: int = 120,
    ) -> None:
        """Initialize the ReAct executor.

        Args:
            llm: Pre-built LLM instance supporting ``invoke``.
            tool_handler: Object exposing ``execute_query_with_retry``.
            table_analyzer: Object exposing ``analyze``.
            table_builder: Object exposing ``build_display_table``.
            domain_knowledge: ``InsightAgentDomainKnowledge`` instance.
            max_iterations_per_step: Max REASON/ACT cycles per step.
            total_timeout_seconds: Overall wall clock budget for the plan.
        """
        self.llm = llm
        self.tool_handler = tool_handler
        self.table_analyzer = table_analyzer
        self.table_builder = table_builder
        self.domain_knowledge = domain_knowledge
        self.max_iterations_per_step = max_iterations_per_step
        self.total_timeout_seconds = total_timeout_seconds

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        stream_callback: Optional[Callable[[dict], None]] = None,
        channel: str = "",
    ) -> dict[int, StepResult]:
        """Execute the plan step-by-step, honoring dependencies and timeout.

        Args:
            plan: ExecutionPlan to run.
            stream_callback: Optional callable invoked with streaming events.

        Returns:
            Mapping of ``step_id`` to ``StepResult`` for every step in the plan.
        """
        start_time = time.monotonic()
        completed: set[int] = set()
        results: dict[int, StepResult] = {}

        while True:
            ready = plan.get_ready_steps(completed)
            if not ready:
                break

            if time.monotonic() - start_time > self.total_timeout_seconds:
                for s in plan.steps:
                    if s.step_id not in results:
                        results[s.step_id] = _timeout_result(s)
                break

            for step in ready:
                if time.monotonic() - start_time > self.total_timeout_seconds:
                    results[step.step_id] = _timeout_result(step)
                    completed.add(step.step_id)
                    continue

                step_result = await self._execute_single_step(step, results, channel)
                results[step.step_id] = step_result
                completed.add(step.step_id)
                self._emit(stream_callback, step, step_result)

        for s in plan.steps:
            if s.step_id not in results:
                results[s.step_id] = _timeout_result(s)
        return results

    def _emit(
        self,
        stream_callback: Optional[Callable[[dict], None]],
        step: PlanStep,
        sr: StepResult,
    ) -> None:
        if stream_callback is None:
            return
        try:
            row_count = sr.table_summary.row_count if sr.table_summary else 0
            stream_callback(
                {
                    "event_type": "step_completed",
                    "step_id": step.step_id,
                    "dimension": step.dimension,
                    "row_count": row_count,
                    "status": sr.status.value,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("stream_callback failed: %s", exc)

    async def _execute_single_step(
        self,
        step: PlanStep,
        prior_results: dict[int, StepResult],
        channel: str = "",
    ) -> StepResult:
        """Run the ReAct loop for one step and return a StepResult."""
        current_nl_query = step.query
        iterations_used = 0
        last_error = ""
        last_trace_id = ""
        last_sql = ""
        max_i = self.max_iterations_per_step

        def _err(msg: str) -> StepResult:
            return StepResult(
                step_id=step.step_id, dimension=step.dimension,
                status=StepStatus.ERROR, sql=last_sql,
                genie_trace_id=last_trace_id,
                iterations_used=iterations_used, error_message=msg,
            )

        for iteration in range(max_i):
            iterations_used = iteration + 1
            is_last = iteration + 1 >= max_i

            if iteration > 0:
                try:
                    refined = self._reason_next_query(
                        step, prior_results, last_error
                    )
                    if refined:
                        current_nl_query = refined
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Reasoning refinement failed: %s", exc)

            # ACT
            try:
                nl_query = (
                    f"For channel = {channel} only. {current_nl_query}"
                    if channel
                    else current_nl_query
                )
                gr = await self.tool_handler.execute_query_with_retry(nl_query)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool call raised for step %s", step.step_id)
                last_error = str(exc)
                if is_last:
                    return _err(last_error)
                continue

            last_sql = gr.sql or last_sql
            last_trace_id = gr.genie_trace_id or last_trace_id
            status = (gr.status or "").lower()

            # OBSERVE
            if status == "error":
                last_error = gr.error_message or "Unknown error"
                if is_last:
                    return _err(last_error)
                continue

            if status == "feedback_needed":
                return StepResult(
                    step_id=step.step_id, dimension=step.dimension,
                    status=StepStatus.PARTIAL, sql=last_sql,
                    genie_trace_id=last_trace_id,
                    iterations_used=iterations_used,
                    error_message=gr.error_message or "Clarification needed",
                )

            if status == "success":
                try:
                    summary = self.table_analyzer.analyze(
                        gr.columns, gr.data_array, gr.row_count
                    )
                    display_table = self.table_builder.build_display_table(
                        gr.columns, gr.data_array,
                        title=step.purpose or step.query,
                        sql=gr.sql,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Post-processing failed on step %s", step.step_id
                    )
                    last_error = f"Post-processing error: {exc}"
                    if is_last:
                        return _err(last_error)
                    continue

                sufficient = gr.row_count > 0 and bool(gr.columns)
                if sufficient or is_last:
                    return StepResult(
                        step_id=step.step_id, dimension=step.dimension,
                        status=StepStatus.SUCCESS if sufficient
                        else StepStatus.PARTIAL,
                        display_table=display_table, table_summary=summary,
                        sql=gr.sql, genie_trace_id=gr.genie_trace_id,
                        iterations_used=iterations_used,
                        error_message="" if sufficient else "Empty result set",
                    )
                last_error = "Empty result set; will refine"
                continue

            last_error = gr.error_message or f"Unknown status: {status}"
            if is_last:
                return _err(last_error)

        return _err(last_error or "Step exhausted without success")

    def _reason_next_query(
        self,
        step: PlanStep,
        prior_results: dict[int, StepResult],
        last_error: str,
    ) -> str:
        """Ask the LLM for a refined NL query after a failure."""
        prior_context = self._compress_prior_results(prior_results, step.step_id)
        domain_note = ""
        try:
            thresholds = self.domain_knowledge.get_minimum_volume_thresholds()
            domain_note = f"Minimum volume thresholds: {thresholds}"
            if last_error:
                domain_note += f"\nPrevious attempt error: {last_error}"
        except Exception:  # noqa: BLE001
            domain_note = (
                f"Previous attempt error: {last_error}" if last_error else ""
            )

        user_content = STEP_REASONING_PROMPT.format(
            task=step.purpose or step.query,
            dimension=step.dimension,
            prior_context=prior_context or "(none)",
            domain_note=domain_note or "(none)",
        )
        try:
            dk = self.domain_knowledge.format_for_subagent()
        except Exception:  # noqa: BLE001
            dk = "(domain knowledge unavailable)"
        system_content = INSIGHT_REACT_PROMPT.format(domain_knowledge=dk)
        response = self.llm.invoke(
            [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        )
        return (getattr(response, "content", "") or "").strip()

    def _compress_prior_results(
        self,
        results: dict[int, StepResult],
        current_step_id: int,
    ) -> str:
        """Return a compact <500-char per-step summary of prior results."""
        parts: list[str] = []
        for sid, sr in sorted(results.items()):
            if sid == current_step_id:
                continue
            row_count = sr.table_summary.row_count if sr.table_summary else 0
            finding = ""
            if sr.table_summary and sr.table_summary.aggregates:
                try:
                    first_agg = next(iter(sr.table_summary.aggregates.items()))
                    finding = f"{first_agg[0]}={first_agg[1]}"
                except Exception:  # noqa: BLE001
                    finding = ""
            entry = (
                f"step {sid} [{sr.dimension}] status={sr.status.value} "
                f"rows={row_count} {finding}"
            )
            parts.append(entry[:500])
        return "\n".join(parts)
