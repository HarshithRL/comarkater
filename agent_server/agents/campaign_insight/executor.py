"""Execute an ExecutionPlan, respecting step dependencies and parallelism."""
from __future__ import annotations

import asyncio
import json
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
from agents.campaign_insight.genie_validator import (
    GenieResultValidator,
    ValidationResult,
)
from agents.campaign_insight.prompts.insight_prompt import (
    INSIGHT_REACT_PROMPT,
    STEP_REASONING_PROMPT,
)
from agents.campaign_insight.query_router import RoutingDecision
from core.tracing import flow_log

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
        total_timeout_seconds: int = 180,
        step_timeout_seconds: int = 60,
        validator: Optional[GenieResultValidator] = None,
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
            step_timeout_seconds: Per-step wall clock cap on the Genie call.
                A stuck step returns ``StepStatus.TIMEOUT`` instead of blocking
                the whole request.
        """
        self.llm = llm
        self.tool_handler = tool_handler
        self.table_analyzer = table_analyzer
        self.table_builder = table_builder
        self.domain_knowledge = domain_knowledge
        self.max_iterations_per_step = max_iterations_per_step
        self.total_timeout_seconds = total_timeout_seconds
        self.step_timeout_seconds = step_timeout_seconds
        self.validator = validator

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        stream_callback: Optional[Callable[[dict], None]] = None,
        channel: str = "",
        request_id: str = "",
        routing: Optional[RoutingDecision] = None,
    ) -> dict[int, StepResult]:
        _ = routing  # reserved for future per-step strategy hints
        """Execute the plan step-by-step, honoring dependencies and timeout.

        Args:
            plan: ExecutionPlan to run.
            stream_callback: Optional callable invoked with streaming events.
            channel: Optional channel scope to prepend to every NL query.
            request_id: Upstream request id, used only for per-step flow logs.

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
                        flow_log(
                            request_id,
                            "step",
                            step_id=s.step_id,
                            dim=s.dimension,
                            status="timeout",
                        )
                break

            for step in ready:
                if time.monotonic() - start_time > self.total_timeout_seconds:
                    results[step.step_id] = _timeout_result(step)
                    completed.add(step.step_id)
                    flow_log(
                        request_id,
                        "step",
                        step_id=step.step_id,
                        dim=step.dimension,
                        status="timeout",
                    )
                    continue

                step_result = await self._execute_single_step(step, results, channel)
                results[step.step_id] = step_result
                completed.add(step.step_id)
                self._emit(stream_callback, step, step_result)
                _row_count = step_result.table_summary.row_count if step_result.table_summary else 0
                _dt = step_result.display_table
                _cols = list(_dt.columns) if _dt else []
                _sample_rows = []
                if _dt and _dt.rows:
                    for _r in _dt.rows[:3]:
                        _sample_rows.append([str(v)[:80] for v in _r])
                try:
                    _sample_json = json.dumps(_sample_rows, default=str)[:600]
                except Exception:
                    _sample_json = "<unserializable>"
                logger.info(
                    "[DEBUG][EXECUTION] step_id=%s status=%s rows=%s columns=%s sample=%s",
                    step.step_id,
                    step_result.status.value,
                    _row_count,
                    _cols,
                    _sample_json,
                )
                flow_log(
                    request_id,
                    "step",
                    step_id=step.step_id,
                    dim=step.dimension,
                    status=step_result.status.value,
                    iters=step_result.iterations_used,
                    rows=_row_count,
                    sql_len=len(step_result.sql or ""),
                    err=(step_result.error_message or "")[:120] or None,
                )

        for s in plan.steps:
            if s.step_id not in results:
                results[s.step_id] = _timeout_result(s)
                flow_log(
                    request_id,
                    "step",
                    step_id=s.step_id,
                    dim=s.dimension,
                    status="timeout",
                )
        return results

    def _emit(
        self,
        stream_callback: Optional[Callable[[dict], None]],
        step: PlanStep,
        sr: StepResult,
    ) -> None:
        """Emit step_completed + table_ready for every completed step.

        Streaming is a first-class execution primitive: every SUCCESS /
        PARTIAL step produces exactly one table_ready event, even if the
        underlying query returned no rows. This guarantees the fast path
        always streams a table, so the synthesizer-fallback layer only
        fills truly missing pieces.

        ERROR / TIMEOUT steps emit step_completed only — the caller surfaces
        the failure through a status event, not an empty data grid.
        """
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

        if sr.status not in (StepStatus.SUCCESS, StepStatus.PARTIAL):
            return

        dt = sr.display_table
        has_rows = dt is not None and bool(dt.columns) and bool(dt.rows)
        try:
            if has_rows:
                columns = [str(c) for c in dt.columns]  # type: ignore[union-attr]
                rows = [list(r) for r in dt.rows]  # type: ignore[union-attr]
                table_payload = {
                    "tableHeaders": columns,
                    "data": rows,
                    "alignment": ["left"] * len(columns),
                }
                title = dt.title or f"Step {step.step_id}"  # type: ignore[union-attr]
            else:
                # Placeholder keeps type="table" so the UI can render a
                # "no rows" block in the Data section instead of hiding
                # the step entirely. Downstream gate treats this like
                # any other table_ready event.
                table_payload = {
                    "tableHeaders": ["Status"],
                    "data": [["No rows returned for this step."]],
                    "alignment": ["left"],
                }
                title = f"Step {step.step_id}"
            stream_callback(
                {
                    "event_type": "table_ready",
                    "item_id": f"table_step_{step.step_id}",
                    "step_id": step.step_id,
                    "title": title,
                    "table": table_payload,
                    "placeholder": not has_rows,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("table_ready emission failed: %s", exc)

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
                # Small backoff lets transient Genie spikes recover before
                # we burn the next refinement attempt. Keep it short so we
                # stay well under step_timeout_seconds.
                try:
                    await asyncio.sleep(min(1.0 * iteration, 2.0))
                except Exception:  # noqa: BLE001
                    pass
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
                gr = await asyncio.wait_for(
                    self.tool_handler.execute_query_with_retry(nl_query),
                    timeout=self.step_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Step %s timed out after %ss",
                    step.step_id, self.step_timeout_seconds,
                )
                return StepResult(
                    step_id=step.step_id,
                    dimension=step.dimension,
                    status=StepStatus.TIMEOUT,
                    sql=last_sql,
                    genie_trace_id=last_trace_id,
                    iterations_used=iterations_used,
                    error_message=(
                        f"Step timed out after {self.step_timeout_seconds}s"
                    ),
                )
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
                # SQL-guard rejection is self-inflicted (the LLM produced
                # SQL-flavored English the guard blocks before Genie). Don't
                # let it burn the iteration — re-reason inline and re-ACT.
                if "sql detected" in last_error.lower():
                    try:
                        stricter = self._reason_next_query(
                            step, prior_results, last_error
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("inline reason after SQL-guard failed: %s", exc)
                        stricter = ""
                    if stricter and stricter != current_nl_query:
                        current_nl_query = stricter
                        logger.info(
                            "Step %s: inline retry after SQL-guard rejection",
                            step.step_id,
                        )
                        try:
                            nl_query2 = (
                                f"For channel = {channel} only. {current_nl_query}"
                                if channel
                                else current_nl_query
                            )
                            gr2 = await asyncio.wait_for(
                                self.tool_handler.execute_query_with_retry(nl_query2),
                                timeout=self.step_timeout_seconds,
                            )
                            last_sql = gr2.sql or last_sql
                            last_trace_id = gr2.genie_trace_id or last_trace_id
                            gr = gr2
                            status = (gr.status or "").lower()
                            if status == "error":
                                last_error = gr.error_message or last_error
                        except asyncio.TimeoutError:
                            pass  # fall through to normal error handling
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("inline SQL-retry ACT failed: %s", exc)
                if status != "success":
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
                vr: Optional[ValidationResult] = None
                if self.validator is not None:
                    try:
                        vr = self.validator.validate(gr)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Genie validator raised on step %s: %s",
                            step.step_id, exc,
                        )
                        vr = None

                if vr is not None and not vr.passed and not is_last:
                    last_error = self.validator.build_refinement_hint(vr) if self.validator else "Validation failed"
                    continue

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
                contract_satisfied = bool(vr and vr.passed)
                violations = list(vr.violations) if vr else []

                if sufficient or is_last:
                    if vr is not None and not vr.passed:
                        final_status = StepStatus.PARTIAL
                        final_err = (
                            "Validation failed after retries: "
                            + "; ".join(violations[:3])
                        )
                    elif sufficient:
                        final_status = StepStatus.SUCCESS
                        final_err = ""
                    else:
                        final_status = StepStatus.PARTIAL
                        final_err = "Empty result set"
                    return StepResult(
                        step_id=step.step_id, dimension=step.dimension,
                        status=final_status,
                        display_table=display_table, table_summary=summary,
                        sql=gr.sql, genie_trace_id=gr.genie_trace_id,
                        iterations_used=iterations_used,
                        error_message=final_err,
                        contract_satisfied=contract_satisfied,
                        validation_violations=violations,
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
        # When the prior attempt was rejected by the SQL guard (the LLM emitted
        # SQL-flavored English like "...where the header is..."), or when Genie
        # itself failed on a complex multi-condition question, switch the
        # refinement into "simplify drastically" mode.
        last_lower = (last_error or "").lower()
        sql_rejection = "sql detected" in last_lower
        genie_failed = (not sql_rejection) and bool(last_error)
        simplify_hint = ""
        if sql_rejection:
            simplify_hint = (
                "CRITICAL: the previous attempt was REJECTED before reaching "
                "the data tool because it contained SQL-flavored phrasing. "
                "Rewrite as a SHORTER plain-English question. Do NOT use the "
                "words 'where', 'with', 'having', 'such that', 'select', "
                "'from', 'order by', 'group by'. Phrase filters with 'for', "
                "'in', 'on', or 'about' instead "
                "(e.g. 'WhatsApp campaigns for the last 3 months with text "
                "headers' becomes 'list WhatsApp campaigns from the last 3 "
                "months and include the header type for each')."
            )
        elif genie_failed:
            simplify_hint = (
                "The previous question failed at the data tool. Simplify it: "
                "drop optional filters and metrics, keep only the most "
                "essential dimension and one or two core metrics. If the "
                "original question combined a filter and a list (e.g. "
                "'campaigns where header is text'), ask for the broader list "
                "first and surface the filtering attribute as a column."
            )
        domain_note_parts: list[str] = []
        try:
            thresholds = self.domain_knowledge.get_minimum_volume_thresholds()
            domain_note_parts.append(f"Minimum volume thresholds: {thresholds}")
        except Exception:  # noqa: BLE001
            pass
        if last_error:
            domain_note_parts.append(f"Previous attempt error: {last_error}")
        if simplify_hint:
            domain_note_parts.append(simplify_hint)
        domain_note = "\n".join(domain_note_parts)

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
