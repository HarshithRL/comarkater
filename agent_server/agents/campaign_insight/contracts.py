"""Dataclass contracts for the Campaign Insight Agent subgraph."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DimensionRole(Enum):
    """Role a dimension plays in an insight analysis."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"
    SCOPE_ONLY = "scope_only"
    NONE = "none"


class StepStatus(Enum):
    """Outcome of a single plan step."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"


class AgentStatus(Enum):
    """Overall outcome of a Campaign Insight Agent run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class GenieResponse:
    """Raw response shape returned from a Genie query."""

    columns: list[dict] = field(default_factory=list)
    data_array: list[list] = field(default_factory=list)
    row_count: int = 0
    sql: str = ""
    status: str = ""
    genie_trace_id: str = ""
    error_message: str = ""


@dataclass
class TableSummary:
    """Summarized view of a result table for LLM consumption.

    ``mode`` is ``"full"`` when the entire table fits under the compression
    threshold and ``"analyzed"`` when only aggregated/truncated views are
    retained.
    """

    mode: str = "full"
    row_count: int = 0
    schema: list[dict] = field(default_factory=list)
    statistical_summary: dict = field(default_factory=dict)
    categorical_distribution: dict = field(default_factory=dict)
    anomalies: list[dict] = field(default_factory=list)
    top_rows: list[list] = field(default_factory=list)
    bottom_rows: list[list] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)
    full_data: Optional[list[list]] = None


@dataclass
class DisplayTable:
    """Table shape rendered to the end user."""

    title: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    source_sql: str = ""


@dataclass
class DimensionConfig:
    """Per-dimension role and query budget."""

    role: DimensionRole = DimensionRole.NONE
    budget: int = 0


@dataclass
class DimensionClassification:
    """Dimension roles chosen for a given user query."""

    primary_analysis: str = ""
    channel: str = ""
    campaign: DimensionConfig = field(default_factory=DimensionConfig)
    audience: DimensionConfig = field(default_factory=DimensionConfig)
    content: DimensionConfig = field(default_factory=DimensionConfig)

    @property
    def total_budget(self) -> int:
        return self.campaign.budget + self.audience.budget + self.content.budget

    @property
    def active_dimensions(self) -> list[str]:
        result: list[str] = []
        for name in ("campaign", "audience", "content"):
            cfg: DimensionConfig = getattr(self, name)
            if cfg.role is not DimensionRole.NONE:
                result.append(name)
        return result


@dataclass
class PlanStep:
    """One step in an adaptive execution plan."""

    step_id: int = 0
    dimension: str = ""
    query: str = ""
    purpose: str = ""
    depends_on: list[int] = field(default_factory=list)
    can_parallel: bool = True


@dataclass
class ExecutionPlan:
    """Ordered set of plan steps with dependency metadata."""

    steps: list[PlanStep] = field(default_factory=list)
    total_budget: int = 0

    def get_ready_steps(self, completed: set[int]) -> list[PlanStep]:
        """Return steps not yet completed whose dependencies are all satisfied."""
        ready: list[PlanStep] = []
        for step in self.steps:
            if step.step_id in completed:
                continue
            if all(dep in completed for dep in step.depends_on):
                ready.append(step)
        return ready


@dataclass
class StepResult:
    """Outcome of executing a single plan step."""

    step_id: int = 0
    dimension: str = ""
    status: StepStatus = StepStatus.SUCCESS
    display_table: Optional[DisplayTable] = None
    table_summary: Optional[TableSummary] = None
    sql: str = ""
    genie_trace_id: str = ""
    iterations_used: int = 0
    error_message: str = ""


@dataclass
class PatternMatch:
    """A known analytical pattern detected in the data."""

    name: str = ""
    description: str = ""
    likely_causes: list[str] = field(default_factory=list)
    severity: str = ""


@dataclass
class Interpretation:
    """Narrative interpretation of the analysis."""

    summary: str = ""
    insights: list[str] = field(default_factory=list)
    patterns: list[PatternMatch] = field(default_factory=list)
    severity: str = ""


@dataclass
class Recommendation:
    """A single actionable recommendation tied to evidence."""

    action: str = ""
    detail: str = ""
    expected_impact: str = ""
    evidence: str = ""
    source_pattern: str = ""


@dataclass
class VerificationResult:
    """Output of the reflector over an interpretation + recommendations."""

    passed: bool = True
    issues_found: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    interpretation: Optional[Interpretation] = None
    recommendations: list[Recommendation] = field(default_factory=list)


@dataclass
class SubagentInput:
    """Input envelope passed into the Campaign Insight Agent."""

    request_id: str = ""
    query: str = ""
    intent: dict = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


@dataclass
class SubagentOutput:
    """Output envelope returned from the Campaign Insight Agent."""

    request_id: str = ""
    status: AgentStatus = AgentStatus.SUCCESS
    execution_summary: dict = field(default_factory=dict)
    tables_for_display: list[DisplayTable] = field(default_factory=list)
    chart_spec: Optional[dict] = None
    interpretation: Optional[Interpretation] = None
    recommendations: list[Recommendation] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
