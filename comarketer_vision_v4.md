# CoMarketer — Complete Implementation Vision (v4.0)
# Domain: Marketing Analytics | Client: Netcore Cloud
# Last Updated: 2026-04-14

---

## 1. System Identity

CoMarketer is a **conversational campaign intelligence agent** for Netcore Cloud's
marketing platform. It enables brand marketing teams (IGP, Pepe Jeans, Crocs, and
future clients) to query campaign performance, audience composition, and content
quality in natural language and receive:

- **Deterministic data tables** (100% accurate, no LLM computation)
- **LLM-narrated insights and recommendations** (grounded in data evidence)
- **Highcharts visualizations** (chart specs generated from structured data)
- **Persistent memory** of brand profile and conversation history

**Non-negotiable constraint:** All numeric values flow from Genie SQL → programmatic
table build. The LLM never computes, invents, or modifies metric values.


---

## 2. Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent Framework | LangGraph 0.3+ | Agentic graph execution, Command routing, ReAct subgraph |
| Agent Serving | MLflow 3 ResponsesAgent | Agent endpoint wrapper, tracing, evaluation |
| Agent Deployment | Databricks Apps (FastAPI) | Production endpoint, SSE streaming, UI |
| Data Access | Databricks Genie REST API | NL-to-SQL over metric views |
| Data Governance | Unity Catalog + RLS | Row-level security per client (cid column) |
| Data Store | Metric Views (YAML v1.1) | Semantic layer over campaign/audience/content |
| Memory (STM) | Lakebase CheckpointSaver | Per-thread conversation persistence |
| Memory (LTM) | Lakebase DatabricksStore | Per-client profile + episode history |
| Tracing | MLflow 3 autolog + manual spans | End-to-end observability |
| Evaluation | MLflow 3 Evaluation | Offline + online quality measurement |
| Charts | Highcharts 11.x | Client-side visualization rendering |


---

## 3. Architecture — Two-Layer Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     CoMarketer (Supervisor)                      │
│              "The Marketer's Intelligent Assistant"               │
│                                                                  │
│  Model: databricks-claude-sonnet-4-5                             │
│  Role: Domain-aware orchestrator                                 │
│  Identity: Marketing expert who understands the domain broadly   │
│            but delegates specialized analysis to subagents        │
│                                                                  │
│  Domain Awareness: YES — broad marketing domain understanding    │
│  Specialized Analysis: NO — never computes, never interprets     │
│                         data directly                             │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                        Structured Contract (JSON)
                                    │
┌───────────────────────────────────▼─────────────────────────────┐
│              Campaign Insight Agent (Subagent)                   │
│              "The Specialized Campaign Analyst"                   │
│                                                                  │
│  Model: databricks-gpt-5-2                                       │
│  Role: Deep campaign/audience/content analysis expert            │
│  Pattern: Adaptive Plan → ReAct Execute → Reflect Verify         │
│                                                                  │
│  Domain Awareness: YES — deep specialized campaign analytics     │
│  Capabilities: Long-running, multi-step, parallel execution,     │
│    large table analysis, Highcharts spec generation,              │
│    data-grounded recommendations, intermediate streaming          │
│                                                                  │
│  Data Access: Single Genie Space with 4 metric view objects       │
│  Dimensions: Campaign (always on) + Audience + Content            │
└─────────────────────────────────────────────────────────────────┘
```


---

## 4. Data Layer — Genie Space

### 4.1 Genie Space: "CoMarketer Campaign Intelligence"

Single Genie Space. One space for all three analytical dimensions.
Primary consumer is the CoMarketer agent via API, not end users.

| Object | Type | Purpose |
|---|---|---|
| `campaign_details_metric_view_v2` | Metric View | Delivery counts, engagement rates, revenue, trends, funnel analysis |
| `audience_metric_view` | Metric View | Intent scores, communication health, app status, retarget frequency |
| `campaign_content_metric_view` | Metric View | Content quality scores, emotions, CTA, templates, content features |
| `igp_content_insights` | Table | Content benchmarks — optimal values at client/vertical/global level |

### 4.2 What Metric Views Do for the Agent

Metric views handle JSONB flattening, formula computation (try_divide), and
semantic naming. The agent sends natural language queries to Genie, Genie generates
MEASURE()-based SQL, and returns flat columnar results.

**The agent never:**
- Parses JSONB
- Constructs SQL
- Computes derived metrics (CTR, CVR, etc.)
- Accesses raw source tables

**The agent receives:**
- Clean flat columns (avg_sends_high_intent, click_through_rate, primary_emotion)
- Pre-computed rates with try_divide for division safety
- Pre-flattened audience distributions as percentage columns
- Content features extracted from JSONB as named dimensions

### 4.3 Genie Space Configuration

**Text Instructions:** 3,449 characters covering default behavior (last 30 days,
formatting rules), channel limitations (SMS/WhatsApp no opens, BPN low CVR tracking),
audience/content summary rules, timestamps (UTC), out of scope handling with
alternatives, clarification behavior, and summary instructions.

**SQL Expressions:** 46 total (27 new for audience/content + existing campaign filters)
**Example SQL Queries:** 29 total (8 existing + 21 new covering all dimensions)
**Table Descriptions:** Per-table routing guidance in knowledge store

### 4.4 Genie Tool Interface

```
Agent sends NL query
    │
    ▼
Genie REST API
    ├── start_conversation(question) → conv_id, msg_id
    ├── poll check_status() with backoff → COMPLETED | FAILED | FEEDBACK
    └── fetch_result() → {columns, data_array, sql, status}
    │
    ▼
Agent receives structured response
    ├── columns: [{name, type}]
    ├── data_array: [[row1], [row2], ...]
    ├── sql: "SELECT MEASURE(...) FROM ..."
    ├── row_count: N
    └── status: success | error | feedback_needed
```


---

## 5. Supervisor — CoMarketer Main Agent

### 5.1 Identity & Domain Awareness

The Supervisor IS a marketing domain expert. It speaks the language of marketers,
understands campaign concepts, channels, metrics vocabulary, and business context.
It is NOT a generic router.

**What the Supervisor knows (broad domain awareness):**
- Platform & business context (Netcore Cloud, 5 channels, 5 campaign types, funnel)
- Metric vocabulary (names + what they mean, channel applicability)
- Intent taxonomy (11+ types with complexity classification)
- Out-of-scope questions and why (ROI needs cost data, no user-level data, etc.)
- Data dimensions available (campaign, audience, content)

**What the Supervisor does NOT know (delegated to subagent):**
- Metric thresholds (what is "good" vs "bad")
- Interpretation rules (what metric combinations mean)
- Diagnostic patterns (why metrics behave certain ways)
- Recommendation patterns (what actions to take)
- Statistical analysis of data tables
- Content scoring rules and benchmark comparisons

### 5.2 Supervisor Responsibilities

1. **Intent Classification** — LLM classification using domain vocabulary.
   Output: {intent_type, complexity, channels, metrics, time_context, target_agent}

2. **Smart Clarification** — Domain-aware questions, not generic ones.
   Can answer from STM without calling subagent for follow-up questions.

3. **Planning** (complex only) — Decompose into ordered sub-tasks using domain
   vocabulary. Plan is INFORMED by domain awareness, EXECUTED by subagent.

4. **Routing** — Registry-based agent selection. Intent → lookup → dispatch.

5. **Response Synthesis** — Assemble final response from subagent structured output.
   COMPOSITIONAL only — arrange, transition, tone. Never add claims not in output.

6. **Out-of-Scope Handling** — Direct response with explanation + alternatives.
   No subagent needed.

### 5.3 Supervisor Domain Knowledge Loading

```python
class SupervisorDomainContext:
    """Broad domain awareness — subset of YAMLs."""

    def __init__(self):
        # From domain_context.yaml — full load
        ctx = load_yaml("domain_context.yaml")
        self.platform = ctx["platform"]           # name, type, description
        self.channels = ctx["channels"]           # 5 channels with codes, tracking caps
        self.campaign_types = ctx["campaign_types"]  # 5 types with counts
        self.campaign_categories = ctx["campaign_categories"]  # 5 categories
        self.funnel = ctx["funnel"]               # 7 stages with drop reasons

        # From intent.yaml — full load (supervisor owns classification)
        self.intents = load_yaml("intent.yaml")["intents"]  # 11 intent types

        # From constraints.yaml — ONLY what supervisor needs
        c = load_yaml("constraints.yaml")
        self.cannot_answer = c["analysis_constraints"]["cannot_answer"]  # 6 topics
        self.channel_constraints = c["channel_constraints"]  # SMS no opens, BPN low CVR
        self.data_scope = c["data_constraints"]["scope"]  # campaign-level only
        # Does NOT load: metric_constraints, minimum_volume, content_constraints

        # From metrics.yaml — ONLY names and channel applicability
        m = load_yaml("metrics.yaml")
        self.base_metric_names = [x["name"] for x in m["base_metrics"]]  # 9 names
        self.derived_metric_names = [x["name"] for x in m["derived_metrics"]]  # 12 names
        self.metric_channel_map = {
            x["name"]: x.get("channel_applicability", {})
            for x in m["derived_metrics"]
        }
        # Does NOT load: formulas, thresholds, interpretation rules,
        # denominator rules, funnel_order

        # Does NOT load anything from: interpretation.yaml, recommendations.yaml
```

### 5.4 Supervisor Module Structure

```
supervisor/
  ├── __init__.py
  ├── supervisor_node.py        # LangGraph node entry point
  ├── intent_classifier.py      # Intent + complexity classification
  ├── planner.py                # Complex query decomposition
  ├── router.py                 # Registry-based agent selection
  ├── synthesizer.py            # Final response assembly
  ├── domain_context.py         # SupervisorDomainContext loader (subset)
  └── prompts/
      └── supervisor_prompt.py  # ~150 lines — orchestration + domain vocabulary
```

### 5.5 Supervisor Prompt Structure (~150 lines target)

```
SECTION 1: Identity (10 lines)
SECTION 2: Domain Vocabulary (30 lines) — injected from YAMLs
SECTION 3: Intent Classification Rules (30 lines) — from intent.yaml
SECTION 4: Out-of-Scope Rules (20 lines) — from constraints.yaml
SECTION 5: Planning Guidelines (20 lines)
SECTION 6: Synthesis Instructions (30 lines) — output format spec
SECTION 7: Guardrails (10 lines) — never compute, never interpret, never add
```


---

## 6. Subagent — Campaign Insight Agent

### 6.1 Identity & Specialization

The Campaign Insight Agent is a **specialized deep analyst** that covers three
analytical dimensions within a single agent:

- **Campaign Performance** — delivery, engagement, revenue, trends, funnel
- **Audience Composition** — intent, communication health, app status, retarget
- **Content Quality** — emotions, CTA, templates, scores, benchmarks

One agent, three internal modules, one Genie Space.

### 6.2 Architecture: Adaptive Plan → ReAct Execute → Reflect Verify

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAMPAIGN INSIGHT AGENT                         │
│              Pattern: AP → ReAct → Reflect                       │
│              Model: databricks-gpt-5-2                           │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: ADAPTIVE PLAN                                    │   │
│  │                                                            │   │
│  │  Phase 1A: Dimension Classification (LLM call)            │   │
│  │    → Determine primary_analysis (campaign|audience|        │   │
│  │      content|mixed)                                        │   │
│  │    → Set active_dimensions with roles and query budgets    │   │
│  │    → Total query budget cap: 8                             │   │
│  │                                                            │   │
│  │  Phase 1B: Deterministic Validation & Clamping             │   │
│  │    → Feature flag gates (ENABLE_AUDIENCE, ENABLE_CONTENT)  │   │
│  │    → Audience safety check (no keywords → force OFF)       │   │
│  │    → Content-primary guard (no campaign cross-pollution)   │   │
│  │    → Budget scaling (trim supporting dimensions first)     │   │
│  │                                                            │   │
│  │  Phase 1C: Per-Dimension Query Planning                    │   │
│  │    → Mark dependencies and parallelism                     │   │
│  │    → If supervisor sent plan: validate + refine            │   │
│  │    → If no plan (simple/medium): generate minimal plan     │   │
│  │                                                            │   │
│  │  → Stream: plan_ready event                                │   │
│  │  → Trace: plan with step count + dependencies              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: ReAct EXECUTE (per step, controlled loop)        │   │
│  │                                                            │   │
│  │  For each step in plan:                                    │   │
│  │    REASON → ACT (Genie call) → OBSERVE → EVALUATE          │   │
│  │                                                            │   │
│  │    OBSERVE includes:                                        │   │
│  │    → If table ≤20 rows: direct to LLM                      │   │
│  │    → If table 21-50 rows: table_analyzer summary to LLM,   │   │
│  │      full table to table_builder                            │   │
│  │    → If table >50 rows: analyzer summary + truncated to    │   │
│  │      LLM, full table via SQL item                           │   │
│  │                                                            │   │
│  │  Parallel execution for independent steps (staggered)      │   │
│  │                                                            │   │
│  │  LOOP CONTROLS:                                            │   │
│  │    max_iterations_per_step: 3                               │   │
│  │    max_genie_retries: 2                                     │   │
│  │    total_timeout: 120s                                      │   │
│  │    on_max_reached: partial result + explain gap             │   │
│  │    on_genie_failure: include error, don't crash             │   │
│  │                                                            │   │
│  │  → Stream: step_progress + intermediate data per step       │   │
│  │  → Trace: per-step (query, SQL, rows, latency)              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: INTERPRET + RECOMMEND                            │   │
│  │                                                            │   │
│  │  INTERPRETATION (interpreter.py + domain knowledge):       │   │
│  │    1. Single-metric evaluation (thresholds from YAML)       │   │
│  │    2. Combination pattern detection (6 diagnostic patterns) │   │
│  │    3. Trend interpretation (direction + polarity)            │   │
│  │    4. Cross-dimension synthesis (campaign + audience +       │   │
│  │       content signals combined)                              │   │
│  │                                                            │   │
│  │  RECOMMENDATION (recommender.py + domain knowledge):       │   │
│  │    → Only for diagnostic/strategic intents                  │   │
│  │    → Pattern → recommendation matching                      │   │
│  │    → Each: action + detail + expected_impact + evidence      │   │
│  │    → Channel-specific and cross-dimension recommendations   │   │
│  │                                                            │   │
│  │  → Trace: patterns detected, recommendations made           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: REFLECT (lightweight verification)               │   │
│  │                                                            │   │
│  │  Single LLM call — NOT iterative:                           │   │
│  │  CHECK 1: Data grounding (every claim references data?)     │   │
│  │  CHECK 2: Constraint compliance (channel caveats applied?)  │   │
│  │  CHECK 3: Recommendation grounding (evidence cited?)        │   │
│  │  CHECK 4: Completeness (all parts of question addressed?)   │   │
│  │                                                            │   │
│  │  If issues: fix inline. If none: proceed unchanged.         │   │
│  │  → Trace: verification result + fixes applied               │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  PHASE 5: BUILD OUTPUT                                     │   │
│  │                                                            │   │
│  │  1. tables_for_display (table_builder — deterministic)      │   │
│  │  2. chart_spec (Highcharts 11.x from structured data)       │   │
│  │  3. interpretation (summary + insights + patterns)          │   │
│  │  4. recommendations (action + detail + impact + evidence)   │   │
│  │  5. caveats (channel limitations, data gaps)                │   │
│  │  6. metadata (genie_calls, confidence, data_freshness)      │   │
│  │                                                            │   │
│  │  → Trace: final output structure                            │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 3-Dimension Classification System

Every user query activates one or more data dimensions:

```
Phase 1A: Dimension Classification (LLM call)
  Input: user query + intent
  Output:
    primary_analysis: campaign | audience | content | mixed
    active_dimensions:
      campaign:  {role: primary|supporting|scope_only, budget: N}
      audience:  {role: primary|supporting|none, budget: N}
      content:   {role: primary|supporting|none, budget: N}

Phase 1B: Deterministic Validation
  → Feature flags: ENABLE_AUDIENCE_ANALYSIS, ENABLE_CONTENT_ANALYSIS
  → Audience gating: requires explicit keywords (segment, intent,
    who clicked, lifecycle) — not activated just because campaigns
    have audiences
  → Content-primary guard: when content is primary, campaign queries
    must NOT compute content scores, filter by template_type, or
    search campaign_name for content features
  → Budget clamping: total across all dimensions ≤ 8
    If exceeded: scale supporting dimensions first
```

### 6.4 Domain Knowledge — 6 Existing + 2 Needed YAML Files

**6 files exist today (1,335 lines total). 2 additional files needed for
audience and content dimensions.**

```
LAYER 1: WHAT EXISTS         → domain_context.yaml (222 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Platform: Netcore Cloud (MarTech, 6,500+ brands)
  Lifecycle focus: Engagement stage

  Channels (5):
    email   — tracks opens/clicks/conversions, volume: high, typical CTR 1-3%
    sms     — NO opens, tracks clicks, volume: low, typical CTR 0.1-1%
    whatsapp — NO opens, tracks clicks, volume: medium, typical CTR 2-5%
    apn     — tracks opens/clicks, volume: high, typical CTR 0.2-0.6%
    bpn     — tracks opens/clicks, ~4% CVR coverage, ~12% suppression

  Campaign Types (5): Regular (5985), STO (838), Split/AB (766),
    AMP (167), Rate Limited (55)

  Campaign Categories (5): New Arrival (4681), Discount (1842),
    Product Recommendation (490), Content Engagement (359), Others (439)

  Funnel (7 stages): Published → Sent → Delivered → Opened → Clicked
    → Conversion → Revenue (with drop reasons at each stage)

  Data Source: campaign_details table + metric view
    ⚠️ GAP: Needs updating to include audience_metric_view,
    content_metric_view (campaign_content_metric_view), igp_content_insights


LAYER 2: WHAT TO MEASURE     → metrics.yaml (257 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Base Metrics (9):
    published, sent, delivered, opened, clicked, conversion,
    revenue, bounce, unsubscribe

  Derived Metrics (12):
    Delivery:    delivery_rate, bounce_rate, suppression_rate
    Engagement:  open_rate, click_through_rate, click_to_open_rate,
                 unsubscribe_rate
    Conversion:  conversion_rate, click_to_conversion_rate
    Revenue:     revenue_per_sent, revenue_per_delivered,
                 revenue_per_conversion

  Each derived metric includes: exact formula, numerator, denominator,
  unit, interpretation thresholds, channel applicability, notes

  Metric Dependencies: explicit dependency graph
  Funnel Order: published → sent → delivered → opened → clicked
    → conversion → revenue

  Key Rules:
    - CTR denominator = delivered (NOT sent, NOT opened)
    - Two CVR variants: delivery-based (default) vs click-based
    - revenue_per_sent is the ONLY metric using sent as denominator
    - Division safety via try_divide() / NULLIF


LAYER 3: WHAT IT MEANS       → interpretation.yaml (266 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Single-Metric Thresholds (6 metrics):
    delivery_rate:  excellent ≥98, good ≥95, warning ≥90, critical <90
    open_rate:      excellent ≥35, good ≥20, warning ≥10, critical <10
    CTR:            excellent ≥5, good ≥2, average ≥1, warning ≥0.5, critical <0.5
    unsubscribe:    good <0.2, warning 0.2-0.5, critical ≥0.5
    conversion:     good ≥0.5, average 0.1-0.5, low <0.1
    bounce_rate:    good <2, warning 2-5, critical ≥5

    Each includes: low/high value causes, channel exceptions

  Combination Diagnostic Patterns (6):
    1. engagement_without_conversion — high CTR, low post-click CVR
    2. opens_without_clicks — high open rate, low CTR
    3. delivery_without_opens — high delivery, low opens
    4. volume_without_delivery — high sent, low delivery rate
    5. revenue_without_volume — high rev/sent, low volume
    6. high_unsubscribe_with_low_engagement — fatigue signal

    Each includes: condition, interpretation, likely_causes, recommendations

  Trend Rules (5 patterns):
    increasing, decreasing, spike, drop, stable
    Polarity-aware: increasing is good for CTR, bad for bounce_rate

  Channel-Specific Rules (5 channels):
    Each with: strengths, typical metrics, key metric, limitations


LAYER 4: WHAT TO DO          → recommendations.yaml (233 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Core Principle: "Every recommendation must be grounded in data evidence"

  Trigger-Based Patterns (6):
    delivery_rate < 90%    → clean lists, check reputation, review tokens
    open_rate < 15%        → test subjects, optimize send time, review sender
    CTR < 0.5%             → improve CTA, reduce length, improve targeting
    click_to_CVR < 1%      → review landing page, simplify flow, check mobile
    unsubscribe > 0.5%     → reduce frequency, segment by engagement
    revenue/sent declining → review audience quality, test offers, optimize mix

    Each includes: severity, diagnosis, multiple actions with expected_impact

  Channel-Specific Recommendations (5 channels):
    Each with: general best practices + when_underperforming actions

  Comparative Recommendations:
    channel_comparison: comparable metrics only + objective mapping
      (awareness→email+APN, engagement→email+WA, conversion→WA+email)
    type_comparison: STO already AI-optimized, split within parent group
    time_comparison: account for seasonality, like-for-like periods

  Response Template:
    summary → data → insight → recommendation → caveat
    Rules: lead with answer, business language, recommendations only
    when warranted, caveats only when applicable


LAYER 5: WHAT NOT TO DO      → constraints.yaml (168 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Data Constraints:
    Scope: campaign-level aggregates only (no user-level)
    Identity: unique key = mid + channel + campaign_name
    Time: UTC, Feb 2025 onwards
    Revenue: no currency symbols, zero may mean tracking off
    RLS: cid column, hidden from users

  Metric Computation Constraints:
    Deterministic rule: LLM NEVER computes derived metrics
    Denominator rules: delivered for rates, sent for revenue_per_sent
    Division safety: try_divide() or NULLIF always
    Minimum volume: ≥100 delivered for general, ≥1000 for rankings

  Channel Constraints:
    SMS/WhatsApp: no opens (never report open rate)
    BPN: ~4% conversion coverage (always caveat)
    Cross-channel: always note incomparable metrics

  Content Constraints:
    Subject lines: email only (push_title for APN/BPN, NULL for SMS/WA)
    Array columns: use array_contains() for tags/segments/lists

  Analysis Constraints — Cannot Answer (6):
    ROI (no cost), user-level (aggregates only), CTA placement (not stored),
    audience quality (no demographics), multi-touch attribution (no journey),
    predictive (no model) — each with alternatives


LAYER 6: HOW TO RESPOND      → intent.yaml (189 lines) ✅ EXISTS
─────────────────────────────────────────────────────────────────
  Intent Taxonomy (11 types):

  Simple:
    performance_lookup — metric lookup for campaigns/channels/time
    ranking — top/bottom N by metric (with volume filters)

  Medium:
    comparison — cross-category (channels, types, categories)
    trend_analysis — daily/weekly/monthly/quarterly/hourly
    funnel_analysis — published→conversion with drop-off rates
    time_optimization — best send times (UTC caveat, STO note)
    content_analysis — subject lines, titles, categories
    ab_test_analysis — split variants by parent_id

  Complex:
    diagnostic — metric mismatch patterns (5 built-in diagnostic patterns)
    strategic_recommendation — multi-dimension, data-grounded actions

  Terminal:
    out_of_scope — 6 specific topics with explanations + alternatives

  Each intent includes: examples, response_pattern, constraints


LAYER 7: AUDIENCE KNOWLEDGE   → audience_knowledge.yaml ❌ NEEDED
─────────────────────────────────────────────────────────────────
  Must define:
    - Audience tag categories (intent, communication health, app status,
      retarget frequency) with interpretation rules
    - Sends vs conversions comparison patterns
    - Intent lift interpretation thresholds
    - Channel-specific audience health rules
    - Proactive nudge trigger conditions
    - Audience-specific recommendation patterns


LAYER 8: CONTENT KNOWLEDGE    → content_knowledge.yaml ❌ NEEDED
─────────────────────────────────────────────────────────────────
  Must define:
    - Content feature definitions (emotion, CTA, value prop, intent clarity)
    - Content score interpretation (open/click/conversion scores)
    - Benchmark comparison rules (client vs vertical vs global)
    - Channel-specific content patterns (email subject vs WA body)
    - Template type interpretation
    - Default to click metric unless user specifies otherwise
    - Content-specific recommendation patterns
```

### 6.5 Large Table Handling — Table Analyzer

```
Genie returns structured table
         │
    ≤20 rows: Full table → LLM + table_builder
         │
    21+ rows: table_analyzer.py (deterministic Python, NO LLM)
         │
         ├── Schema extraction (columns, types, row count)
         ├── Statistical summary (min/max/mean/median/std per numeric)
         ├── Categorical distribution (top-N by frequency + metric)
         ├── Anomaly detection (outliers, unexpected zeros, nulls)
         ├── Slice extraction (top 5, bottom 5, anomalies, trend endpoints)
         └── Pre-computed aggregates (group-by, period-over-period, rollups)
         │
         ├── Summary → LLM (for reasoning)
         └── Full table → table_builder (for deterministic display)
```

### 6.6 Highcharts Spec Generation

Charts generated by subagent from structured data. Chart data points extracted
deterministically; LLM decides chart type, layout, annotations.

```
WHEN TO GENERATE:
  trend_analysis      → Line chart
  comparison          → Bar chart
  funnel_analysis     → Funnel chart
  ranking             → Horizontal bar
  diagnostic (trend)  → Line with annotations

WHEN NOT TO:
  performance_lookup (single number), clarification, out_of_scope

RULES:
  1. Data from SAME Genie results as tables (no separate fetch)
  2. LLM generates Highcharts 11.x spec structure
  3. Data points extracted deterministically
  4. Supervisor passes through unchanged — never modifies
```

### 6.7 Subagent Module Structure

```
agents/
  └── campaign_insight/
      ├── __init__.py
      ├── agent.py               # Entry point — orchestrates 5 phases
      ├── dimension_classifier.py # Phase 1A: LLM dimension classification
      ├── dimension_validator.py  # Phase 1B: Deterministic validation/clamping
      ├── adaptive_planner.py    # Phase 1C: Query planning with dependencies
      ├── executor.py            # Phase 2: ReAct loop controller
      ├── tool_handler.py        # Genie REST integration (submit→poll→fetch)
      ├── table_analyzer.py      # Large table processing (deterministic)
      ├── table_builder.py       # Deterministic table construction for display
      ├── interpreter.py         # Phase 3a: Domain reasoning
      ├── recommender.py         # Phase 3b: Action generation
      ├── reflector.py           # Phase 4: Single-pass verification
      ├── chart_builder.py       # Highcharts 11.x spec generation
      ├── output_builder.py      # Phase 5: Structured output assembly
      ├── domain_knowledge.py    # YAML loader with lookup methods
      ├── prompts/
      │   ├── insight_prompt.py
      │   ├── reasoning_prompt.py
      │   ├── interpretation_prompt.py
      │   └── reflection_prompt.py
      └── domain_knowledge/
          ├── metrics.yaml            # 257 lines — 9 base + 12 derived metrics
          ├── interpretation.yaml     # 266 lines — thresholds + 6 diagnostic patterns
          ├── constraints.yaml        # 168 lines — data/metric/channel constraints
          ├── recommendations.yaml    # 233 lines — trigger patterns + channel recs
          ├── intent.yaml             # 189 lines — 11 intent types
          ├── domain_context.yaml     # 222 lines — platform, channels, funnel
          ├── audience_knowledge.yaml  # ❌ TODO — audience tag interpretation
          └── content_knowledge.yaml   # ❌ TODO — content score interpretation
```

### 6.8 Context Budget & Preservation

```
CONTEXT BUDGET (databricks-gpt-5-2):

  Prompt (domain knowledge + instructions):    ~3,000 tokens
  Contract from supervisor:                     ~500 tokens
  Per Genie step (query + SQL + table summary): ~800-1,500 tokens
  Interpretation + recommendations:             ~1,000 tokens
  Reflection check:                             ~500 tokens
  Output generation:                            ~1,000 tokens

  4-step complex query ≈ 10,800 tokens — well within limits.
  table_analyzer keeps each step to ~800-1,500 tokens regardless of table size.

CONTEXT PRESERVATION:
  1. Step results accumulated in structured state (not raw history)
  2. Step N receives: full domain knowledge + compressed steps 1..N-1 + full step N
  3. Sliding window ensures current step has full detail

TIMEOUT & DEGRADATION:
  Total: 120s | Per-step: 30s
  On timeout: complete current step → skip remaining → interpret available →
  return partial with explicit note → trace timeout event
```


---

## 7. Supervisor ↔ Subagent Contract

### 7.1 Input Contract (Supervisor → Subagent)

```json
{
  "request_id": "uuid",
  "query": "Why did email performance drop last week?",
  "intent": {
    "type": "diagnostic",
    "complexity": "complex",
    "channels_mentioned": ["email"],
    "metrics_mentioned": [],
    "time_context": "last_week"
  },
  "plan": [
    "Get overall email metrics trend for last 2 weeks",
    "Breakdown performance by campaign category",
    "Identify top declining campaigns by CTR",
    "Check delivery rate for potential issues"
  ],
  "context": {
    "client_id": "igp",
    "sp_token": "...",
    "genie_space_id": "01f0ead8...2",
    "conversation_history": ["...last 7 messages (trimmed)..."],
    "ltm_profile": {"brand": "IGP", "primary_channels": ["email", "whatsapp"]},
    "ltm_episodes": ["Last week user asked about WhatsApp revenue..."]
  },
  "config": {
    "max_genie_calls": 8,
    "max_iterations_per_step": 3,
    "timeout_seconds": 120,
    "chart_enabled": true,
    "enable_audience": true,
    "enable_content": true
  }
}
```

### 7.2 Output Contract (Subagent → Supervisor)

```json
{
  "request_id": "uuid",
  "status": "success | partial | error",
  "execution_summary": {
    "steps_planned": 4,
    "steps_completed": 4,
    "genie_calls_made": 4,
    "dimensions_activated": ["campaign"],
    "total_duration_ms": 8500,
    "pattern_detected": "opens_without_clicks"
  },
  "results": {
    "tables_for_display": [
      {
        "title": "Email Performance — Last 2 Weeks",
        "columns": ["Week", "Sent", "Delivered", "CTR", "Revenue"],
        "rows": [["W1 Apr", 45000, 43200, "2.1%", 23400]],
        "source_sql": "SELECT ..."
      }
    ],
    "chart_spec": {
      "highcharts_options": { "chart": {"type": "line"}, "..." : "..." }
    }
  },
  "interpretation": {
    "summary": "Email CTR dropped 23% week-over-week",
    "pattern": { "name": "opens_without_clicks", "likely_causes": ["..."] },
    "insights": ["Open rate stable at 30%", "CTR dropped to 1.6%"],
    "severity": "warning"
  },
  "recommendations": [
    {
      "action": "Improve CTA visibility in Discount Promotion campaigns",
      "detail": "...",
      "expected_impact": "20-50% lift in CTR",
      "evidence": "Open rate is 30% but CTR only 1.6%",
      "source_pattern": "opens_without_clicks"
    }
  ],
  "caveats": ["Analysis covers email channel only"],
  "metadata": {
    "confidence": "high",
    "data_freshness": "2026-04-12",
    "model": "databricks-gpt-5-2",
    "reflection_issues_found": 0,
    "trace_id": "mlflow-trace-uuid"
  }
}
```


---

## 8. LangGraph Topology

```
                    START
                      │
                      ▼
            ┌─────────────────┐
            │   supervisor     │  (databricks-claude-sonnet-4-5)
            │   classify       │
            └────────┬────────┘
                     │
         ┌───────────┼────────────┬──────────────┐
         ▼           ▼            ▼              ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐
    │greeting │ │clarify   │ │out_of_   │ │campaign_      │
    │(no LLM) │ │(optional │ │scope     │ │insight_agent  │
    │         │ │ LLM)     │ │(no LLM)  │ │(databricks-   │
    └────┬────┘ └────┬─────┘ └────┬─────┘ │ gpt-5-2)      │
         │           │            │        │ Plan→ReAct→   │
         │           │            │        │ Reflect→Build │
         │           │            │        └───────┬───────┘
         │           │            │                │
         │           │            │                ▼
         │           │            │        ┌───────────────┐
         │           │            │        │  supervisor    │
         │           │            │        │  synthesize    │
         │           │            │        └───────┬───────┘
         │           │            │                │
         └───────────┴────────────┴────────────────┘
                                  │
                                  ▼
                                 END
```

Node count: 6 (down from 10+ in current implementation).
Graph builder reads from AGENT_REGISTRY — adding a new subagent = register + module.


---

## 9. Pluggable Agent Registry

### 9.1 Registry Structure

```python
@dataclass
class AgentRegistration:
    name: str
    module_path: str                    # "agents.campaign_insight.agent"
    class_name: str                     # "CampaignInsightAgent"
    handles_intents: List[str]          # Intent types this agent handles
    genie_space_config_key: str         # Config key for Genie Space ID
    domain_knowledge_path: str          # Path to YAML files
    model: str                          # LLM model
    max_parallel_calls: int = 6         # Genie concurrency
    max_iterations_per_step: int = 3    # ReAct loop cap
    timeout_seconds: int = 120          # Total timeout
    chart_capable: bool = True
    tools: List[str] = ["genie_search"] # Registered tool names
    description: str = ""
```

### 9.2 Adding a New Agent

```
1. Create agents/new_agent/ with standard structure
2. Add domain knowledge YAMLs
3. Register in AGENT_REGISTRY with intent mappings
4. Done — no graph.py edits, no supervisor prompt changes, no routing changes
```


---

## 10. Memory Architecture

### 10.1 Four-Layer Memory

| Layer | Storage | Scope | Used By |
|---|---|---|---|
| Working | In-memory (AgentState) | Current request | Both |
| STM | Lakebase CheckpointSaver | Per thread/session | Supervisor (classification) + Subagent |
| LTM Profile | Lakebase DatabricksStore | Per client | Subagent only |
| LTM Episodes | Lakebase DatabricksStore | Per client | Subagent only (keyword search) |
| Fallback | InMemorySaver | Per session | Both (must surface degradation) |

### 10.2 Memory Injection

```
Supervisor receives: STM (last 7 messages) for classification context only
Subagent receives: LTM profile + relevant episodes + STM for reasoning

On Lakebase failure: degrade to InMemorySaver + warn user
  "Using session memory only — history may not persist"
```

### 10.3 Memory Policy (to be defined)

- What gets stored in LTM profile vs episodes
- When episodes get created (every turn vs meaningful interactions)
- Decay rules for old episodes
- Injection strategy (system context vs user message)


---

## 11. Streaming — SSE Events

```
1. acknowledgment    → "Working on it…" (immediate)
2. plan_ready        → Plan steps (complex queries only)
3. step_progress     → "Analyzing CTR trends..." (per Genie call)
4. data_table        → Deterministic table data (mid-stream)
5. chart             → Highcharts spec (if applicable)
6. analysis          → Interpretation narrative
7. recommendations   → Actionable next steps
8. complete          → Final JSON {items: [{type, id, value, hidden, name}]}
```

Intermediate streaming from subagent ReAct loop enables transparency —
user sees thought process and intermediate results as the agent works.


---

## 12. Output Contract

### 12.1 Final Output Format

```json
{
  "items": [
    {"type": "text",  "id": "summary",  "value": "...", "name": "Summary"},
    {"type": "table", "id": "data_tbl", "value": {...},  "name": "Campaign Performance"},
    {"type": "chart", "id": "trend_ch", "value": {...},  "name": "CTR Trend"},
    {"type": "text",  "id": "insight",  "value": "...", "name": "Insight"},
    {"type": "text",  "id": "recom",    "value": "...", "name": "Recommendation"},
    {"type": "sql",   "id": "query",    "value": "...", "name": "SQL Query", "hidden": true}
  ]
}
```

### 12.2 Item Ownership

| Type | Source | LLM Involved? |
|---|---|---|
| text | Supervisor synthesizer | Yes (compositional narrative) |
| table | Subagent table_builder | **No** (deterministic) |
| chart | Subagent chart_builder | Yes (spec generation, data deterministic) |
| sql | Subagent tool_handler | No (pass-through from Genie) |

Custom output format structure to be provided by Nivethan — will replace
the items array with a 2D array format.


---

## 13. Multi-Tenancy & Authentication

```
Request → resolve_client(sp_id) → CLIENT_REGISTRY
    │
    ├── SecretsLoader: per-client OAuth creds from Databricks Secrets Scope
    ├── SecureTokenProvider: OAuth2 client_credentials, cached, 120s pre-expiry
    └── Token threaded: config["configurable"]["sp_token"]
        → Every Genie + LLM call uses requester's identity
        → RLS enforced via Unity Catalog (cid column filter)
```

No changes needed — this layer is production-grade today.


---

## 14. MLflow Tracing — End-to-End

Every request produces a complete trace:

```
Request Trace (root span)
│
├── supervisor_classify
│   ├── input: query + history
│   ├── output: intent classification
│   └── attributes: intent_type, complexity, target_agent
│
├── supervisor_plan (complex only)
│
├── campaign_insight_agent
│   ├── phase_1_plan
│   │   ├── dimension_classification
│   │   ├── dimension_validation
│   │   └── query_planning
│   │
│   ├── phase_2_execute
│   │   ├── step_1
│   │   │   ├── react_iteration_1
│   │   │   │   ├── genie_call (NL query, SQL, rows, latency)
│   │   │   │   ├── table_analysis (if large)
│   │   │   │   └── observe
│   │   │   └── step_complete
│   │   └── step_N ...
│   │
│   ├── phase_3_interpret
│   │   ├── metric_evaluation
│   │   ├── pattern_detection
│   │   ├── trend_analysis
│   │   └── recommendation_generation
│   │
│   ├── phase_4_reflect
│   │   ├── checks: [grounding, constraints, recommendations, completeness]
│   │   └── fixes_applied
│   │
│   └── phase_5_output
│       ├── tables_built, chart_generated
│       └── output_size_tokens
│
├── supervisor_synthesize
│
└── request_metadata
    ├── total_duration_ms, total_genie_calls, total_llm_calls
    ├── models: supervisor + subagent
    └── client_id, intent_type, confidence
```


---

## 15. Evaluation Framework

### 15.1 Domain-Specific Criteria

| Criterion | What It Checks | Source of Truth |
|---|---|---|
| Metric accuracy | Numbers match Genie response exactly | Raw Genie output vs final |
| Constraint compliance | Channel caveats applied correctly | constraints.yaml |
| Interpretation correctness | Right thresholds, right labels | interpretation.yaml |
| Recommendation grounding | Every recommendation cites data | recommendations.yaml |
| Intent classification accuracy | Correct routing | intent.yaml taxonomy |
| Caveat completeness | Limitations mentioned when applicable | constraints.yaml |
| Out-of-scope handling | Correct refusal + alternatives | constraints.yaml |
| Dimension classification | Right dimensions activated | User query keywords |

### 15.2 Offline Evaluation

MLflow Evaluation with curated test set (30+ questions from test plan).
Run against recorded Genie responses. Score each criterion. Track regression.

### 15.3 Online Evaluation

Production trace analysis: pattern detection accuracy, recommendation grounding
rate, constraint violation rate, timeout frequency. Aggregated into quality
dashboard.


---

## 16. Configuration — Externalized Knobs

```python
# Genie
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
GENIE_MAX_WORKERS = int(os.getenv("GENIE_MAX_WORKERS", "6"))
GENIE_WORKER_STAGGER_SEC = float(os.getenv("GENIE_WORKER_STAGGER_SEC", "1.0"))
GENIE_SUBMIT_RETRIES = int(os.getenv("GENIE_SUBMIT_RETRIES", "2"))
GENIE_POLL_RETRIES = int(os.getenv("GENIE_POLL_RETRIES", "3"))
GENIE_TABLE_MAX_ROWS = int(os.getenv("GENIE_TABLE_MAX_ROWS", "50"))
GENIE_TIMEOUT_SEC = int(os.getenv("GENIE_TIMEOUT_SEC", "60"))

# Memory
STM_TRIM_MESSAGES = int(os.getenv("STM_TRIM_MESSAGES", "7"))
LTM_EPISODE_SEARCH_ENABLED = os.getenv("LTM_EPISODE_SEARCH_ENABLED", "true")

# LLM
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "databricks-claude-sonnet-4-5")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "databricks-gpt-5-2")

# Dimensions
ENABLE_AUDIENCE_ANALYSIS = os.getenv("ENABLE_AUDIENCE_ANALYSIS", "true")
ENABLE_CONTENT_ANALYSIS = os.getenv("ENABLE_CONTENT_ANALYSIS", "true")
DIMENSION_QUERY_BUDGET = int(os.getenv("DIMENSION_QUERY_BUDGET", "8"))

# Subagent
MAX_ITERATIONS_PER_STEP = int(os.getenv("MAX_ITERATIONS_PER_STEP", "3"))
MAX_GENIE_RETRIES = int(os.getenv("MAX_GENIE_RETRIES", "2"))
TOTAL_TIMEOUT_SEC = int(os.getenv("TOTAL_TIMEOUT_SEC", "120"))
TABLE_ANALYZER_THRESHOLD = int(os.getenv("TABLE_ANALYZER_THRESHOLD", "20"))
```


---

## 17. Error Handling — No Silent Degradation

| Scenario | Behavior |
|---|---|
| Lakebase fails | InMemorySaver + warn user: "session memory only" |
| Genie 429 | Retry with backoff (max 2), stagger parallel calls |
| Genie FAILED | Rephrase query + retry (max 2). If still fails: partial result |
| Genie FEEDBACK_NEEDED | Surface clarification to user |
| LLM parse error | Log to trace, return structured error with available data |
| Timeout (120s) | Complete current step, skip remaining, interpret available |
| table_analyzer error | Fallback to truncated table (top 20 rows) |
| All errors | Traced as spans with error attributes |


---

## 18. Token Optimization

- **Prompt caching**: Leverage Databricks AI Gateway prompt caching for
  domain knowledge (YAML content) that stays constant across requests
- **Structured state accumulation**: Step results stored as compressed
  summaries, not raw conversation history
- **table_analyzer**: Keeps per-step token contribution to ~800-1,500
  regardless of actual table size (50 rows → same token cost as 500 rows)
- **Sliding window**: LLM receives full current step + compressed prior steps
- **Temperature 0.0** on all nodes (deterministic, no wasted tokens on variation)


---

## 19. Target Repository Structure

```
comarketer/
├── agent_server/
│   ├── agent.py                    # SLIM: ResponsesAgent wrapper + auth
│   ├── start_server.py             # FastAPI + SSE + UI mount
│   │
│   ├── core/
│   │   ├── config.py               # All externalized knobs
│   │   ├── state.py                # AgentState + CapabilityState
│   │   ├── graph.py                # Registry-driven graph builder
│   │   └── streaming.py            # SSE event emission
│   │
│   ├── auth/
│   │   ├── client_resolver.py      # resolve_client + CLIENT_REGISTRY
│   │   ├── secrets_loader.py       # Databricks Secrets Scope
│   │   └── token_provider.py       # SecureTokenProvider (OAuth2)
│   │
│   ├── memory/
│   │   ├── stm_manager.py          # Lakebase CheckpointSaver + fallback
│   │   ├── ltm_manager.py          # Profile + episode CRUD
│   │   └── memory_injector.py      # Decides what goes where
│   │
│   ├── supervisor/
│   │   ├── __init__.py
│   │   ├── supervisor_node.py
│   │   ├── intent_classifier.py
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── synthesizer.py
│   │   ├── domain_context.py
│   │   └── prompts/
│   │       └── supervisor_prompt.py
│   │
│   ├── agents/
│   │   ├── registry.py             # AGENT_REGISTRY
│   │   ├── base_agent.py           # Abstract base class
│   │   │
│   │   ├── campaign_insight/       # Fat subagent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── dimension_classifier.py
│   │   │   ├── dimension_validator.py
│   │   │   ├── adaptive_planner.py
│   │   │   ├── executor.py
│   │   │   ├── tool_handler.py
│   │   │   ├── table_analyzer.py
│   │   │   ├── table_builder.py
│   │   │   ├── interpreter.py
│   │   │   ├── recommender.py
│   │   │   ├── reflector.py
│   │   │   ├── chart_builder.py
│   │   │   ├── output_builder.py
│   │   │   ├── domain_knowledge.py
│   │   │   ├── prompts/
│   │   │   │   ├── insight_prompt.py
│   │   │   │   ├── reasoning_prompt.py
│   │   │   │   ├── interpretation_prompt.py
│   │   │   │   └── reflection_prompt.py
│   │   │   └── domain_knowledge/
│   │   │       ├── metrics.yaml
│   │   │       ├── interpretation.yaml
│   │   │       ├── constraints.yaml
│   │   │       ├── recommendations.yaml
│   │   │       ├── intent.yaml
│   │   │       ├── domain_context.yaml
│   │   │       ├── audience_knowledge.yaml   # TODO
│   │   │       └── content_knowledge.yaml    # TODO
│   │   │
│   │   ├── greeting.py             # Lightweight — no domain knowledge
│   │   └── clarification.py        # Lightweight — STM only
│   │
│   ├── tools/
│   │   ├── registry.py             # Tool registration
│   │   ├── genie_client.py         # httpx Genie REST wrapper
│   │   └── genie_rate_limiter.py   # Stagger + 429 retry
│   │
│   ├── parsers/
│   │   └── table_truncation.py     # LLM budget truncation
│   │
│   ├── ui/                         # Flask UI at /ui/
│   │
│   └── utils/
│       ├── tracing.py              # MLflow span helpers
│       └── formatting.py           # Response formatting
│
├── bundles/                        # Databricks Asset Bundle config
├── docs/
│   └── IMPLEMENTATION_VISION.md    # This document
├── databricks.yml
└── CLAUDE.md                       # <200 lines, project rules for AI assistants
```


---

## 20. What Gets Deleted from Current Codebase

| File | LOC | Action | Reason |
|---|---|---|---|
| `agents/agentbricks.py` | 479 | **Delete** | Dead code, superseded by Genie |
| `agents/registry.py` (current) | 32 | **Replace** | Stub that's never consulted |
| `feedback.py` | ~30 | **Keep, wire up** | Needs endpoint + UI hookup |
| Orphaned helpers | ~100 | **Delete** | `_json_decode_attr`, `_get_text`, etc. |
| `agent.py` (current 700 LOC) | 700 | **Decompose** | Split into auth/ + core/ + supervisor/ |
| `supervisor_prompt.py` (700 LOC) | 700 | **Decompose** | Split: ~150 lines supervisor + domain YAMLs |


---

## 21. Design Decisions Summary

| # | Decision | Rationale |
|---|---|---|
| 1 | Supervisor = thin orchestrator with broad domain awareness | Smart routing + synthesis without domain logic duplication |
| 2 | Subagent = fat domain expert with internal modules | Self-contained, testable, extensible |
| 3 | All data through Genie metric views, never LLM | 100% data accuracy guarantee |
| 4 | Single Genie Space for all 3 dimensions | Simpler tool interface, one space ID |
| 5 | Metric views flatten JSONB | Agent receives clean columns, no parsing |
| 6 | Domain knowledge in 8 YAML files (6 exist + 2 needed) | Versionable, testable, swappable per client |
| 7 | Memory injected into subagents, not supervisor | Keeps supervisor clean |
| 8 | Structured contracts everywhere | Eliminates free-text coupling |
| 9 | Registry-driven graph construction | Adding agents = config change |
| 10 | 3-Dimension Classification with deterministic validation | Prevents LLM hallucinating dimension relevance |
| 11 | table_analyzer for large tables | Preserves accuracy + manages context |
| 12 | Config-based tools, prompts, subagents | Scalable without code changes |
| 13 | No silent degradation | Every fallback is observable + user-communicated |
| 14 | AP → ReAct → Reflect for subagent | Best fit for long-running, multi-step, verified analysis |
| 15 | Metric view latency acceptable | Agent streams to user, Genie compilation hidden |


---

## 22. Open Items / Next Iteration

| # | Topic | Status | Next Step |
|---|---|---|---|
| 1 | **audience_knowledge.yaml** | ❌ Not created | Define audience tag interpretation rules, intent lift thresholds, nudge triggers |
| 2 | **content_knowledge.yaml** | ❌ Not created | Define content score interpretation, benchmark comparison rules, feature definitions |
| 3 | **domain_context.yaml data_source** | ⚠️ Outdated | Add audience_metric_view, campaign_content_metric_view, igp_content_insights to data_source section |
| 4 | **intent.yaml audience/content intents** | ⚠️ Gap | Add audience_composition, audience_engagement, content_quality, content_benchmark intents |
| 5 | Custom output format (2D array) | Structure TBD | Nivethan to provide example |
| 6 | Memory policy | Architecture defined, policy TBD | Define what/when/decay rules |
| 7 | Genie Space cleanup | 5 duplicate SQL expressions | Merge "when to use" into YAML comments, remove duplicates |
| 8 | Prompt caching | Strategy identified | Test with Databricks AI Gateway |
| 9 | Evaluation harness | Criteria defined, harness TBD | Build MLflow Evaluation pipeline |
| 10 | Genie Space testing | 30-question test plan ready | Execute and validate 80% accuracy |
| 11 | Entity matching | Not enabled | Enable for categorical columns in Genie Space |
| 12 | PK/FK RELY constraints | Not set | Set on underlying tables for query optimizer |
| 13 | Claude Code implementation plan | Deferred | Session-by-session execution strategy |
| 14 | Prompt engineering skill | Created | Apply Anthropic best practices to all prompts |
