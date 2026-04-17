"""Agent registry — single config for all agents.

Adding a new agent = one entry here + one handler file.
The graph builder reads this to wire nodes.
"""

AGENT_REGISTRY = {
    "supervisor": {
        "handler": "agents.supervisor.supervisor_node",
        "description": "Intent classification + Command routing",
        "llm_model": "databricks-gpt-5-2",
        "timeout_ms": 5000,
    },
    "greeting": {
        "handler": "agents.greeting.greeting_node",
        "description": "Handles greetings and general messages",
        "llm_model": None,
        "timeout_ms": 100,
    },
    "genie_data": {
        "handler": "agents.genie_agent.genie_data_node",
        "description": "Genie API call + table build (no LLM) [simple path]",
        "llm_model": None,
        "timeout_ms": 120000,
    },
    "genie_analysis": {
        "handler": "agents.genie_analysis.genie_analysis_node",
        "description": "LLM analysis of Genie query results [simple path]",
        "llm_model": "databricks-gpt-5-2",
        "timeout_ms": 30000,
    },
    "insight_agent": {
        "handler": "capabilities.insight_agent.insight_agent_node",
        "description": "ReAct subgraph for campaign analytics (replaces genie_worker for complex queries)",
        "llm_model": "databricks-gpt-5-2",
        "timeout_ms": 180000,
    },
}
