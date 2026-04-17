"""Import-level smoke tests.

These verify that the core modules can be imported and the LangGraph
topology compiles. They do not hit Databricks or require SP credentials,
so they are safe to run in CI and as a Phase B cutover gate.

The previous integration test (`from agent import non_streaming`) was
stale — `non_streaming` was never defined on the new `CoMarketerAgent`
ResponsesAgent subclass. Replace with lightweight checks that catch
the breakage pattern we actually care about during the migration:
import-time failures after dead-code deletion.
"""

import sys

sys.path.insert(0, ".")


def test_agent_module_imports():
    """The root agent module must import and expose the singleton."""
    import agent

    assert hasattr(agent, "CoMarketerAgent")
    assert hasattr(agent, "agent")
    assert agent.agent.__class__.__name__ == "CoMarketerAgent"


def test_graph_compiles():
    """The compiled LangGraph must build without errors."""
    from core.graph import get_compiled_graph

    compiled = get_compiled_graph()
    assert compiled is not None
    nodes = set(compiled.get_graph().nodes.keys())
    expected = {
        "supervisor_classify",
        "greeting",
        "clarification",
        "out_of_scope",
        "campaign_insight_agent",
        "supervisor_synthesize",
    }
    assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"


def test_is_greeting_importable():
    """Greeting detector must be reachable from supervisor.intent_classifier.

    This is the canonical location after the Phase B repoint. Kept as
    an explicit check so the repoint cannot silently regress.
    """
    from supervisor.intent_classifier import _is_greeting

    assert _is_greeting("hi") is True
    assert _is_greeting("how is my email CTR trending last week") is False
