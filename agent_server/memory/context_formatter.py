"""Format LTM data into supervisor prompt context.

Pure functions — no store access, no side effects.
Called after LTM read, before supervisor graph invocation.
"""
from typing import Dict, List


def format_ltm_context(profile: Dict, episodes: List[Dict] = None) -> str:
    """Convert client profile + episodes into supervisor prompt section.

    Returns empty string if no meaningful data exists.
    Gets injected into custom_inputs.ltm_context, which
    create_supervisor_with_context() appends to the prompt.

    Output format:
        **Frequently discussed channels:** WhatsApp (42), Email (38)
        **Frequently discussed metrics:** revenue (35), click_rate (28)
        **Recent focus:** Compare WhatsApp vs Email revenue Dec 2025

        **Relevant past findings:**
        - Email unsubs: Split testing reduces 53.5%
    """
    parts = []

    # Channel frequency (top 3)
    ch_freq = profile.get("channel_frequency", {})
    if ch_freq:
        top_channels = sorted(ch_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        ch_str = ", ".join(f"{ch} ({count})" for ch, count in top_channels)
        parts.append(f"**Frequently discussed channels:** {ch_str}")

    # Metric frequency (top 3)
    m_freq = profile.get("metric_frequency", {})
    if m_freq:
        top_metrics = sorted(m_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        m_str = ", ".join(f"{m} ({count})" for m, count in top_metrics)
        parts.append(f"**Frequently discussed metrics:** {m_str}")

    # Recent queries (last 2 for brevity in prompt)
    recent = profile.get("recent_queries", [])
    if recent:
        recent_str = "; ".join(recent[-2:])
        parts.append(f"**Recent focus:** {recent_str}")

    # Episodes (past findings from semantic search)
    if episodes:
        findings = []
        for ep in episodes[:3]:
            finding = ep.get("finding", "")
            if finding:
                findings.append(f"- {finding}")
        if findings:
            parts.append("**Relevant past findings:**\n" + "\n".join(findings))

    return "\n".join(parts) if parts else ""


def format_greeting_context(profile: Dict) -> Dict:
    """Extract greeting personalization data from profile.

    Returns dict with keys used by the greeting builder:
        {
            "top_channels": ["whatsapp", "email"],
            "recent_query": "Compare WhatsApp vs Email...",
            "total_queries": 147,
        }
    """
    ch_freq = profile.get("channel_frequency", {})
    top_channels = sorted(
        ch_freq.keys(), key=lambda x: ch_freq[x], reverse=True
    )[:2] if ch_freq else []

    recent = profile.get("recent_queries", [])
    recent_query = recent[-1] if recent else None

    return {
        "top_channels": top_channels,
        "recent_query": recent_query,
        "total_queries": profile.get("total_queries", 0),
    }
