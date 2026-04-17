"""Entity and insight extraction for LTM writes.

All functions are pure — no side effects, no store access.
Called post-yield in agent.py to populate LTM.
"""
import re
import json
import logging
from typing import Dict, List

from memory.constants import KNOWN_METRICS, METRIC_CANONICAL, CHANNEL_NAMES

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns (compile once at module load)
_RE_MONTH_FULL = re.compile(
    r'(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december)\s*\d{4}'
)
_RE_MONTH_SHORT = re.compile(
    r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}'
)
_RE_QUARTER = re.compile(r'q[1-4]\s*\d{4}')
_RE_RELATIVE_TIME = re.compile(
    r'(?:last\s+(?:week|month|quarter|year)|this\s+(?:week|month|quarter|year)|'
    r'yesterday|today|past\s+\d+\s+days)'
)


def extract_entities_from_query(query: str) -> Dict:
    """Extract channels, metrics, and time periods from user query.

    Returns:
        {
            "channels": ["whatsapp", "email"],
            "metrics": ["revenue", "click_rate"],
            "time_periods": ["december 2025", "last month"],
        }
    """
    query_lower = query.lower()

    channels = [ch for ch in CHANNEL_NAMES if ch in query_lower]
    metrics = extract_metrics_from_query(query_lower)

    time_periods = []
    time_periods.extend(_RE_MONTH_FULL.findall(query_lower))
    time_periods.extend(_RE_MONTH_SHORT.findall(query_lower))
    time_periods.extend(_RE_QUARTER.findall(query_lower))
    time_periods.extend(_RE_RELATIVE_TIME.findall(query_lower))

    return {
        "channels": channels,
        "metrics": metrics,
        "time_periods": time_periods,
    }


def extract_metrics_from_query(query_lower: str) -> List[str]:
    """Extract mentioned metrics and normalize to canonical names.

    "show me click rate and revenue" -> ["click_rate", "revenue"]
    "unsub trends" -> ["unsubscribe_rate"]
    """
    found = []
    for metric in KNOWN_METRICS:
        if metric in query_lower:
            canonical = METRIC_CANONICAL.get(metric, metric.replace(" ", "_"))
            if canonical not in found:
                found.append(canonical)
    return found


def extract_insights_from_response(response_content: str) -> List[Dict]:
    """Extract key findings from supervisor JSON response for episodic memory.

    Looks for "What Happened:" lines in text items of the structured response.
    Returns list of finding dicts ready for store.put().
    """
    insights = []
    if not response_content:
        return insights

    try:
        if response_content.strip().startswith('{'):
            data = json.loads(response_content)
            items = data.get("items", [])
            for item in items:
                if item.get("type") == "text":
                    text = item.get("value", "")
                    if "What Happened:" in text or "What Did We Find?" in text:
                        for line in text.split("\n"):
                            line_stripped = line.strip()
                            if line_stripped.startswith("**What Happened:**"):
                                finding = line_stripped.replace("**What Happened:**", "").strip()
                                if finding and len(finding) > 10:
                                    insights.append({
                                        "finding": finding[:500],
                                        "type": "key_finding",
                                    })
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"EXTRACTOR: Could not extract insights: {e}")

    return insights
